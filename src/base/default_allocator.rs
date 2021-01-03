//! The default matrix data storage allocator.
//!
//! This will use stack-allocated buffers for matrices with dimensions known at compile-time, and
//! heap-allocated buffers for matrices with at least one dimension unknown at compile-time.

use std::cmp;
use std::mem;
use std::ptr;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::Const;
use crate::base::allocator::{Allocator, Reallocator};
use crate::base::array_storage::ArrayStorage;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::base::dimension::Dynamic;
use crate::base::dimension::{Dim, DimName};
use crate::base::storage::{Storage, StorageMut};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::vec_storage::VecStorage;
use crate::base::Scalar;

/*
 *
 * Allocator.
 *
 */
/// An allocator based on `GenericArray` and `VecStorage` for statically-sized and dynamically-sized
/// matrices respectively.
pub struct DefaultAllocator;

// Static - Static
impl<N: Scalar, const R: usize, const C: usize> Allocator<N, Const<R>, Const<C>>
    for DefaultAllocator
{
    type Buffer = ArrayStorage<N, R, C>;

    #[inline]
    unsafe fn allocate_uninitialized(_: R, _: C) -> mem::MaybeUninit<Self::Buffer> {
        mem::MaybeUninit::<Self::Buffer>::uninit()
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: Const<R>,
        ncols: Const<C>,
        iter: I,
    ) -> Self::Buffer {
        #[cfg(feature = "no_unsound_assume_init")]
        let mut res: Self::Buffer = unimplemented!();
        #[cfg(not(feature = "no_unsound_assume_init"))]
        let mut res = unsafe { Self::allocate_uninitialized(nrows, ncols).assume_init() };
        let mut count = 0;

        for (res, e) in res.as_mut_slice().iter_mut().zip(iter.into_iter()) {
            *res = e;
            count += 1;
        }

        assert!(
            count == nrows.value() * ncols.value(),
            "Matrix init. from iterator: iterator not long enough."
        );

        res
    }
}

// Dynamic - Static
// Dynamic - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, C: Dim> Allocator<N, Dynamic, C> for DefaultAllocator {
    type Buffer = VecStorage<N, Dynamic, C>;

    #[inline]
    unsafe fn allocate_uninitialized(nrows: Dynamic, ncols: C) -> mem::MaybeUninit<Self::Buffer> {
        let mut res = Vec::new();
        let length = nrows.value() * ncols.value();
        res.reserve_exact(length);
        res.set_len(length);

        mem::MaybeUninit::new(VecStorage::new(nrows, ncols, res))
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: Dynamic,
        ncols: C,
        iter: I,
    ) -> Self::Buffer {
        let it = iter.into_iter();
        let res: Vec<N> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

// Static - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, R: DimName> Allocator<N, R, Dynamic> for DefaultAllocator {
    type Buffer = VecStorage<N, R, Dynamic>;

    #[inline]
    unsafe fn allocate_uninitialized(nrows: R, ncols: Dynamic) -> mem::MaybeUninit<Self::Buffer> {
        let mut res = Vec::new();
        let length = nrows.value() * ncols.value();
        res.reserve_exact(length);
        res.set_len(length);

        mem::MaybeUninit::new(VecStorage::new(nrows, ncols, res))
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: R,
        ncols: Dynamic,
        iter: I,
    ) -> Self::Buffer {
        let it = iter.into_iter();
        let res: Vec<N> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

/*
 *
 * Reallocator.
 *
 */
// Anything -> Static × Static
impl<N: Scalar, RFrom, CFrom, const RTO: usize, const CTO: usize>
    Reallocator<N, RFrom, CFrom, Const<RTO>, Const<CTO>> for DefaultAllocator
where
    RFrom: Dim,
    CFrom: Dim,
    Self: Allocator<N, RFrom, CFrom>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Const<RTO>,
        cto: Const<CTO>,
        buf: <Self as Allocator<N, RFrom, CFrom>>::Buffer,
    ) -> ArrayStorage<N, RTo, CTo> {
        #[cfg(feature = "no_unsound_assume_init")]
        let mut res: ArrayStorage<N, RTo, CTo> = unimplemented!();
        #[cfg(not(feature = "no_unsound_assume_init"))]
        let mut res =
            <Self as Allocator<N, RTo, CTo>>::allocate_uninitialized(rto, cto).assume_init();

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(buf.ptr(), res.ptr_mut(), cmp::min(len_from, len_to));

        res
    }
}

// Static × Static -> Dynamic × Any
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, CTo, const RFROM: usize, const CFROM: usize>
    Reallocator<N, Const<RFROM>, Const<CFROM>, Dynamic, CTo> for DefaultAllocator
where
    CTo: Dim,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: ArrayStorage<N, RFROM, CFROM>,
    ) -> VecStorage<N, Dynamic, CTo> {
        #[cfg(feature = "no_unsound_assume_init")]
        let mut res: VecStorage<N, Dynamic, CTo> = unimplemented!();
        #[cfg(not(feature = "no_unsound_assume_init"))]
        let mut res =
            <Self as Allocator<N, Dynamic, CTo>>::allocate_uninitialized(rto, cto).assume_init();

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(buf.ptr(), res.ptr_mut(), cmp::min(len_from, len_to));

        res
    }
}

// Static × Static -> Static × Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, RTo, const RFROM: usize, const CFROM: usize>
    Reallocator<N, Const<RFROM>, Const<CFROM>, RTo, Dynamic> for DefaultAllocator
where
    RTo: DimName,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: ArrayStorage<N, RFROM, CFROM>,
    ) -> VecStorage<N, RTo, Dynamic> {
        #[cfg(feature = "no_unsound_assume_init")]
        let mut res: VecStorage<N, RTo, Dynamic> = unimplemented!();
        #[cfg(not(feature = "no_unsound_assume_init"))]
        let mut res =
            <Self as Allocator<N, RTo, Dynamic>>::allocate_uninitialized(rto, cto).assume_init();

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(buf.ptr(), res.ptr_mut(), cmp::min(len_from, len_to));

        res
    }
}

// All conversion from a dynamic buffer to a dynamic buffer.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, CFrom: Dim, CTo: Dim> Reallocator<N, Dynamic, CFrom, Dynamic, CTo>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<N, Dynamic, CFrom>,
    ) -> VecStorage<N, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, CFrom: Dim, RTo: DimName> Reallocator<N, Dynamic, CFrom, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<N, Dynamic, CFrom>,
    ) -> VecStorage<N, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, RFrom: DimName, CTo: Dim> Reallocator<N, RFrom, Dynamic, Dynamic, CTo>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<N, RFrom, Dynamic>,
    ) -> VecStorage<N, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, RFrom: DimName, RTo: DimName> Reallocator<N, RFrom, Dynamic, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<N, RFrom, Dynamic>,
    ) -> VecStorage<N, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}
