//! The default matrix data storage allocator.
//!
//! This will use stack-allocated buffers for matrices with dimensions known at compile-time, and
//! heap-allocated buffers for matrices with at least one dimension unknown at compile-time.

use std::cmp;
use std::mem;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::ptr;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::Const;
use crate::base::allocator::{Allocator, Reallocator};
use crate::base::array_storage::ArrayStorage;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::base::dimension::Dynamic;
use crate::base::dimension::{Dim, DimName};
use crate::base::storage::{ContiguousStorageMut, Storage, StorageMut};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::vec_storage::VecStorage;

/*
 *
 * Allocator.
 *
 */
/// An allocator based on `GenericArray` and `VecStorage` for statically-sized and dynamically-sized
/// matrices respectively.
pub struct DefaultAllocator;

// Static - Static
impl<T, const R: usize, const C: usize> Allocator<T, Const<R>, Const<C>> for DefaultAllocator {
    type Buffer = ArrayStorage<T, R, C>;
    type UninitBuffer = ArrayStorage<MaybeUninit<T>, R, C>;

    #[inline]
    fn allocate_uninitialized(_: Const<R>, _: Const<C>) -> Self::UninitBuffer {
        ArrayStorage([[MaybeUninit::uninit(); R]; C])
    }

    #[inline]
    unsafe fn assume_init(uninit: Self::UninitBuffer) -> Self::Buffer {
        // Safety: MaybeUninit<T> has the same alignment and layout as T, and by
        // extension so do arrays based on these.
        mem::transmute(uninit)
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: Const<R>,
        ncols: Const<C>,
        iter: I,
    ) -> Self::Buffer {
        let mut res = Self::allocate_uninitialized(nrows, ncols);
        let mut count = 0;

        for (res, e) in res.as_mut_slice().iter_mut().zip(iter.into_iter()) {
            *res = MaybeUninit::new(e);
            count += 1;
        }

        assert!(
            count == nrows.value() * ncols.value(),
            "Matrix init. from iterator: iterator not long enough."
        );

        // Safety: we have initialized all entries.
        unsafe { <Self as Allocator<T, Const<R>, Const<C>>>::assume_init(res) }
    }
}

// Dynamic - Static
// Dynamic - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, C: Dim> Allocator<T, Dynamic, C> for DefaultAllocator {
    type Buffer = VecStorage<T, Dynamic, C>;
    type UninitBuffer = VecStorage<MaybeUninit<T>, Dynamic, C>;

    #[inline]
    fn allocate_uninitialized(nrows: Dynamic, ncols: C) -> Self::UninitBuffer {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(uninit: Self::UninitBuffer) -> Self::Buffer {
        let mut data = ManuallyDrop::new(uninit.data);

        // Safety: MaybeUninit<T> has the same alignment and layout as T.
        let new_data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr() as *mut T, data.len(), data.capacity())
        };

        VecStorage::new(uninit.nrows, uninit.ncols, new_data)
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: Dynamic,
        ncols: C,
        iter: I,
    ) -> Self::Buffer {
        let it = iter.into_iter();
        let res: Vec<T> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

// Static - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, R: DimName> Allocator<T, R, Dynamic> for DefaultAllocator {
    type Buffer = VecStorage<T, R, Dynamic>;
    type UninitBuffer = VecStorage<MaybeUninit<T>, R, Dynamic>;

    #[inline]
    fn allocate_uninitialized(nrows: R, ncols: Dynamic) -> Self::UninitBuffer {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(uninit: Self::UninitBuffer) -> Self::Buffer {
        let mut data = ManuallyDrop::new(uninit.data);

        // Safety: MaybeUninit<T> has the same alignment and layout as T.
        let new_data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr() as *mut T, data.len(), data.capacity())
        };

        VecStorage::new(uninit.nrows, uninit.ncols, new_data)
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: Dynamic,
        iter: I,
    ) -> Self::Buffer {
        let it = iter.into_iter();
        let res: Vec<T> = it.collect();
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
impl<T, RFrom, CFrom, const RTO: usize, const CTO: usize>
    Reallocator<T, RFrom, CFrom, Const<RTO>, Const<CTO>> for DefaultAllocator
where
    RFrom: Dim,
    CFrom: Dim,
    Self: Allocator<T, RFrom, CFrom>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Const<RTO>,
        cto: Const<CTO>,
        buf: <Self as Allocator<T, RFrom, CFrom>>::Buffer,
    ) -> ArrayStorage<T, RTO, CTO> {
        let mut res =
            <Self as Allocator<T, Const<RTO>, Const<CTO>>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        // Safety: TODO
        <Self as Allocator<T, Const<RTO>, Const<CTO>>>::assume_init(res)
    }
}

// Static × Static -> Dynamic × Any
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CTo, const RFROM: usize, const CFROM: usize>
    Reallocator<T, Const<RFROM>, Const<CFROM>, Dynamic, CTo> for DefaultAllocator
where
    CTo: Dim,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: ArrayStorage<T, RFROM, CFROM>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let mut res = <Self as Allocator<T, Dynamic, CTo>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        <Self as Allocator<T, Dynamic, CTo>>::assume_init(res)
    }
}

// Static × Static -> Static × Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RTo, const RFROM: usize, const CFROM: usize>
    Reallocator<T, Const<RFROM>, Const<CFROM>, RTo, Dynamic> for DefaultAllocator
where
    RTo: DimName,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: ArrayStorage<T, RFROM, CFROM>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let mut res = <Self as Allocator<T, RTo, Dynamic>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        <Self as Allocator<T, RTo, Dynamic>>::assume_init(res)
    }
}

// All conversion from a dynamic buffer to a dynamic buffer.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CFrom: Dim, CTo: Dim> Reallocator<T, Dynamic, CFrom, Dynamic, CTo> for DefaultAllocator {
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<T, Dynamic, CFrom>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CFrom: Dim, RTo: DimName> Reallocator<T, Dynamic, CFrom, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<T, Dynamic, CFrom>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RFrom: DimName, CTo: Dim> Reallocator<T, RFrom, Dynamic, Dynamic, CTo>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<T, RFrom, Dynamic>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RFrom: DimName, RTo: DimName> Reallocator<T, RFrom, Dynamic, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<T, RFrom, Dynamic>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}
