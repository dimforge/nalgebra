//! The default matrix data storage allocator.
//!
//! This will use stack-allocated buffers for matrices with dimensions known at compile-time, and
//! heap-allocated buffers for matrices with at least one dimension unknown at compile-time.

use std::cmp;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::ptr;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::Const;
use crate::base::allocator::{Allocator, InnerAllocator, Reallocator};
use crate::base::array_storage::ArrayStorage;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::base::dimension::Dynamic;
use crate::base::dimension::{Dim, DimName};
use crate::base::storage::{ContiguousStorageMut, Storage, StorageMut};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::vec_storage::VecStorage;
use crate::storage::Owned;

type DefaultBuffer<T, R, C> = <DefaultAllocator as InnerAllocator<T, R, C>>::Buffer;
type DefaultUninitBuffer<T, R, C> =
    <DefaultAllocator as InnerAllocator<MaybeUninit<T>, R, C>>::Buffer;

/*
 *
 * Allocator.
 *
 */
/// A helper struct that controls how the storage for a matrix should be allocated.
///
/// This struct is useless on its own. Instead, it's used in trait
/// An allocator based on `GenericArray` and `VecStorage` for statically-sized and dynamically-sized
/// matrices respectively.
pub struct DefaultAllocator;

// Static - Static
impl<T, const R: usize, const C: usize> InnerAllocator<T, Const<R>, Const<C>> for DefaultAllocator {
    type Buffer = ArrayStorage<T, R, C>;

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

impl<T, const R: usize, const C: usize> Allocator<T, Const<R>, Const<C>> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(
        _: Const<R>,
        _: Const<C>,
    ) -> Owned<MaybeUninit<T>, Const<R>, Const<C>> {
        // SAFETY: An uninitialized `[MaybeUninit<_>; LEN]` is valid.
        let array = unsafe { MaybeUninit::uninit().assume_init() };
        ArrayStorage(array)
    }

    #[inline]
    unsafe fn assume_init(
        uninit: <Self as InnerAllocator<MaybeUninit<T>, Const<R>, Const<C>>>::Buffer,
    ) -> Owned<T, Const<R>, Const<C>> {
        // SAFETY:
        // * The caller guarantees that all elements of the array are initialized
        // * `MaybeUninit<T>` and T are guaranteed to have the same layout
        // * MaybeUnint does not drop, so there are no double-frees
        // * `ArrayStorage` is transparent.
        // And thus the conversion is safe
        ArrayStorage((&uninit as *const _ as *const [_; C]).read())
    }
}

// Dynamic - Static
// Dynamic - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, C: Dim> InnerAllocator<T, Dynamic, C> for DefaultAllocator {
    type Buffer = VecStorage<T, Dynamic, C>;

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

impl<T, C: Dim> Allocator<T, Dynamic, C> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(nrows: Dynamic, ncols: C) -> Owned<MaybeUninit<T>, Dynamic, C> {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(uninit: Owned<MaybeUninit<T>, Dynamic, C>) -> Owned<T, Dynamic, C> {
        let mut data = ManuallyDrop::new(uninit.data);

        // Safety: MaybeUninit<T> has the same alignment and layout as T.
        let new_data =
            Vec::from_raw_parts(data.as_mut_ptr() as *mut T, data.len(), data.capacity());

        VecStorage::new(uninit.nrows, uninit.ncols, new_data)
    }
}

// Static - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, R: DimName> InnerAllocator<T, R, Dynamic> for DefaultAllocator {
    type Buffer = VecStorage<T, R, Dynamic>;

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: Dynamic,
        iter: I,
    ) -> Owned<T, R, Dynamic> {
        let it = iter.into_iter();
        let res: Vec<T> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

impl<T, R: DimName> Allocator<T, R, Dynamic> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(nrows: R, ncols: Dynamic) -> Owned<MaybeUninit<T>, R, Dynamic> {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(uninit: Owned<MaybeUninit<T>, R, Dynamic>) -> Owned<T, R, Dynamic> {
        let mut data = ManuallyDrop::new(uninit.data);

        // Safety: MaybeUninit<T> has the same alignment and layout as T.
        let new_data =
            Vec::from_raw_parts(data.as_mut_ptr() as *mut T, data.len(), data.capacity());

        VecStorage::new(uninit.nrows, uninit.ncols, new_data)
    }
}

/*
 *
 * Reallocator.
 *
 */
// Anything -> Static × Static
impl<T, RFrom: Dim, CFrom: Dim, const RTO: usize, const CTO: usize>
    Reallocator<T, RFrom, CFrom, Const<RTO>, Const<CTO>> for DefaultAllocator
where
    Self: Allocator<T, RFrom, CFrom>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Const<RTO>,
        cto: Const<CTO>,
        buf: Owned<T, RFrom, CFrom>,
    ) -> ArrayStorage<T, RTO, CTO> {
        let mut res =
            <Self as Allocator<_, Const<RTO>, Const<CTO>>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        // Safety: TODO
        <Self as Allocator<_, Const<RTO>, Const<CTO>>>::assume_init(res)
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
