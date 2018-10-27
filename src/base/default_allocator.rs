//! The default matrix data storage allocator.
//!
//! This will use stack-allocated buffers for matrices with dimensions known at compile-time, and
//! heap-allocated buffers for matrices with at least one dimension unknown at compile-time.

use std::cmp;
use std::mem;
use std::ops::Mul;
use std::ptr;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use generic_array::ArrayLength;
use typenum::Prod;

use base::allocator::{Allocator, Reallocator};
#[cfg(any(feature = "alloc", feature = "std"))]
use base::dimension::Dynamic;
use base::dimension::{Dim, DimName};
use base::matrix_array::MatrixArray;
#[cfg(any(feature = "std", feature = "alloc"))]
use base::matrix_vec::MatrixVec;
use base::storage::{Storage, StorageMut};
use base::Scalar;

/*
 *
 * Allocator.
 *
 */
/// An allocator based on `GenericArray` and `MatrixVec` for statically-sized and dynamically-sized
/// matrices respectively.
pub struct DefaultAllocator;

// Static - Static
impl<N, R, C> Allocator<N, R, C> for DefaultAllocator
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Buffer = MatrixArray<N, R, C>;

    #[inline]
    unsafe fn allocate_uninitialized(_: R, _: C) -> Self::Buffer {
        mem::uninitialized()
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer
    {
        let mut res = unsafe { Self::allocate_uninitialized(nrows, ncols) };
        let mut count = 0;

        for (res, e) in res.iter_mut().zip(iter.into_iter()) {
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
    type Buffer = MatrixVec<N, Dynamic, C>;

    #[inline]
    unsafe fn allocate_uninitialized(nrows: Dynamic, ncols: C) -> Self::Buffer {
        let mut res = Vec::new();
        let length = nrows.value() * ncols.value();
        res.reserve_exact(length);
        res.set_len(length);

        MatrixVec::new(nrows, ncols, res)
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: Dynamic,
        ncols: C,
        iter: I,
    ) -> Self::Buffer
    {
        let it = iter.into_iter();
        let res: Vec<N> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        MatrixVec::new(nrows, ncols, res)
    }
}

// Static - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, R: DimName> Allocator<N, R, Dynamic> for DefaultAllocator {
    type Buffer = MatrixVec<N, R, Dynamic>;

    #[inline]
    unsafe fn allocate_uninitialized(nrows: R, ncols: Dynamic) -> Self::Buffer {
        let mut res = Vec::new();
        let length = nrows.value() * ncols.value();
        res.reserve_exact(length);
        res.set_len(length);

        MatrixVec::new(nrows, ncols, res)
    }

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(
        nrows: R,
        ncols: Dynamic,
        iter: I,
    ) -> Self::Buffer
    {
        let it = iter.into_iter();
        let res: Vec<N> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        MatrixVec::new(nrows, ncols, res)
    }
}

/*
 *
 * Reallocator.
 *
 */
// Anything -> Static × Static
impl<N: Scalar, RFrom, CFrom, RTo, CTo> Reallocator<N, RFrom, CFrom, RTo, CTo> for DefaultAllocator
where
    RFrom: Dim,
    CFrom: Dim,
    RTo: DimName,
    CTo: DimName,
    Self: Allocator<N, RFrom, CFrom>,
    RTo::Value: Mul<CTo::Value>,
    Prod<RTo::Value, CTo::Value>: ArrayLength<N>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: CTo,
        buf: <Self as Allocator<N, RFrom, CFrom>>::Buffer,
    ) -> MatrixArray<N, RTo, CTo>
    {
        let mut res = <Self as Allocator<N, RTo, CTo>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(buf.ptr(), res.ptr_mut(), cmp::min(len_from, len_to));

        res
    }
}

// Static × Static -> Dynamic × Any
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, RFrom, CFrom, CTo> Reallocator<N, RFrom, CFrom, Dynamic, CTo> for DefaultAllocator
where
    RFrom: DimName,
    CFrom: DimName,
    CTo: Dim,
    RFrom::Value: Mul<CFrom::Value>,
    Prod<RFrom::Value, CFrom::Value>: ArrayLength<N>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: MatrixArray<N, RFrom, CFrom>,
    ) -> MatrixVec<N, Dynamic, CTo>
    {
        let mut res = <Self as Allocator<N, Dynamic, CTo>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(buf.ptr(), res.ptr_mut(), cmp::min(len_from, len_to));

        res
    }
}

// Static × Static -> Static × Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, RFrom, CFrom, RTo> Reallocator<N, RFrom, CFrom, RTo, Dynamic> for DefaultAllocator
where
    RFrom: DimName,
    CFrom: DimName,
    RTo: DimName,
    RFrom::Value: Mul<CFrom::Value>,
    Prod<RFrom::Value, CFrom::Value>: ArrayLength<N>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: MatrixArray<N, RFrom, CFrom>,
    ) -> MatrixVec<N, RTo, Dynamic>
    {
        let mut res = <Self as Allocator<N, RTo, Dynamic>>::allocate_uninitialized(rto, cto);

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
        buf: MatrixVec<N, Dynamic, CFrom>,
    ) -> MatrixVec<N, Dynamic, CTo>
    {
        let new_buf = buf.resize(rto.value() * cto.value());
        MatrixVec::new(rto, cto, new_buf)
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
        buf: MatrixVec<N, Dynamic, CFrom>,
    ) -> MatrixVec<N, RTo, Dynamic>
    {
        let new_buf = buf.resize(rto.value() * cto.value());
        MatrixVec::new(rto, cto, new_buf)
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
        buf: MatrixVec<N, RFrom, Dynamic>,
    ) -> MatrixVec<N, Dynamic, CTo>
    {
        let new_buf = buf.resize(rto.value() * cto.value());
        MatrixVec::new(rto, cto, new_buf)
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
        buf: MatrixVec<N, RFrom, Dynamic>,
    ) -> MatrixVec<N, RTo, Dynamic>
    {
        let new_buf = buf.resize(rto.value() * cto.value());
        MatrixVec::new(rto, cto, new_buf)
    }
}
