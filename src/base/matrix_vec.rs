#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::ops::Deref;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use base::allocator::Allocator;
use base::default_allocator::DefaultAllocator;
use base::dimension::{Dim, DimName, Dynamic, U1};
use base::storage::{ContiguousStorage, ContiguousStorageMut, Owned, Storage, StorageMut};
use base::Scalar;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

/*
 *
 * Storage.
 *
 */
/// A Vec-based matrix data storage. It may be dynamically-sized.
#[repr(C)]
#[derive(Eq, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MatrixVec<N, R: Dim, C: Dim> {
    data: Vec<N>,
    nrows: R,
    ncols: C,
}

impl<N, R: Dim, C: Dim> MatrixVec<N, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<N>) -> MatrixVec<N, R, C> {
        assert!(
            nrows.value() * ncols.value() == data.len(),
            "Data storage buffer dimension mismatch."
        );
        MatrixVec {
            data: data,
            nrows: nrows,
            ncols: ncols,
        }
    }

    /// The underlying data storage.
    #[inline]
    pub fn data(&self) -> &Vec<N> {
        &self.data
    }

    /// The underlying mutable data storage.
    ///
    /// This is unsafe because this may cause UB if the vector is modified by the user.
    #[inline]
    pub unsafe fn data_mut(&mut self) -> &mut Vec<N> {
        &mut self.data
    }

    /// Resizes the underlying mutable data storage and unwraps it.
    ///
    /// If `sz` is larger than the current size, additional elements are uninitialized.
    /// If `sz` is smaller than the current size, additional elements are truncated.
    #[inline]
    pub unsafe fn resize(mut self, sz: usize) -> Vec<N> {
        let len = self.len();

        if sz < len {
            self.data.set_len(sz);
            self.data.shrink_to_fit();
        } else {
            self.data.reserve_exact(sz - len);
            self.data.set_len(sz);
        }

        self.data
    }
}

impl<N, R: Dim, C: Dim> Deref for MatrixVec<N, R, C> {
    type Target = Vec<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/*
 *
 * Dynamic − Static
 * Dynamic − Dynamic
 *
 */
unsafe impl<N: Scalar, C: Dim> Storage<N, Dynamic, C> for MatrixVec<N, Dynamic, C>
where
    DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>,
{
    type RStride = U1;
    type CStride = Dynamic;

    #[inline]
    fn ptr(&self) -> *const N {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Dynamic, C) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, Dynamic, C>
    where
        DefaultAllocator: Allocator<N, Dynamic, C>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, Dynamic, C>
    where
        DefaultAllocator: Allocator<N, Dynamic, C>,
    {
        self.clone()
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }
}

unsafe impl<N: Scalar, R: DimName> Storage<N, R, Dynamic> for MatrixVec<N, R, Dynamic>
where
    DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>,
{
    type RStride = U1;
    type CStride = R;

    #[inline]
    fn ptr(&self) -> *const N {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, Dynamic) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, R, Dynamic>
    where
        DefaultAllocator: Allocator<N, R, Dynamic>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, Dynamic>
    where
        DefaultAllocator: Allocator<N, R, Dynamic>,
    {
        self.clone()
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }
}

/*
 *
 * StorageMut, ContiguousStorage.
 *
 */
unsafe impl<N: Scalar, C: Dim> StorageMut<N, Dynamic, C> for MatrixVec<N, Dynamic, C>
where
    DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

unsafe impl<N: Scalar, C: Dim> ContiguousStorage<N, Dynamic, C> for MatrixVec<N, Dynamic, C>
where
    DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>,
{
}

unsafe impl<N: Scalar, C: Dim> ContiguousStorageMut<N, Dynamic, C> for MatrixVec<N, Dynamic, C>
where
    DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>,
{
}

unsafe impl<N: Scalar, R: DimName> StorageMut<N, R, Dynamic> for MatrixVec<N, R, Dynamic>
where
    DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Abomonation, R: Dim, C: Dim> Abomonation for MatrixVec<N, R, C> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

unsafe impl<N: Scalar, R: DimName> ContiguousStorage<N, R, Dynamic> for MatrixVec<N, R, Dynamic>
where
    DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>,
{
}

unsafe impl<N: Scalar, R: DimName> ContiguousStorageMut<N, R, Dynamic> for MatrixVec<N, R, Dynamic>
where
    DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>,
{
}
