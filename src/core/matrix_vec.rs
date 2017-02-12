use std::ops::Deref;

use core::Scalar;
use core::dimension::{Dim, DimName, Dynamic, U1};
use core::storage::{Storage, StorageMut, Owned, OwnedStorage};
use core::default_allocator::DefaultAllocator;

/*
 *
 * Storage.
 *
 */
/// A Vec-based matrix data storage. It may be dynamically-sized.
#[repr(C)]
#[derive(Eq, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixVec<N, R: Dim, C: Dim> {
    data:   Vec<N>,
    nrows:  R,
    ncols:  C
}

impl<N, R: Dim, C: Dim> MatrixVec<N, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<N>) -> MatrixVec<N, R, C> {
        assert!(nrows.value() * ncols.value() == data.len(), "Data storage buffer dimension mismatch.");
        MatrixVec {
            data:   data,
            nrows:  nrows,
            ncols:  ncols
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
unsafe impl<N: Scalar, C: Dim> Storage<N, Dynamic, C> for MatrixVec<N, Dynamic, C> {
    type RStride = U1;
    type CStride = Dynamic;
    type Alloc   = DefaultAllocator;

    #[inline]
    fn into_owned(self) -> Owned<N, Dynamic, C, Self::Alloc> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, Dynamic, C, Self::Alloc> {
        self.clone()
    }

    #[inline]
    fn ptr(&self) -> *const N {
        self[..].as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Dynamic, C) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }
}


unsafe impl<N: Scalar, R: DimName> Storage<N, R, Dynamic> for MatrixVec<N, R, Dynamic> {
    type RStride = U1;
    type CStride = R;
    type Alloc   = DefaultAllocator;

    #[inline]
    fn into_owned(self) -> Owned<N, R, Dynamic, Self::Alloc> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, Dynamic, Self::Alloc> {
        self.clone()
    }

    #[inline]
    fn ptr(&self) -> *const N {
        self[..].as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, Dynamic) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }
}




/*
 *
 * StorageMut, OwnedStorage.
 *
 */
unsafe impl<N: Scalar, C: Dim> StorageMut<N, Dynamic, C> for MatrixVec<N, Dynamic, C> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.as_mut_slice().as_mut_ptr()
    }
}

unsafe impl<N: Scalar, C: Dim> OwnedStorage<N, Dynamic, C> for MatrixVec<N, Dynamic, C> {
    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}


unsafe impl<N: Scalar, R: DimName> StorageMut<N, R, Dynamic> for MatrixVec<N, R, Dynamic> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.as_mut_slice().as_mut_ptr()
    }
}

unsafe impl<N: Scalar, R: DimName> OwnedStorage<N, R, Dynamic> for MatrixVec<N, R, Dynamic> {
    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}
