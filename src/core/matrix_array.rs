use std::ops::{Deref, DerefMut, Mul};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};

use typenum::Prod;
use generic_array::{ArrayLength, GenericArray};

use core::Scalar;
use core::dimension::{DimName, U1};
use core::storage::{Storage, StorageMut, Owned, OwnedStorage};
use core::allocator::Allocator;
use core::default_allocator::DefaultAllocator;


/*
 *
 * Static Storage.
 *
 */
/// A array-based statically sized matrix data storage.
#[repr(C)]
pub struct MatrixArray<N, R, C>
where R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {

    data: GenericArray<N, Prod<R::Value, C::Value>>
}

impl<N, R, C> Hash for MatrixArray<N, R, C>
where N: Hash,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data[..].hash(state)
    }
}

impl<N, R, C> Deref for MatrixArray<N, R, C>
where R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {
    type Target = GenericArray<N, Prod<R::Value, C::Value>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<N, R, C> DerefMut for MatrixArray<N, R, C>
where R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<N, R, C> Debug for MatrixArray<N, R, C>
where N: Debug,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> Result {
        self.data.fmt(fmt)
    }
}

impl<N, R, C> Copy for MatrixArray<N, R, C>
    where N: Copy,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N>,
          GenericArray<N, Prod<R::Value, C::Value>> : Copy
{ }

impl<N, R, C> Clone for MatrixArray<N, R, C>
    where N: Clone,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn clone(&self) -> Self {
        MatrixArray {
            data: self.data.clone()
        }
    }
}

impl<N, R, C> Eq for MatrixArray<N, R, C>
    where N: Eq,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
}

impl<N, R, C> PartialEq for MatrixArray<N, R, C>
    where N: PartialEq,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.data  == right.data
    }
}


unsafe impl<N, R, C> Storage<N, R, C> for MatrixArray<N, R, C>
    where N: Scalar,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
    type RStride = U1;
    type CStride = R;
    type Alloc   = DefaultAllocator;

    #[inline]
    fn into_owned(self) -> Owned<N, R, C, Self::Alloc> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, C, Self::Alloc> {
        let it = self.iter().cloned();

        Self::Alloc::allocate_from_iterator(self.shape().0, self.shape().1, it)
    }

    #[inline]
    fn ptr(&self) -> *const N {
        self[..].as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, C) {
        (R::name(), C::name())
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), Self::CStride::name())
    }
}

unsafe impl<N, R, C> StorageMut<N, R, C> for MatrixArray<N, R, C>
    where N: Scalar,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self[..].as_mut_ptr()
    }
}

unsafe impl<N, R, C> OwnedStorage<N, R, C> for MatrixArray<N, R, C>
    where N: Scalar,
          R: DimName,
          C: DimName,
          R::Value: Mul<C::Value>,
          Prod<R::Value, C::Value>: ArrayLength<N> {
    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self[..]
    }
}
