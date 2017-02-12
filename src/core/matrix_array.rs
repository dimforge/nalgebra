use std::mem;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Mul};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Serializer, Deserialize, Deserializer};
use serde::ser::SerializeSeq;
use serde::de::{SeqVisitor, Visitor};

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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
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


/*
 *
 * Allocation-less serde impls.
 *
 */
// XXX: open an issue for GenericArray so that it implements serde traits?
impl<N, R, C> Serialize for MatrixArray<N, R, C>
where N: Scalar + Serialize,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {


    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {
            let mut serializer = serializer.serialize_seq_fixed_size(R::dim() * C::dim())?;

            for e in self.iter() {
                serializer.serialize_element(e)?;
            }

            serializer.end()
        }
}


impl<N, R, C> Deserialize for MatrixArray<N, R, C>
where N: Scalar + Deserialize,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {


    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer {

            let len = R::dim() * C::dim();
            deserializer.deserialize_seq_fixed_size(len, MatrixArrayVisitor::new())
        }
}


/// A visitor that produces a matrix array.
struct MatrixArrayVisitor<N, R, C> {
    marker: PhantomData<(N, R, C)>
}

impl<N, R, C> MatrixArrayVisitor<N, R, C>
where N: Scalar,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {

    /// Construct a new sequence visitor.
    pub fn new() -> Self {
        MatrixArrayVisitor {
            marker: PhantomData,
        }
    }
}

impl<N, R, C> Visitor for MatrixArrayVisitor<N, R, C>
where N: Scalar + Deserialize,
      R: DimName,
      C: DimName,
      R::Value: Mul<C::Value>,
      Prod<R::Value, C::Value>: ArrayLength<N> {

    type Value = MatrixArray<N, R, C>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a matrix array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<MatrixArray<N, R, C>, V::Error>
        where V: SeqVisitor {

        let mut out: Self::Value = unsafe { mem::uninitialized() };
        let mut curr = 0;

        while let Some(value) = try!(visitor.visit()) {
            out[curr] = value;
            curr += 1;
        }

        Ok(out)
    }
}
