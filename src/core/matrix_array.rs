use std::ops::{Deref, DerefMut, Mul};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "serde-serialize")]
use serde::ser::SerializeSeq;
#[cfg(feature = "serde-serialize")]
use serde::de::{Error, SeqAccess, Visitor};
#[cfg(feature = "serde-serialize")]
use std::mem;
#[cfg(feature = "serde-serialize")]
use std::marker::PhantomData;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use typenum::Prod;
use generic_array::{ArrayLength, GenericArray};

use core::Scalar;
use core::dimension::{DimName, U1};
use core::storage::{ContiguousStorage, ContiguousStorageMut, Owned, Storage, StorageMut};
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
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    data: GenericArray<N, Prod<R::Value, C::Value>>,
}

impl<N, R, C> Hash for MatrixArray<N, R, C>
where
    N: Hash,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data[..].hash(state)
    }
}

impl<N, R, C> Deref for MatrixArray<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Target = GenericArray<N, Prod<R::Value, C::Value>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<N, R, C> DerefMut for MatrixArray<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<N, R, C> Debug for MatrixArray<N, R, C>
where
    N: Debug,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        self.data.fmt(fmt)
    }
}

macro_rules! impl_copy {
    ($NRows: ty, $NCols: ty) => {
        impl<N: Copy> Copy for MatrixArray<N, $NRows, $NCols> {}
    };
}

use core::dimension::{U2, U3, U4, U5, U6};
impl_copy!(U1, U1);
impl_copy!(U1, U2);
impl_copy!(U1, U3);
impl_copy!(U1, U4);
impl_copy!(U1, U5);
impl_copy!(U1, U6);

impl_copy!(U2, U1);
impl_copy!(U2, U2);
impl_copy!(U2, U3);
impl_copy!(U2, U4);
impl_copy!(U2, U5);
impl_copy!(U2, U6);

impl_copy!(U3, U1);
impl_copy!(U3, U2);
impl_copy!(U3, U3);
impl_copy!(U3, U4);
impl_copy!(U3, U5);
impl_copy!(U3, U6);

impl_copy!(U4, U1);
impl_copy!(U4, U2);
impl_copy!(U4, U3);
impl_copy!(U4, U4);
impl_copy!(U4, U5);
impl_copy!(U4, U6);

impl_copy!(U5, U1);
impl_copy!(U5, U2);
impl_copy!(U5, U3);
impl_copy!(U5, U4);
impl_copy!(U5, U5);
impl_copy!(U5, U6);

impl_copy!(U6, U1);
impl_copy!(U6, U2);
impl_copy!(U6, U3);
impl_copy!(U6, U4);
impl_copy!(U6, U5);
impl_copy!(U6, U6);

impl<N, R, C> Clone for MatrixArray<N, R, C>
where
    N: Clone,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn clone(&self) -> Self {
        MatrixArray {
            data: self.data.clone(),
        }
    }
}

impl<N, R, C> Eq for MatrixArray<N, R, C>
where
    N: Eq,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
}

impl<N, R, C> PartialEq for MatrixArray<N, R, C>
where
    N: PartialEq,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.data == right.data
    }
}

unsafe impl<N, R, C> Storage<N, R, C> for MatrixArray<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
    type RStride = U1;
    type CStride = R;

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

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        let it = self.iter().cloned();

        DefaultAllocator::allocate_from_iterator(self.shape().0, self.shape().1, it)
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }
}

unsafe impl<N, R, C> StorageMut<N, R, C> for MatrixArray<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self[..].as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self[..]
    }
}

unsafe impl<N, R, C> ContiguousStorage<N, R, C> for MatrixArray<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
}

unsafe impl<N, R, C> ContiguousStorageMut<N, R, C> for MatrixArray<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
}

/*
 *
 * Allocation-less serde impls.
 *
 */
// XXX: open an issue for GenericArray so that it implements serde traits?
#[cfg(feature = "serde-serialize")]
impl<N, R, C> Serialize for MatrixArray<N, R, C>
where
    N: Scalar + Serialize,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut serializer = serializer.serialize_seq(Some(R::dim() * C::dim()))?;

        for e in self.iter() {
            serializer.serialize_element(e)?;
        }

        serializer.end()
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, R, C> Deserialize<'a> for MatrixArray<N, R, C>
where
    N: Scalar + Deserialize<'a>,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        deserializer.deserialize_seq(MatrixArrayVisitor::new())
    }
}

#[cfg(feature = "serde-serialize")]
/// A visitor that produces a matrix array.
struct MatrixArrayVisitor<N, R, C> {
    marker: PhantomData<(N, R, C)>,
}

#[cfg(feature = "serde-serialize")]
impl<N, R, C> MatrixArrayVisitor<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    /// Construct a new sequence visitor.
    pub fn new() -> Self {
        MatrixArrayVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, R, C> Visitor<'a> for MatrixArrayVisitor<N, R, C>
where
    N: Scalar + Deserialize<'a>,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Value = MatrixArray<N, R, C>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a matrix array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<MatrixArray<N, R, C>, V::Error>
    where
        V: SeqAccess<'a>,
    {
        let mut out: Self::Value = unsafe { mem::uninitialized() };
        let mut curr = 0;

        while let Some(value) = try!(visitor.next_element()) {
            out[curr] = value;
            curr += 1;
        }

        if curr == R::dim() * C::dim() {
            Ok(out)
        } else {
            Err(V::Error::invalid_length(curr, &self))
        }
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, R, C> Abomonation for MatrixArray<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    N: Abomonation,
{
    unsafe fn entomb(&self, writer: &mut Vec<u8>) {
        for element in self.data.as_slice() {
            element.entomb(writer);
        }
    }

    unsafe fn embalm(&mut self) {
        for element in self.data.as_mut_slice() {
            element.embalm();
        }
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        for element in self.data.as_mut_slice() {
            let temp = bytes;
            bytes = if let Some(remainder) = element.exhume(temp) {
                remainder
            } else {
                return None;
            }
        }
        Some(bytes)
    }
}
