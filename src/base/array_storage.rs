use std::fmt::{self, Debug, Formatter};
// use std::hash::{Hash, Hasher};
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::ops::Mul;

#[cfg(feature = "serde-serialize")]
use serde::de::{Error, SeqAccess, Visitor};
#[cfg(feature = "serde-serialize")]
use serde::ser::SerializeSeq;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "serde-serialize")]
use std::marker::PhantomData;
#[cfg(feature = "serde-serialize")]
use std::mem;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use crate::base::allocator::Allocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Const, ToTypenum};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, Owned, ReshapableStorage, Storage, StorageMut,
};
use crate::base::Scalar;

/*
 *
 * Static Storage.
 *
 */
/// A array-based statically sized matrix data storage.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ArrayStorage<N, const R: usize, const C: usize> {
    data: [[N; R]; C],
}

// TODO: remove this once the stdlib implements Default for arrays.
impl<N: Default, const R: usize, const C: usize> Default for ArrayStorage<N, R, C>
where
    [[N; R]; C]: Default,
{
    #[inline]
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<N: Debug, const R: usize, const C: usize> Debug for ArrayStorage<N, R, C> {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        self.data.fmt(fmt)
    }
}

unsafe impl<N, const R: usize, const C: usize> Storage<N, Const<R>, Const<C>>
    for ArrayStorage<N, R, C>
where
    N: Scalar,
    DefaultAllocator: Allocator<N, Const<R>, Const<C>, Buffer = Self>,
{
    type RStride = Const<1>;
    type CStride = Const<R>;

    #[inline]
    fn ptr(&self) -> *const N {
        self.data.as_ptr() as *const N
    }

    #[inline]
    fn shape(&self) -> (Const<R>, Const<C>) {
        (Const, Const)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Const, Const)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, Const<R>, Const<C>>
    where
        DefaultAllocator: Allocator<N, Const<R>, Const<C>>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, Const<R>, Const<C>>
    where
        DefaultAllocator: Allocator<N, Const<R>, Const<C>>,
    {
        let it = self.as_slice().iter().cloned();
        DefaultAllocator::allocate_from_iterator(self.shape().0, self.shape().1, it)
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        unsafe { std::slice::from_raw_parts(self.ptr(), R * C) }
    }
}

unsafe impl<N, const R: usize, const C: usize> StorageMut<N, Const<R>, Const<C>>
    for ArrayStorage<N, R, C>
where
    N: Scalar,
    DefaultAllocator: Allocator<N, Const<R>, Const<C>, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.data.as_mut_ptr() as *mut N
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr_mut(), R * C) }
    }
}

unsafe impl<N, const R: usize, const C: usize> ContiguousStorage<N, Const<R>, Const<C>>
    for ArrayStorage<N, R, C>
where
    N: Scalar,
    DefaultAllocator: Allocator<N, Const<R>, Const<C>, Buffer = Self>,
{
}

unsafe impl<N, const R: usize, const C: usize> ContiguousStorageMut<N, Const<R>, Const<C>>
    for ArrayStorage<N, R, C>
where
    N: Scalar,
    DefaultAllocator: Allocator<N, Const<R>, Const<C>, Buffer = Self>,
{
}

impl<N, const R1: usize, const C1: usize, const R2: usize, const C2: usize>
    ReshapableStorage<N, Const<R1>, Const<C1>, Const<R2>, Const<C2>> for ArrayStorage<N, R1, C1>
where
    N: Scalar,
    Const<R1>: ToTypenum,
    Const<C1>: ToTypenum,
    Const<R2>: ToTypenum,
    Const<C2>: ToTypenum,
    <Const<R1> as ToTypenum>::Typenum: Mul<<Const<C1> as ToTypenum>::Typenum>,
    <Const<R2> as ToTypenum>::Typenum: Mul<
        <Const<C2> as ToTypenum>::Typenum,
        Output = typenum::Prod<
            <Const<R1> as ToTypenum>::Typenum,
            <Const<C1> as ToTypenum>::Typenum,
        >,
    >,
{
    type Output = ArrayStorage<N, R2, C2>;

    fn reshape_generic(self, _: Const<R2>, _: Const<C2>) -> Self::Output {
        unsafe {
            let data: [[N; R2]; C2] = std::mem::transmute_copy(&self.data);
            std::mem::forget(self.data);
            ArrayStorage { data }
        }
    }
}

/*
 *
 * Serialization.
 *
 */
// XXX: open an issue for serde so that it allows the serialization/deserialization of all arrays?
#[cfg(feature = "serde-serialize")]
impl<N, const R: usize, const C: usize> Serialize for ArrayStorage<N, R, C>
where
    N: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut serializer = serializer.serialize_seq(Some(R * C))?;

        for e in self.as_slice().iter() {
            serializer.serialize_element(e)?;
        }

        serializer.end()
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, const R: usize, const C: usize> Deserialize<'a> for ArrayStorage<N, R, C>
where
    N: Scalar + Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        deserializer.deserialize_seq(ArrayStorageVisitor::new())
    }
}

#[cfg(feature = "serde-serialize")]
/// A visitor that produces a matrix array.
struct ArrayStorageVisitor<N, const R: usize, const C: usize> {
    marker: PhantomData<N>,
}

#[cfg(feature = "serde-serialize")]
impl<N, const R: usize, const C: usize> ArrayStorageVisitor<N, R, C>
where
    N: Scalar,
{
    /// Construct a new sequence visitor.
    pub fn new() -> Self {
        ArrayStorageVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, const R: usize, const C: usize> Visitor<'a> for ArrayStorageVisitor<N, R, C>
where
    N: Scalar + Deserialize<'a>,
{
    type Value = ArrayStorage<N, R, C>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a matrix array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<ArrayStorage<N, R, C>, V::Error>
    where
        V: SeqAccess<'a>,
    {
        let mut out: Self::Value = unsafe { mem::MaybeUninit::uninit().assume_init() };
        let mut curr = 0;

        while let Some(value) = visitor.next_element()? {
            *out.as_mut_slice()
                .get_mut(curr)
                .ok_or_else(|| V::Error::invalid_length(curr, &self))? = value;
            curr += 1;
        }

        if curr == R * C {
            Ok(out)
        } else {
            Err(V::Error::invalid_length(curr, &self))
        }
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<N: Scalar + bytemuck::Zeroable, R: DimName, C: DimName> bytemuck::Zeroable
    for ArrayStorage<N, R, C>
where
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    Self: Copy,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<N: Scalar + bytemuck::Pod, R: DimName, C: DimName> bytemuck::Pod
    for ArrayStorage<N, R, C>
where
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    Self: Copy,
{
}

#[cfg(feature = "abomonation-serialize")]
impl<N, const R: usize, const C: usize> Abomonation for ArrayStorage<N, R, C>
where
    N: Scalar + Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        for element in self.as_slice() {
            element.entomb(writer)?;
        }

        Ok(())
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        for element in self.as_mut_slice() {
            let temp = bytes;
            bytes = if let Some(remainder) = element.exhume(temp) {
                remainder
            } else {
                return None;
            }
        }
        Some(bytes)
    }

    fn extent(&self) -> usize {
        self.as_slice().iter().fold(0, |acc, e| acc + e.extent())
    }
}
