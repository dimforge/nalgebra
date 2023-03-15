#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use crate::base::allocator::Allocator;
use crate::base::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Dim, DimName, Dyn, U1};
use crate::base::storage::{IsContiguous, Owned, RawStorage, RawStorageMut, ReshapableStorage};
use crate::base::{Scalar, Vector};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{
    de::{Deserialize, DeserializeSeed, Deserializer, Error, SeqAccess, Unexpected, Visitor},
    ser::{Serialize, SerializeTuple, Serializer},
};

use crate::Storage;
#[cfg(feature = "serde-serialize")]
use core::{fmt, marker::PhantomData};
use std::mem::MaybeUninit;

/*
 *
 * RawStorage.
 *
 */
/// A Vec-based matrix data storage. It may be dynamically-sized.
#[repr(C)]
#[derive(Eq, Debug, Clone, PartialEq)]
pub struct VecStorage<T, R: Dim, C: Dim> {
    data: Vec<T>,
    nrows: R,
    ncols: C,
}

#[cfg(feature = "serde-serialize")]
impl<T, R: Dim, C: Dim> Serialize for VecStorage<T, R, C>
where
    T: Serialize,
    R: Serialize,
    C: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut serializer = serializer.serialize_tuple(3)?;
        serializer.serialize_element(&self.nrows)?;
        serializer.serialize_element(&self.ncols)?;
        serializer.serialize_element(&DynamicTuple(&self.data))?;
        serializer.end()
    }
}

#[cfg(feature = "serde-serialize")]
struct DynamicTuple<'a, T>(&'a [T]);

#[cfg(feature = "serde-serialize")]
impl<T> Serialize for DynamicTuple<'_, T>
where
    T: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut serializer = serializer.serialize_tuple(self.0.len())?;
        for elt in self.0 {
            serializer.serialize_element(elt)?;
        }
        serializer.end()
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, T, R: Dim, C: Dim> Deserialize<'a> for VecStorage<T, R, C>
where
    T: Deserialize<'a>,
    R: Deserialize<'a>,
    C: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        deserializer.deserialize_tuple(3, VecStorageVisitor(PhantomData))
    }
}

#[cfg(feature = "serde-serialize")]
struct VecStorageVisitor<T, R, C>(PhantomData<fn() -> (T, R, C)>);

#[cfg(feature = "serde-serialize")]
impl<'de, T, R: Dim, C: Dim> Visitor<'de> for VecStorageVisitor<T, R, C>
where
    T: Deserialize<'de>,
    R: Deserialize<'de>,
    C: Deserialize<'de>,
{
    type Value = VecStorage<T, R, C>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a matrix")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let nrows = seq
            .next_element::<R>()?
            .ok_or_else(|| Error::invalid_length(0, &"a matrix"))?;
        let ncols = seq
            .next_element::<C>()?
            .ok_or_else(|| Error::invalid_length(1, &"a matrix"))?;

        let size = nrows.value().checked_mul(ncols.value()).ok_or_else(|| {
            Error::invalid_value(
                Unexpected::Unsigned(ncols.value() as u64),
                &"in-bounds dimensions",
            )
        })?;
        let data = seq
            .next_element_seed(DeserializeDynamicTuple(size, PhantomData))?
            .ok_or_else(|| Error::invalid_length(2, &"a matrix"))?;

        Ok(VecStorage { data, nrows, ncols })
    }
}

#[cfg(feature = "serde-serialize")]
struct DeserializeDynamicTuple<T>(usize, PhantomData<fn() -> T>);

#[cfg(feature = "serde-serialize")]
impl<'a, T> DeserializeSeed<'a> for DeserializeDynamicTuple<T>
where
    T: Deserialize<'a>,
{
    type Value = Vec<T>;

    fn deserialize<Des>(self, deserializer: Des) -> Result<Vec<T>, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        deserializer.deserialize_tuple(self.0, DynamicTupleVisitor(self.0, PhantomData))
    }
}

#[cfg(feature = "serde-serialize")]
struct DynamicTupleVisitor<T>(usize, PhantomData<fn() -> T>);

#[cfg(feature = "serde-serialize")]
impl<'de, T> Visitor<'de> for DynamicTupleVisitor<T>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "matrix elements")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut v = Vec::with_capacity(self.0);
        for n in 0..self.0 {
            let elt = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(n, &"a complete matrix"))?;
            v.push(elt);
        }
        Ok(v)
    }
}

#[deprecated(note = "renamed to `VecStorage`")]
/// Renamed to [`VecStorage`].
pub type MatrixVec<T, R, C> = VecStorage<T, R, C>;

impl<T, R: Dim, C: Dim> VecStorage<T, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<T>) -> Self {
        assert!(
            nrows.value() * ncols.value() == data.len(),
            "Data storage buffer dimension mismatch."
        );
        Self { data, nrows, ncols }
    }

    /// The underlying data storage.
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// The underlying mutable data storage.
    ///
    /// # Safety
    /// This is unsafe because this may cause UB if the size of the vector is changed
    /// by the user.
    #[inline]
    pub unsafe fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Resizes the underlying mutable data storage and unwraps it.
    ///
    /// # Safety
    /// - If `sz` is larger than the current size, additional elements are uninitialized.
    /// - If `sz` is smaller than the current size, additional elements are truncated but **not** dropped.
    ///   It is the responsibility of the caller of this method to drop these elements.
    #[inline]
    pub unsafe fn resize(mut self, sz: usize) -> Vec<MaybeUninit<T>> {
        let len = self.len();

        let new_data = if sz < len {
            // Use `set_len` instead of `truncate` because we don’t want to
            // drop the removed elements (it’s the caller’s responsibility).
            self.data.set_len(sz);
            self.data.shrink_to_fit();

            // Safety:
            // - MaybeUninit<T> has the same alignment and layout as T.
            // - The length and capacity come from a valid vector.
            Vec::from_raw_parts(
                self.data.as_mut_ptr() as *mut MaybeUninit<T>,
                self.data.len(),
                self.data.capacity(),
            )
        } else {
            self.data.reserve_exact(sz - len);

            // Safety:
            // - MaybeUninit<T> has the same alignment and layout as T.
            // - The length and capacity come from a valid vector.
            let mut new_data = Vec::from_raw_parts(
                self.data.as_mut_ptr() as *mut MaybeUninit<T>,
                self.data.len(),
                self.data.capacity(),
            );

            // Safety: we can set the length here because MaybeUninit is always assumed
            //         to be initialized.
            new_data.set_len(sz);
            new_data
        };

        // Avoid double-free by forgetting `self` because its data buffer has
        // been transfered to `new_data`.
        std::mem::forget(self);
        new_data
    }

    /// The number of elements on the underlying vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the underlying vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A slice containing all the components stored in this storage in column-major order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..]
    }

    /// A mutable slice containing all the components stored in this storage in column-major order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

impl<T, R: Dim, C: Dim> From<VecStorage<T, R, C>> for Vec<T> {
    fn from(vec: VecStorage<T, R, C>) -> Self {
        vec.data
    }
}

/*
 *
 * Dyn − Static
 * Dyn − Dyn
 *
 */
unsafe impl<T, C: Dim> RawStorage<T, Dyn, C> for VecStorage<T, Dyn, C> {
    type RStride = U1;
    type CStride = Dyn;

    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Dyn, C) {
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
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        &self.data
    }
}

unsafe impl<T: Scalar, C: Dim> Storage<T, Dyn, C> for VecStorage<T, Dyn, C>
where
    DefaultAllocator: Allocator<T, Dyn, C, Buffer = Self>,
{
    #[inline]
    fn into_owned(self) -> Owned<T, Dyn, C>
    where
        DefaultAllocator: Allocator<T, Dyn, C>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, Dyn, C>
    where
        DefaultAllocator: Allocator<T, Dyn, C>,
    {
        self.clone()
    }
}

unsafe impl<T, R: DimName> RawStorage<T, R, Dyn> for VecStorage<T, R, Dyn> {
    type RStride = U1;
    type CStride = R;

    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, Dyn) {
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
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        &self.data
    }
}

unsafe impl<T: Scalar, R: DimName> Storage<T, R, Dyn> for VecStorage<T, R, Dyn>
where
    DefaultAllocator: Allocator<T, R, Dyn, Buffer = Self>,
{
    #[inline]
    fn into_owned(self) -> Owned<T, R, Dyn>
    where
        DefaultAllocator: Allocator<T, R, Dyn>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, R, Dyn>
    where
        DefaultAllocator: Allocator<T, R, Dyn>,
    {
        self.clone()
    }
}

/*
 *
 * RawStorageMut, ContiguousStorage.
 *
 */
unsafe impl<T, C: Dim> RawStorageMut<T, Dyn, C> for VecStorage<T, Dyn, C> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

unsafe impl<T, R: Dim, C: Dim> IsContiguous for VecStorage<T, R, C> {}

impl<T, C1, C2> ReshapableStorage<T, Dyn, C1, Dyn, C2> for VecStorage<T, Dyn, C1>
where
    T: Scalar,
    C1: Dim,
    C2: Dim,
{
    type Output = VecStorage<T, Dyn, C2>;

    fn reshape_generic(self, nrows: Dyn, ncols: C2) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

impl<T, C1, R2> ReshapableStorage<T, Dyn, C1, R2, Dyn> for VecStorage<T, Dyn, C1>
where
    T: Scalar,
    C1: Dim,
    R2: DimName,
{
    type Output = VecStorage<T, R2, Dyn>;

    fn reshape_generic(self, nrows: R2, ncols: Dyn) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

unsafe impl<T, R: DimName> RawStorageMut<T, R, Dyn> for VecStorage<T, R, Dyn> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

impl<T, R1, C2> ReshapableStorage<T, R1, Dyn, Dyn, C2> for VecStorage<T, R1, Dyn>
where
    T: Scalar,
    R1: DimName,
    C2: Dim,
{
    type Output = VecStorage<T, Dyn, C2>;

    fn reshape_generic(self, nrows: Dyn, ncols: C2) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

impl<T, R1, R2> ReshapableStorage<T, R1, Dyn, R2, Dyn> for VecStorage<T, R1, Dyn>
where
    T: Scalar,
    R1: DimName,
    R2: DimName,
{
    type Output = VecStorage<T, R2, Dyn>;

    fn reshape_generic(self, nrows: R2, ncols: Dyn) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

impl<T, R: Dim> Extend<T> for VecStorage<T, R, Dyn> {
    /// Extends the number of columns of the `VecStorage` with elements
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `VecStorage`.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
        self.ncols = Dyn(self.data.len() / self.nrows.value());
        assert!(self.data.len() % self.nrows.value() == 0,
          "The number of elements produced by the given iterator was not a multiple of the number of rows.");
    }
}

impl<'a, T: 'a + Copy, R: Dim> Extend<&'a T> for VecStorage<T, R, Dyn> {
    /// Extends the number of columns of the `VecStorage` with elements
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `VecStorage`.
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied())
    }
}

impl<T, R, RV, SV> Extend<Vector<T, RV, SV>> for VecStorage<T, R, Dyn>
where
    T: Scalar,
    R: Dim,
    RV: Dim,
    SV: RawStorage<T, RV>,
    ShapeConstraint: SameNumberOfRows<R, RV>,
{
    /// Extends the number of columns of the `VecStorage` with vectors
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of rows of each `Vector`
    /// yielded by the iterator is not equal to the number of rows
    /// of this `VecStorage`.
    fn extend<I: IntoIterator<Item = Vector<T, RV, SV>>>(&mut self, iter: I) {
        let nrows = self.nrows.value();
        let iter = iter.into_iter();
        let (lower, _upper) = iter.size_hint();
        self.data.reserve(nrows * lower);
        for vector in iter {
            assert_eq!(nrows, vector.shape().0);
            self.data.extend(vector.iter().cloned());
        }
        self.ncols = Dyn(self.data.len() / nrows);
    }
}

impl<T> Extend<T> for VecStorage<T, Dyn, U1> {
    /// Extends the number of rows of the `VecStorage` with elements
    /// from the given iterator.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
        self.nrows = Dyn(self.data.len());
    }
}
