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
    de::{Deserialize, Deserializer, Error},
    ser::{Serialize, Serializer},
};

use crate::Storage;
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

impl<T> Default for VecStorage<T, Dyn, Dyn> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            nrows: Dyn::from_usize(0),
            ncols: Dyn::from_usize(0),
        }
    }
}

impl<T, R: DimName> Default for VecStorage<T, R, Dyn> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            nrows: R::name(),
            ncols: Dyn::from_usize(0),
        }
    }
}

impl<T, C: DimName> Default for VecStorage<T, Dyn, C> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            nrows: Dyn::from_usize(0),
            ncols: C::name(),
        }
    }
}

impl<T: Default, R: DimName, C: DimName> Default for VecStorage<T, R, C> {
    fn default() -> Self {
        let nrows = R::name();
        let ncols = C::name();
        let mut data = Vec::new();
        data.resize_with(nrows.value() * ncols.value(), Default::default);
        Self { data, nrows, ncols }
    }
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
        (&self.data, &self.nrows, &self.ncols).serialize(serializer)
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
        let (data, nrows, ncols): (Vec<T>, R, C) = Deserialize::deserialize(deserializer)?;

        // SAFETY: make sure the data we deserialize have the
        //         correct number of elements.
        if nrows.value() * ncols.value() != data.len() {
            return Err(Des::Error::custom(format!(
                "Expected {} components, found {}",
                nrows.value() * ncols.value(),
                data.len()
            )));
        }

        Ok(Self { data, nrows, ncols })
    }
}

#[deprecated(note = "renamed to `VecStorage`")]
/// Renamed to [`VecStorage`].
pub type MatrixVec<T, R, C> = VecStorage<T, R, C>;

impl<T, R: Dim, C: Dim> VecStorage<T, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    ///
    /// This function constructs a `VecStorage` by wrapping a `Vec<T>` with the specified
    /// number of rows and columns. The storage uses column-major order, meaning elements
    /// are stored column by column.
    ///
    /// # Arguments
    ///
    /// * `nrows` - The number of rows in the matrix
    /// * `ncols` - The number of columns in the matrix
    /// * `data` - A vector containing all matrix elements in column-major order
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` does not equal `nrows * ncols`.
    ///
    /// # Examples
    ///
    /// Creating a 2x3 matrix storage with dynamic dimensions:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// // Create storage for a 2x3 matrix in column-major order
    /// // Matrix:
    /// // [1, 3, 5]
    /// // [2, 4, 6]
    /// let data = vec![1, 2, 3, 4, 5, 6]; // Column-major: col1[1,2], col2[3,4], col3[5,6]
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// assert_eq!(storage.len(), 6);
    /// ```
    ///
    /// Creating storage from computed values:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// // Create a 3x2 matrix where element (i,j) = i + j
    /// let nrows = 3;
    /// let ncols = 2;
    /// let mut data = Vec::with_capacity(nrows * ncols);
    ///
    /// // Fill in column-major order
    /// for col in 0..ncols {
    ///     for row in 0..nrows {
    ///         data.push(row + col);
    ///     }
    /// }
    ///
    /// let storage = VecStorage::new(Dyn(nrows), Dyn(ncols), data);
    /// assert_eq!(storage.as_vec().len(), 6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`as_vec`](Self::as_vec) - Get a reference to the underlying vector
    /// * [`len`](Self::len) - Get the total number of elements
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<T>) -> Self {
        assert!(
            nrows.value() * ncols.value() == data.len(),
            "Data storage buffer dimension mismatch."
        );
        Self { data, nrows, ncols }
    }

    /// Returns a reference to the underlying vector storage.
    ///
    /// This function provides read-only access to the internal `Vec<T>` that stores
    /// all matrix elements in column-major order. This is useful when you need to
    /// work with the raw data or pass it to functions expecting a vector.
    ///
    /// # Returns
    ///
    /// A reference to the underlying `Vec<T>` containing all matrix elements.
    ///
    /// # Examples
    ///
    /// Accessing the underlying vector:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// let vec_ref = storage.as_vec();
    /// assert_eq!(vec_ref.len(), 4);
    /// assert_eq!(vec_ref[0], 1);
    /// ```
    ///
    /// Using the vector reference for computations:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// // Calculate sum of all elements
    /// let sum: f64 = storage.as_vec().iter().sum();
    /// assert_eq!(sum, 21.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`as_vec_mut`](Self::as_vec_mut) - Get a mutable reference to the underlying vector (unsafe)
    /// * [`as_slice`](Self::as_slice) - Get a slice view of the data
    /// * [`len`](Self::len) - Get the number of elements
    #[inline]
    #[must_use]
    pub const fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable reference to the underlying vector storage.
    ///
    /// This function provides mutable access to the internal `Vec<T>` that stores
    /// all matrix elements. This is marked unsafe because modifying the vector's
    /// length could break the storage's dimension invariants.
    ///
    /// # Safety
    ///
    /// The caller must ensure that they do not change the length of the vector.
    /// Changing the vector's length will cause undefined behavior because the
    /// storage's dimensions (nrows × ncols) must always equal the vector's length.
    ///
    /// You may modify the elements themselves, but operations like `push`, `pop`,
    /// `truncate`, `resize`, or `clear` must not be used.
    ///
    /// # Examples
    ///
    /// Safely modifying elements (safe usage):
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// // SAFE: We're only modifying elements, not changing the length
    /// unsafe {
    ///     let vec = storage.as_vec_mut();
    ///     vec[0] = 10;
    ///     vec[1] = 20;
    /// }
    ///
    /// assert_eq!(storage.as_vec()[0], 10);
    /// assert_eq!(storage.as_vec()[1], 20);
    /// assert_eq!(storage.len(), 4); // Length unchanged
    /// ```
    ///
    /// Using iterator methods that don't change length:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// // SAFE: iter_mut doesn't change the length
    /// unsafe {
    ///     storage.as_vec_mut().iter_mut().for_each(|x| *x *= 2.0);
    /// }
    ///
    /// assert_eq!(storage.as_vec()[0], 2.0);
    /// assert_eq!(storage.as_vec()[3], 8.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`as_vec`](Self::as_vec) - Get an immutable reference to the underlying vector
    /// * [`as_mut_slice`](Self::as_mut_slice) - Get a mutable slice (safe alternative)
    #[inline]
    pub const unsafe fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Resizes the underlying vector storage and returns it as uninitialized memory.
    ///
    /// This function consumes the `VecStorage`, resizes its internal vector to the
    /// specified size, and returns it as a `Vec<MaybeUninit<T>>`. This is a low-level
    /// operation used internally for efficient memory management during matrix
    /// reshaping operations.
    ///
    /// # Arguments
    ///
    /// * `sz` - The new size for the vector
    ///
    /// # Returns
    ///
    /// A `Vec<MaybeUninit<T>>` containing the resized data.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - If `sz` is larger than the current size, the additional elements are uninitialized.
    ///   The caller must initialize these elements before reading them.
    /// - If `sz` is smaller than the current size, the removed elements are truncated but
    ///   **not** dropped. It is the caller's responsibility to properly drop these elements
    ///   to avoid memory leaks.
    ///
    /// # Examples
    ///
    /// Growing the storage (caller must initialize new elements):
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    /// use std::mem::MaybeUninit;
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// unsafe {
    ///     // Resize to hold 6 elements
    ///     let mut resized = storage.resize(6);
    ///
    ///     // The first 4 elements are initialized, last 2 are not
    ///     assert_eq!(resized.len(), 6);
    ///
    ///     // Initialize the new elements
    ///     resized[4] = MaybeUninit::new(5);
    ///     resized[5] = MaybeUninit::new(6);
    /// }
    /// ```
    ///
    /// Shrinking the storage (caller must handle dropped elements):
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// unsafe {
    ///     // Resize to hold only 4 elements
    ///     // Elements 5 and 6 are truncated but not dropped (caller's responsibility)
    ///     let resized = storage.resize(4);
    ///     assert_eq!(resized.len(), 4);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Create a new VecStorage with a specific size
    /// * [`len`](Self::len) - Get the current number of elements
    #[inline]
    pub unsafe fn resize(mut self, sz: usize) -> Vec<MaybeUninit<T>> {
        unsafe {
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
            // been transferred to `new_data`.
            std::mem::forget(self);
            new_data
        }
    }

    /// Returns the total number of elements stored in the vector.
    ///
    /// This returns the length of the underlying vector, which equals the product
    /// of the number of rows and columns (nrows × ncols) in the matrix storage.
    ///
    /// # Returns
    ///
    /// The total number of elements in the storage.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// assert_eq!(storage.len(), 6); // 2 rows × 3 columns = 6 elements
    /// ```
    ///
    /// Empty storage:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let storage: VecStorage<f64, Dyn, Dyn> = VecStorage::new(Dyn(0), Dyn(0), vec![]);
    ///
    /// assert_eq!(storage.len(), 0);
    /// assert!(storage.is_empty());
    /// ```
    ///
    /// Verifying storage dimensions:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let nrows = 4;
    /// let ncols = 5;
    /// let data = vec![0; nrows * ncols];
    /// let storage = VecStorage::new(Dyn(nrows), Dyn(ncols), data);
    ///
    /// assert_eq!(storage.len(), nrows * ncols);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_empty`](Self::is_empty) - Check if the storage contains no elements
    /// * [`as_vec`](Self::as_vec) - Get a reference to the underlying vector
    /// * [`as_slice`](Self::as_slice) - Get a slice view of the elements
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the storage contains no elements.
    ///
    /// This is equivalent to checking if `len() == 0`. An empty storage represents
    /// a matrix with zero rows or zero columns (or both).
    ///
    /// # Returns
    ///
    /// `true` if the storage is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// Empty storage:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let storage: VecStorage<i32, Dyn, Dyn> = VecStorage::new(Dyn(0), Dyn(0), vec![]);
    ///
    /// assert!(storage.is_empty());
    /// assert_eq!(storage.len(), 0);
    /// ```
    ///
    /// Non-empty storage:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// assert!(!storage.is_empty());
    /// assert_eq!(storage.len(), 4);
    /// ```
    ///
    /// Checking before operations:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let storage = VecStorage::new(Dyn(3), Dyn(1), data);
    ///
    /// if !storage.is_empty() {
    ///     let sum: f64 = storage.as_slice().iter().sum();
    ///     assert_eq!(sum, 6.0);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`len`](Self::len) - Get the number of elements in the storage
    /// * [`as_vec`](Self::as_vec) - Get a reference to the underlying vector
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice containing all elements in column-major order.
    ///
    /// This provides an immutable view of all matrix elements as a contiguous slice.
    /// The elements are stored in column-major order, meaning all elements of the
    /// first column come first, followed by all elements of the second column, and so on.
    ///
    /// # Returns
    ///
    /// A slice `&[T]` containing all matrix elements in column-major order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// // Create a 2x3 matrix storage:
    /// // [1, 3, 5]
    /// // [2, 4, 6]
    /// let data = vec![1, 2, 3, 4, 5, 6]; // Column-major order
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// let slice = storage.as_slice();
    /// assert_eq!(slice, &[1, 2, 3, 4, 5, 6]);
    /// assert_eq!(slice.len(), 6);
    /// ```
    ///
    /// Iterating over all elements:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// let sum: f64 = storage.as_slice().iter().sum();
    /// assert_eq!(sum, 10.0);
    /// ```
    ///
    /// Using slice methods:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![5, 2, 8, 1, 9, 3];
    /// let storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// let slice = storage.as_slice();
    /// assert_eq!(slice.first(), Some(&5));
    /// assert_eq!(slice.last(), Some(&3));
    /// assert_eq!(*slice.iter().max().unwrap(), 9);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`as_mut_slice`](Self::as_mut_slice) - Get a mutable slice
    /// * [`as_vec`](Self::as_vec) - Get a reference to the underlying vector
    /// * [`len`](Self::len) - Get the number of elements
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..]
    }

    /// Returns a mutable slice containing all elements in column-major order.
    ///
    /// This provides a mutable view of all matrix elements as a contiguous slice.
    /// The elements are stored in column-major order, meaning all elements of the
    /// first column come first, followed by all elements of the second column, and so on.
    ///
    /// Unlike [`as_vec_mut`](Self::as_vec_mut), this is a safe function because it only
    /// provides access to modify elements, not to change the vector's length.
    ///
    /// # Returns
    ///
    /// A mutable slice `&mut [T]` containing all matrix elements in column-major order.
    ///
    /// # Examples
    ///
    /// Modifying elements:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// // Modify elements through the mutable slice
    /// let slice = storage.as_mut_slice();
    /// slice[0] = 10;
    /// slice[3] = 40;
    ///
    /// assert_eq!(storage.as_slice(), &[10, 2, 3, 40]);
    /// ```
    ///
    /// Applying operations to all elements:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(3), data);
    ///
    /// // Double all elements
    /// storage.as_mut_slice().iter_mut().for_each(|x| *x *= 2.0);
    ///
    /// assert_eq!(storage.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    ///
    /// Filling with a specific value:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// storage.as_mut_slice().fill(0);
    ///
    /// assert_eq!(storage.as_slice(), &[0, 0, 0, 0]);
    /// ```
    ///
    /// Swapping elements:
    ///
    /// ```
    /// use nalgebra::base::{VecStorage, Dyn};
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let mut storage = VecStorage::new(Dyn(2), Dyn(2), data);
    ///
    /// storage.as_mut_slice().swap(0, 3);
    ///
    /// assert_eq!(storage.as_slice(), &[4, 2, 3, 1]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`as_slice`](Self::as_slice) - Get an immutable slice
    /// * [`as_vec_mut`](Self::as_vec_mut) - Get a mutable vector reference (unsafe)
    /// * [`len`](Self::len) - Get the number of elements
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
    DefaultAllocator: Allocator<Dyn, C, Buffer<T> = Self>,
{
    #[inline]
    fn into_owned(self) -> Owned<T, Dyn, C>
    where
        DefaultAllocator: Allocator<Dyn, C>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, Dyn, C>
    where
        DefaultAllocator: Allocator<Dyn, C>,
    {
        self.clone()
    }

    #[inline]
    fn forget_elements(mut self) {
        // SAFETY: setting the length to zero is always sound, as it does not
        // cause any memory to be deemed initialized. If the previous length was
        // non-zero, it is equivalent to using mem::forget to leak each element.
        // Then, when this function returns, self.data is dropped, freeing the
        // allocated memory, but the elements are not dropped because they are
        // now considered uninitialized.
        unsafe { self.data.set_len(0) };
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
    DefaultAllocator: Allocator<R, Dyn, Buffer<T> = Self>,
{
    #[inline]
    fn into_owned(self) -> Owned<T, R, Dyn>
    where
        DefaultAllocator: Allocator<R, Dyn>,
    {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, R, Dyn>
    where
        DefaultAllocator: Allocator<R, Dyn>,
    {
        self.clone()
    }

    #[inline]
    fn forget_elements(mut self) {
        // SAFETY: setting the length to zero is always sound, as it does not
        // cause any memory to be deemed initialized. If the previous length was
        // non-zero, it is equivalent to using mem::forget to leak each element.
        // Then, when this function returns, self.data is dropped, freeing the
        // allocated memory, but the elements are not dropped because they are
        // now considered uninitialized.
        unsafe { self.data.set_len(0) };
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
        assert!(
            self.data.len() % self.nrows.value() == 0,
            "The number of elements produced by the given iterator was not a multiple of the number of rows."
        );
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
