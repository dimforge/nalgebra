use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};
use std::slice;

use crate::ReshapableStorage;
use crate::base::allocator::Allocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Const, Dim, DimName, Dyn, IsNotStaticOne, U1};
use crate::base::iter::MatrixIter;
use crate::base::storage::{IsContiguous, Owned, RawStorage, RawStorageMut, Storage};
use crate::base::{Matrix, Scalar};
use crate::constraint::{DimEq, ShapeConstraint};

macro_rules! view_storage_impl (
    ($doc: expr_2021; $Storage: ident as $SRef: ty; $legacy_name:ident => $T: ident.$get_addr: ident ($Ptr: ty as $Ref: ty)) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $T<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>  {
            ptr:       $Ptr,
            shape:     (R, C),
            strides:   (RStride, CStride),
            _phantoms: PhantomData<$Ref>,
        }

        #[doc = $doc]
        ///
        /// This type alias exists only for legacy purposes and is deprecated. It will be removed
        /// in a future release. Please use
        /// [`
        #[doc = stringify!($T)]
        /// `] instead. See [issue #1076](https://github.com/dimforge/nalgebra/issues/1076)
        /// for the rationale.
        #[deprecated = "Use ViewStorage(Mut) instead."]
        pub type $legacy_name<'a, T, R, C, RStride, CStride> = $T<'a, T, R, C, RStride, CStride>;

        unsafe impl<'a, T: Send, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Send
            for $T<'a, T, R, C, RStride, CStride>
        {}

        unsafe impl<'a, T: Sync, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Sync
            for $T<'a, T, R, C, RStride, CStride>
        {}

        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> $T<'a, T, R, C, RStride, CStride> {
            /// Create a new matrix view without bounds checking and from a raw pointer.
            ///
            /// # Safety
            ///
            /// `*ptr` must point to memory that is valid `[T; R * C]`.
            #[inline]
            pub const unsafe fn from_raw_parts(ptr:     $Ptr,
                                         shape:   (R, C),
                                         strides: (RStride, CStride))
                                        -> Self
                where RStride: Dim,
                      CStride: Dim {

                $T {
                    ptr,
                    shape,
                    strides,
                    _phantoms: PhantomData
                }
            }
        }

        // Dyn is arbitrary. It's just to be able to call the constructors with `Slice::`
        impl<'a, T, R: Dim, C: Dim> $T<'a, T, R, C, Dyn, Dyn> {
            /// Create a new matrix view without bounds checking.
            ///
            /// # Safety
            ///
            /// `storage` contains sufficient elements beyond `start + R * C` such that all
            /// accesses are within bounds.
            #[inline]
            pub unsafe fn new_unchecked<RStor, CStor, S>(storage: $SRef, start: (usize, usize), shape: (R, C))
                -> $T<'a, T, R, C, S::RStride, S::CStride>
                where RStor: Dim,
                      CStor: Dim,
                      S:     $Storage<T, RStor, CStor> { unsafe {

                let strides = storage.strides();
                $T::new_with_strides_unchecked(storage, start, shape, strides)
            }}

            /// Create a new matrix view without bounds checking.
            ///
            /// # Safety
            ///
            /// `strides` must be a valid stride indexing.
            #[inline]
            pub unsafe fn new_with_strides_unchecked<S, RStor, CStor, RStride, CStride>(storage: $SRef,
                                                                                        start:   (usize, usize),
                                                                                        shape:   (R, C),
                                                                                        strides: (RStride, CStride))
                -> $T<'a, T, R, C, RStride, CStride>
                where RStor: Dim,
                      CStor: Dim,
                      S: $Storage<T, RStor, CStor>,
                      RStride: Dim,
                      CStride: Dim { unsafe {
                $T::from_raw_parts(storage.$get_addr(start.0, start.1), shape, strides)
            }}
        }

        impl <'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
            $T<'a, T, R, C, RStride, CStride>
        where
            Self: RawStorage<T, R, C> + IsContiguous
        {
            /// Extracts the underlying data slice from this contiguous matrix view storage.
            ///
            /// This method consumes the view storage and returns a reference to the underlying
            /// contiguous slice of data. This is only available for contiguous matrix views,
            /// where the matrix elements are stored sequentially in memory without gaps.
            ///
            /// A matrix view is contiguous when its elements are laid out in column-major order
            /// without any stride gaps (i.e., elements in each column are consecutive in memory).
            ///
            /// # Returns
            /// A slice containing all the matrix elements in column-major order.
            ///
            /// # Example
            /// ```
            /// use nalgebra::Matrix3x2;
            ///
            /// let m = Matrix3x2::new(
            ///     1.0, 4.0,
            ///     2.0, 5.0,
            ///     3.0, 6.0,
            /// );
            ///
            /// let view = m.view((0, 0), (3, 2));
            /// let slice = view.data.into_slice();
            /// // Elements are in column-major order
            /// assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            /// ```
            pub fn into_slice(self) -> &'a [T] {
                let (nrows, ncols) = self.shape();
                if nrows.value() != 0 && ncols.value() != 0 {
                    let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
                    unsafe { slice::from_raw_parts(self.ptr, sz + 1) }
                } else {
                    unsafe { slice::from_raw_parts(self.ptr, 0) }
                }
            }
        }
    }
);

view_storage_impl!("A matrix data storage for a matrix view. Only contains an internal reference \
                     to another matrix data storage.";
    RawStorage as &'a S; SliceStorage => ViewStorage.get_address_unchecked(*const T as &'a T));

view_storage_impl!("A mutable matrix data storage for mutable matrix view. Only contains an \
                     internal mutable reference to another matrix data storage.";
    RawStorageMut as &'a mut S; SliceStorageMut => ViewStorageMut.get_address_unchecked_mut(*mut T as &'a mut T)
);

impl<T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Copy
    for ViewStorage<'_, T, R, C, RStride, CStride>
{
}

impl<T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Clone
    for ViewStorage<'_, T, R, C, RStride, CStride>
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    ViewStorageMut<'a, T, R, C, RStride, CStride>
where
    Self: RawStorageMut<T, R, C> + IsContiguous,
{
    /// Extracts the underlying mutable data slice from this contiguous matrix view storage.
    ///
    /// This method consumes the mutable view storage and returns a mutable reference to the
    /// underlying contiguous slice of data. This is only available for contiguous matrix views,
    /// where the matrix elements are stored sequentially in memory without gaps.
    ///
    /// A matrix view is contiguous when its elements are laid out in column-major order
    /// without any stride gaps (i.e., elements in each column are consecutive in memory).
    ///
    /// # Returns
    /// A mutable slice containing all the matrix elements in column-major order.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let mut m = Matrix3x2::new(
    ///     1.0, 4.0,
    ///     2.0, 5.0,
    ///     3.0, 6.0,
    /// );
    ///
    /// let mut view = m.view_mut((0, 0), (3, 2));
    /// let slice = view.data.into_slice_mut();
    /// // Modify elements through the slice (in column-major order)
    /// slice[0] = 10.0;
    /// assert_eq!(slice, &[10.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    ///
    /// # See Also
    /// - [`into_slice`](ViewStorage::into_slice) - Immutable version
    pub fn into_slice_mut(self) -> &'a mut [T] {
        let (nrows, ncols) = self.shape();
        if nrows.value() != 0 && ncols.value() != 0 {
            let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
            unsafe { slice::from_raw_parts_mut(self.ptr, sz + 1) }
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr, 0) }
        }
    }
}

macro_rules! storage_impl(
    ($($T: ident),* $(,)*) => {$(
        unsafe impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> RawStorage<T, R, C>
            for $T<'a, T, R, C, RStride, CStride> {

            type RStride = RStride;
            type CStride = CStride;

            #[inline]
            fn ptr(&self) -> *const T {
                self.ptr
            }

            #[inline]
            fn shape(&self) -> (R, C) {
                self.shape
            }

            #[inline]
            fn strides(&self) -> (Self::RStride, Self::CStride) {
                self.strides
            }

            #[inline]
            fn is_contiguous(&self) -> bool {
                // Common cases that can be deduced at compile-time even if one of the dimensions
                // is Dyn.
                if (RStride::is::<U1>() && C::is::<U1>()) || // Column vector.
                   (CStride::is::<U1>() && R::is::<U1>()) {  // Row vector.
                    true
                }
                else {
                    let (nrows, _)     = self.shape();
                    let (srows, scols) = self.strides();

                    srows.value() == 1 && scols.value() == nrows.value()
                }
            }

            #[inline]
            unsafe fn as_slice_unchecked(&self) -> &[T] { unsafe {
                let (nrows, ncols) = self.shape();
                if nrows.value() != 0 && ncols.value() != 0 {
                    let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
                    slice::from_raw_parts(self.ptr, sz + 1)
                }
                else {
                    slice::from_raw_parts(self.ptr, 0)
                }
            }}
        }

        unsafe impl<'a, T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Storage<T, R, C>
            for $T<'a, T, R, C, RStride, CStride> {
            #[inline]
            fn into_owned(self) -> Owned<T, R, C>
                where DefaultAllocator: Allocator<R, C> {
                self.clone_owned()
            }

            #[inline]
            fn clone_owned(&self) -> Owned<T, R, C>
                where DefaultAllocator: Allocator<R, C> {
                let (nrows, ncols) = self.shape();
                let it = MatrixIter::new(self).cloned();
                DefaultAllocator::allocate_from_iterator(nrows, ncols, it)
            }

            #[inline]
            fn forget_elements(self) {
                // No cleanup required.
            }
        }
    )*}
);

storage_impl!(ViewStorage, ViewStorageMut);

unsafe impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> RawStorageMut<T, R, C>
    for ViewStorageMut<'_, T, R, C, RStride, CStride>
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        unsafe {
            let (nrows, ncols) = self.shape();
            if nrows.value() != 0 && ncols.value() != 0 {
                let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
                slice::from_raw_parts_mut(self.ptr, sz + 1)
            } else {
                slice::from_raw_parts_mut(self.ptr, 0)
            }
        }
    }
}

unsafe impl<T, R: Dim, CStride: Dim> IsContiguous for ViewStorage<'_, T, R, U1, U1, CStride> {}
unsafe impl<T, R: Dim, CStride: Dim> IsContiguous for ViewStorageMut<'_, T, R, U1, U1, CStride> {}

unsafe impl<T, R: DimName, C: Dim + IsNotStaticOne> IsContiguous
    for ViewStorage<'_, T, R, C, U1, R>
{
}
unsafe impl<T, R: DimName, C: Dim + IsNotStaticOne> IsContiguous
    for ViewStorageMut<'_, T, R, C, U1, R>
{
}

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    #[inline]
    fn assert_view_index(
        &self,
        start: (usize, usize),
        shape: (usize, usize),
        steps: (usize, usize),
    ) {
        let my_shape = self.shape();
        // NOTE: we don't do any subtraction to avoid underflow for zero-sized matrices.
        //
        // Terms that would have been negative are moved to the other side of the inequality
        // instead.
        assert!(
            start.0 + (steps.0 + 1) * shape.0 <= my_shape.0 + steps.0,
            "Matrix slicing out of bounds."
        );
        assert!(
            start.1 + (steps.1 + 1) * shape.1 <= my_shape.1 + steps.1,
            "Matrix slicing out of bounds."
        );
    }
}

macro_rules! matrix_view_impl (
    ($me: ident: $Me: ty, $MatrixView: ident, $ViewStorage: ident, $Storage: ident.$get_addr: ident (), $data: expr_2021;
     $row: ident,
     $row_part: ident,
     $rows: ident,
     $rows_with_step: ident,
     $fixed_rows: ident,
     $fixed_rows_with_step: ident,
     $rows_generic: ident,
     $rows_generic_with_step: ident,
     $column: ident,
     $column_part: ident,
     $columns: ident,
     $columns_with_step: ident,
     $fixed_columns: ident,
     $fixed_columns_with_step: ident,
     $columns_generic: ident,
     $columns_generic_with_step: ident,
     $slice: ident => $view:ident,
     $slice_with_steps: ident => $view_with_steps:ident,
     $fixed_slice: ident => $fixed_view:ident,
     $fixed_slice_with_steps: ident => $fixed_view_with_steps:ident,
     $generic_slice: ident => $generic_view:ident,
     $generic_slice_with_steps: ident => $generic_view_with_steps:ident,
     $rows_range_pair: ident,
     $columns_range_pair: ident) => {
        /*
         *
         * Row slicing.
         *
         */
        /// Returns a view containing the i-th row of this matrix.
        ///
        /// A matrix view (also called a slice) is a lightweight reference to a portion of a matrix.
        /// Unlike creating a new matrix, a view doesn't copy any data - it simply provides a window
        /// into the original matrix's memory. This makes views very efficient for working with
        /// sub-portions of matrices.
        ///
        /// # Parameters
        /// - `i`: The zero-based index of the row to extract.
        ///
        /// # Panics
        /// Panics if `i` is greater than or equal to the number of rows in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x4;
        ///
        /// let m = Matrix3x4::new(
        ///     1.0, 2.0, 3.0, 4.0,
        ///     5.0, 6.0, 7.0, 8.0,
        ///     9.0, 10.0, 11.0, 12.0,
        /// );
        ///
        /// // Get the second row (index 1)
        /// let row = m.row(1);
        /// assert_eq!(row, nalgebra::RowVector4::new(5.0, 6.0, 7.0, 8.0));
        ///
        /// // Views don't copy data - they reference the original matrix
        /// assert_eq!(row[0], 5.0);
        /// assert_eq!(row[3], 8.0);
        /// ```
        ///
        /// # See Also
        /// - [`row_part`](Self::row_part) - Extract only part of a row
        /// - [`rows`](Self::rows) - Extract multiple consecutive rows
        /// - [`column`](Self::column) - Extract a single column
        #[inline]
        pub fn $row($me: $Me, i: usize) -> $MatrixView<'_, T, U1, C, S::RStride, S::CStride> {
            $me.$fixed_rows::<1>(i)
        }

        /// Returns a view containing the first `n` elements of the i-th row of this matrix.
        ///
        /// This is useful when you need to work with only a portion of a row. The view provides
        /// a window into the original matrix data without copying.
        ///
        /// # Parameters
        /// - `i`: The zero-based index of the row to extract from.
        /// - `n`: The number of elements to include from the beginning of the row.
        ///
        /// # Panics
        /// Panics if `i >= nrows` or if `n > ncols`.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x4;
        ///
        /// let m = Matrix3x4::new(
        ///     1.0, 2.0, 3.0, 4.0,
        ///     5.0, 6.0, 7.0, 8.0,
        ///     9.0, 10.0, 11.0, 12.0,
        /// );
        ///
        /// // Get only the first 2 elements of the second row
        /// let partial_row = m.row_part(1, 2);
        /// assert_eq!(partial_row.len(), 2);
        /// assert_eq!(partial_row[0], 5.0);
        /// assert_eq!(partial_row[1], 6.0);
        /// ```
        ///
        /// # See Also
        /// - [`row`](Self::row) - Extract a complete row
        /// - [`column_part`](Self::column_part) - Extract part of a column
        /// - [`view`](Self::view) - Create a view of an arbitrary sub-matrix
        #[inline]
        pub fn $row_part($me: $Me, i: usize, n: usize) -> $MatrixView<'_, T, U1, Dyn, S::RStride, S::CStride> {
            $me.$generic_view((i, 0), (Const::<1>, Dyn(n)))
        }

        /// Extracts a view containing a set of consecutive rows from this matrix.
        ///
        /// Matrix views are lightweight references that don't copy data. This function creates
        /// a view of multiple consecutive rows, which is useful for operations on horizontal
        /// bands of a matrix.
        ///
        /// # Parameters
        /// - `first_row`: The zero-based index of the first row to include.
        /// - `nrows`: The number of consecutive rows to include.
        ///
        /// # Panics
        /// Panics if `first_row + nrows` exceeds the number of rows in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix4x3;
        ///
        /// let m = Matrix4x3::new(
        ///     1.0, 2.0, 3.0,
        ///     4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0,
        ///     10.0, 11.0, 12.0,
        /// );
        ///
        /// // Extract rows 1 and 2 (the middle two rows)
        /// let middle_rows = m.rows(1, 2);
        /// assert_eq!(middle_rows.nrows(), 2);
        /// assert_eq!(middle_rows.ncols(), 3);
        /// assert_eq!(middle_rows[(0, 0)], 4.0);
        /// assert_eq!(middle_rows[(1, 2)], 9.0);
        /// ```
        ///
        /// # See Also
        /// - [`row`](Self::row) - Extract a single row
        /// - [`rows_with_step`](Self::rows_with_step) - Extract rows with gaps
        /// - [`fixed_rows`](Self::fixed_rows) - Extract rows with compile-time size
        /// - [`columns`](Self::columns) - Extract consecutive columns
        #[inline]
        pub fn $rows($me: $Me, first_row: usize, nrows: usize)
            -> $MatrixView<'_, T, Dyn, C, S::RStride, S::CStride> {

            $me.$rows_generic(first_row, Dyn(nrows))
        }

        /// Extracts a view of rows from this matrix, regularly skipping `step` rows between each selected row.
        ///
        /// This creates a strided view that samples rows at regular intervals. For example, with
        /// `step = 1`, every other row is selected (rows 0, 2, 4, ...). With `step = 2`, every
        /// third row is selected (rows 0, 3, 6, ...).
        ///
        /// # Parameters
        /// - `first_row`: The zero-based index of the first row to include.
        /// - `nrows`: The number of rows to include in the view.
        /// - `step`: The number of rows to skip between each selected row (0 means consecutive).
        ///
        /// # Panics
        /// Panics if the selected rows would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix6x2;
        ///
        /// let m = Matrix6x2::new(
        ///     1.0, 2.0,
        ///     3.0, 4.0,
        ///     5.0, 6.0,
        ///     7.0, 8.0,
        ///     9.0, 10.0,
        ///     11.0, 12.0,
        /// );
        ///
        /// // Select 3 rows starting at row 0, skipping 1 row each time (rows 0, 2, 4)
        /// let strided = m.rows_with_step(0, 3, 1);
        /// assert_eq!(strided.nrows(), 3);
        /// assert_eq!(strided[(0, 0)], 1.0);  // Row 0
        /// assert_eq!(strided[(1, 0)], 5.0);  // Row 2
        /// assert_eq!(strided[(2, 0)], 9.0);  // Row 4
        /// ```
        ///
        /// # See Also
        /// - [`rows`](Self::rows) - Extract consecutive rows
        /// - [`fixed_rows_with_step`](Self::fixed_rows_with_step) - Strided rows with compile-time size
        /// - [`columns_with_step`](Self::columns_with_step) - Strided column extraction
        #[inline]
        pub fn $rows_with_step($me: $Me, first_row: usize, nrows: usize, step: usize)
            -> $MatrixView<'_, T, Dyn, C, Dyn, S::CStride> {

            $me.$rows_generic_with_step(first_row, Dyn(nrows), step)
        }

        /// Extracts a view containing a compile-time number of consecutive rows from this matrix.
        ///
        /// This is similar to [`rows`](Self::rows), but the number of rows is known at compile time,
        /// which can enable better compiler optimizations and type checking. The size is specified
        /// as a const generic parameter `RVIEW`.
        ///
        /// # Parameters
        /// - `first_row`: The zero-based index of the first row to include.
        ///
        /// # Type Parameters
        /// - `RVIEW`: The number of rows to extract (known at compile time).
        ///
        /// # Panics
        /// Panics if `first_row + RVIEW` exceeds the number of rows in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix5x3;
        ///
        /// let m = Matrix5x3::new(
        ///     1.0, 2.0, 3.0,
        ///     4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0,
        ///     10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0,
        /// );
        ///
        /// // Extract exactly 2 rows starting at row 1
        /// let two_rows = m.fixed_rows::<2>(1);
        /// assert_eq!(two_rows.nrows(), 2);
        /// assert_eq!(two_rows[(0, 0)], 4.0);
        /// assert_eq!(two_rows[(1, 0)], 7.0);
        /// ```
        ///
        /// # See Also
        /// - [`rows`](Self::rows) - Extract rows with runtime size
        /// - [`fixed_rows_with_step`](Self::fixed_rows_with_step) - Fixed-size strided rows
        /// - [`fixed_columns`](Self::fixed_columns) - Extract compile-time columns
        #[inline]
        pub fn $fixed_rows<const RVIEW: usize>($me: $Me, first_row: usize)
            -> $MatrixView<'_, T, Const<RVIEW>, C, S::RStride, S::CStride> {

            $me.$rows_generic(first_row, Const::<RVIEW>)
        }

        /// Extracts a compile-time number of rows from this matrix, regularly skipping `step` rows.
        ///
        /// This combines the benefits of compile-time sizing with strided access. The number of
        /// rows is known at compile time (specified by `RVIEW`), while the stride is a runtime
        /// parameter.
        ///
        /// # Parameters
        /// - `first_row`: The zero-based index of the first row to include.
        /// - `step`: The number of rows to skip between each selected row (0 means consecutive).
        ///
        /// # Type Parameters
        /// - `RVIEW`: The number of rows to extract (known at compile time).
        ///
        /// # Panics
        /// Panics if the selected rows would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix6x2;
        ///
        /// let m = Matrix6x2::new(
        ///     1.0, 2.0,
        ///     3.0, 4.0,
        ///     5.0, 6.0,
        ///     7.0, 8.0,
        ///     9.0, 10.0,
        ///     11.0, 12.0,
        /// );
        ///
        /// // Extract exactly 2 rows starting at row 1, skipping 1 row each time
        /// let strided = m.fixed_rows_with_step::<2>(1, 1);
        /// assert_eq!(strided.nrows(), 2);
        /// assert_eq!(strided[(0, 0)], 3.0);  // Row 1
        /// assert_eq!(strided[(1, 0)], 7.0);  // Row 3
        /// ```
        ///
        /// # See Also
        /// - [`fixed_rows`](Self::fixed_rows) - Consecutive compile-time rows
        /// - [`rows_with_step`](Self::rows_with_step) - Strided rows with runtime size
        /// - [`fixed_columns_with_step`](Self::fixed_columns_with_step) - Strided compile-time columns
        #[inline]
        pub fn $fixed_rows_with_step<const RVIEW: usize>($me: $Me, first_row: usize, step: usize)
            -> $MatrixView<'_, T, Const<RVIEW>, C, Dyn, S::CStride> {

            $me.$rows_generic_with_step(first_row, Const::<RVIEW>, step)
        }

        /// Extracts a view of `nrows` consecutive rows from this matrix.
        ///
        /// This is the most generic row extraction method. The number of rows can be specified
        /// either at compile time (using `Const<N>`) or at runtime (using `Dyn`). This function
        /// is typically used internally by other more convenient methods like [`rows`](Self::rows)
        /// and [`fixed_rows`](Self::fixed_rows).
        ///
        /// # Parameters
        /// - `row_start`: The zero-based index of the first row to include.
        /// - `nrows`: The number of rows to extract (can be compile-time or runtime dimension).
        ///
        /// # Type Parameters
        /// - `RView`: A dimension type (either `Const<N>` or `Dyn`) specifying the number of rows.
        ///
        /// # Panics
        /// Panics if `row_start + nrows` exceeds the number of rows in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix4x3, Dyn, Const};
        ///
        /// let m = Matrix4x3::new(
        ///     1.0, 2.0, 3.0,
        ///     4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0,
        ///     10.0, 11.0, 12.0,
        /// );
        ///
        /// // Using runtime dimension
        /// let dynamic_rows = m.rows_generic(1, Dyn(2));
        /// assert_eq!(dynamic_rows.nrows(), 2);
        ///
        /// // Using compile-time dimension
        /// let static_rows = m.rows_generic(1, Const::<2>);
        /// assert_eq!(static_rows.nrows(), 2);
        /// ```
        ///
        /// # See Also
        /// - [`rows`](Self::rows) - Simpler interface for runtime-sized extraction
        /// - [`fixed_rows`](Self::fixed_rows) - Simpler interface for compile-time-sized extraction
        /// - [`rows_generic_with_step`](Self::rows_generic_with_step) - Generic strided row extraction
        #[inline]
        pub fn $rows_generic<RView: Dim>($me: $Me, row_start: usize, nrows: RView)
            -> $MatrixView<'_, T, RView, C, S::RStride, S::CStride> {

            let my_shape   = $me.shape_generic();
            $me.assert_view_index((row_start, 0), (nrows.value(), my_shape.1.value()), (0, 0));

            let shape = (nrows, my_shape.1);

            unsafe {
                let data = $ViewStorage::new_unchecked($data, (row_start, 0), shape);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /// Extracts a view of `nrows` rows from this matrix, regularly skipping `step` rows.
        ///
        /// This is the most generic strided row extraction method. The number of rows can be
        /// specified either at compile time (using `Const<N>`) or at runtime (using `Dyn`).
        /// This function is typically used internally by other methods like
        /// [`rows_with_step`](Self::rows_with_step) and [`fixed_rows_with_step`](Self::fixed_rows_with_step).
        ///
        /// # Parameters
        /// - `row_start`: The zero-based index of the first row to include.
        /// - `nrows`: The number of rows to extract (can be compile-time or runtime dimension).
        /// - `step`: The number of rows to skip between each selected row (0 means consecutive).
        ///
        /// # Type Parameters
        /// - `RView`: A dimension type (either `Const<N>` or `Dyn`) specifying the number of rows.
        ///
        /// # Panics
        /// Panics if the selected rows would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix6x2, Dyn, Const};
        ///
        /// let m = Matrix6x2::new(
        ///     1.0, 2.0,
        ///     3.0, 4.0,
        ///     5.0, 6.0,
        ///     7.0, 8.0,
        ///     9.0, 10.0,
        ///     11.0, 12.0,
        /// );
        ///
        /// // Using runtime dimension: select 3 rows with step 1 (rows 0, 2, 4)
        /// let dynamic_strided = m.rows_generic_with_step(0, Dyn(3), 1);
        /// assert_eq!(dynamic_strided[(0, 0)], 1.0);
        /// assert_eq!(dynamic_strided[(1, 0)], 5.0);
        ///
        /// // Using compile-time dimension
        /// let static_strided = m.rows_generic_with_step(0, Const::<3>, 1);
        /// assert_eq!(static_strided[(2, 0)], 9.0);
        /// ```
        ///
        /// # See Also
        /// - [`rows_with_step`](Self::rows_with_step) - Simpler interface for runtime-sized strided extraction
        /// - [`fixed_rows_with_step`](Self::fixed_rows_with_step) - Simpler interface for compile-time strided extraction
        /// - [`rows_generic`](Self::rows_generic) - Generic consecutive row extraction
        #[inline]
        pub fn $rows_generic_with_step<RView>($me: $Me, row_start: usize, nrows: RView, step: usize)
            -> $MatrixView<'_, T, RView, C, Dyn, S::CStride>
            where RView: Dim {

            let my_shape   = $me.shape_generic();
            let my_strides = $me.data.strides();
            $me.assert_view_index((row_start, 0), (nrows.value(), my_shape.1.value()), (step, 0));

            let strides = (Dyn((step + 1) * my_strides.0.value()), my_strides.1);
            let shape   = (nrows, my_shape.1);

            unsafe {
                let data = $ViewStorage::new_with_strides_unchecked($data, (row_start, 0), shape, strides);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /*
         *
         * Column slicing.
         *
         */
        /// Returns a view containing the i-th column of this matrix.
        ///
        /// A matrix view is a lightweight, non-copying reference to a portion of the matrix.
        /// This function extracts a single column as a column vector view.
        ///
        /// # Parameters
        /// - `i`: The zero-based index of the column to extract.
        ///
        /// # Panics
        /// Panics if `i` is greater than or equal to the number of columns in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x4;
        ///
        /// let m = Matrix3x4::new(
        ///     1.0, 2.0, 3.0, 4.0,
        ///     5.0, 6.0, 7.0, 8.0,
        ///     9.0, 10.0, 11.0, 12.0,
        /// );
        ///
        /// // Get the third column (index 2)
        /// let col = m.column(2);
        /// assert_eq!(col, nalgebra::Vector3::new(3.0, 7.0, 11.0));
        ///
        /// // Column views reference the original matrix data
        /// assert_eq!(col[0], 3.0);
        /// assert_eq!(col[2], 11.0);
        /// ```
        ///
        /// # See Also
        /// - [`column_part`](Self::column_part) - Extract only part of a column
        /// - [`columns`](Self::columns) - Extract multiple consecutive columns
        /// - [`row`](Self::row) - Extract a single row
        #[inline]
        pub fn $column($me: $Me, i: usize) -> $MatrixView<'_, T, R, U1, S::RStride, S::CStride> {
            $me.$fixed_columns::<1>(i)
        }

        /// Returns a view containing the first `n` elements of the i-th column of this matrix.
        ///
        /// This extracts a partial column view, useful when you only need to work with the
        /// beginning portion of a column vector.
        ///
        /// # Parameters
        /// - `i`: The zero-based index of the column to extract from.
        /// - `n`: The number of elements to include from the beginning of the column.
        ///
        /// # Panics
        /// Panics if `i >= ncols` or if `n > nrows`.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix4x3;
        ///
        /// let m = Matrix4x3::new(
        ///     1.0, 2.0, 3.0,
        ///     4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0,
        ///     10.0, 11.0, 12.0,
        /// );
        ///
        /// // Get only the first 2 elements of the second column
        /// let partial_col = m.column_part(1, 2);
        /// assert_eq!(partial_col.len(), 2);
        /// assert_eq!(partial_col[0], 2.0);
        /// assert_eq!(partial_col[1], 5.0);
        /// ```
        ///
        /// # See Also
        /// - [`column`](Self::column) - Extract a complete column
        /// - [`row_part`](Self::row_part) - Extract part of a row
        /// - [`view`](Self::view) - Create a view of an arbitrary sub-matrix
        #[inline]
        pub fn $column_part($me: $Me, i: usize, n: usize) -> $MatrixView<'_, T, Dyn, U1, S::RStride, S::CStride> {
            $me.$generic_view((0, i), (Dyn(n), Const::<1>))
        }

        /// Extracts a view containing a set of consecutive columns from this matrix.
        ///
        /// Matrix views are lightweight references that don't copy data. This function creates
        /// a view of multiple consecutive columns, which is useful for operations on vertical
        /// bands of a matrix.
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        /// - `ncols`: The number of consecutive columns to include.
        ///
        /// # Panics
        /// Panics if `first_col + ncols` exceeds the number of columns in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x5;
        ///
        /// let m = Matrix3x5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        /// );
        ///
        /// // Extract columns 1, 2, and 3 (indices 1-3)
        /// let middle_cols = m.columns(1, 3);
        /// assert_eq!(middle_cols.nrows(), 3);
        /// assert_eq!(middle_cols.ncols(), 3);
        /// assert_eq!(middle_cols[(0, 0)], 2.0);
        /// assert_eq!(middle_cols[(2, 2)], 13.0);
        /// ```
        ///
        /// # See Also
        /// - [`column`](Self::column) - Extract a single column
        /// - [`columns_with_step`](Self::columns_with_step) - Extract columns with gaps
        /// - [`fixed_columns`](Self::fixed_columns) - Extract columns with compile-time size
        /// - [`rows`](Self::rows) - Extract consecutive rows
        #[inline]
        pub fn $columns($me: $Me, first_col: usize, ncols: usize)
            -> $MatrixView<'_, T, R, Dyn, S::RStride, S::CStride> {

            $me.$columns_generic(first_col, Dyn(ncols))
        }

        /// Extracts a view of columns from this matrix, regularly skipping `step` columns between each selected column.
        ///
        /// This creates a strided view that samples columns at regular intervals. For example, with
        /// `step = 1`, every other column is selected (columns 0, 2, 4, ...). With `step = 2`, every
        /// third column is selected (columns 0, 3, 6, ...).
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        /// - `ncols`: The number of columns to include in the view.
        /// - `step`: The number of columns to skip between each selected column (0 means consecutive).
        ///
        /// # Panics
        /// Panics if the selected columns would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x6;
        ///
        /// let m = Matrix3x6::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        /// );
        ///
        /// // Select 3 columns starting at column 0, skipping 1 column each time (columns 0, 2, 4)
        /// let strided = m.columns_with_step(0, 3, 1);
        /// assert_eq!(strided.ncols(), 3);
        /// assert_eq!(strided[(0, 0)], 1.0);  // Column 0
        /// assert_eq!(strided[(0, 1)], 3.0);  // Column 2
        /// assert_eq!(strided[(0, 2)], 5.0);  // Column 4
        /// ```
        ///
        /// # See Also
        /// - [`columns`](Self::columns) - Extract consecutive columns
        /// - [`fixed_columns_with_step`](Self::fixed_columns_with_step) - Strided columns with compile-time size
        /// - [`rows_with_step`](Self::rows_with_step) - Strided row extraction
        #[inline]
        pub fn $columns_with_step($me: $Me, first_col: usize, ncols: usize, step: usize)
            -> $MatrixView<'_, T, R, Dyn, S::RStride, Dyn> {

            $me.$columns_generic_with_step(first_col, Dyn(ncols), step)
        }

        /// Extracts a view containing a compile-time number of consecutive columns from this matrix.
        ///
        /// This is similar to [`columns`](Self::columns), but the number of columns is known at compile time,
        /// which can enable better compiler optimizations and type checking. The size is specified
        /// as a const generic parameter `CVIEW`.
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        ///
        /// # Type Parameters
        /// - `CVIEW`: The number of columns to extract (known at compile time).
        ///
        /// # Panics
        /// Panics if `first_col + CVIEW` exceeds the number of columns in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x5;
        ///
        /// let m = Matrix3x5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        /// );
        ///
        /// // Extract exactly 2 columns starting at column 1
        /// let two_cols = m.fixed_columns::<2>(1);
        /// assert_eq!(two_cols.ncols(), 2);
        /// assert_eq!(two_cols[(0, 0)], 2.0);
        /// assert_eq!(two_cols[(0, 1)], 3.0);
        /// ```
        ///
        /// # See Also
        /// - [`columns`](Self::columns) - Extract columns with runtime size
        /// - [`fixed_columns_with_step`](Self::fixed_columns_with_step) - Fixed-size strided columns
        /// - [`fixed_rows`](Self::fixed_rows) - Extract compile-time rows
        #[inline]
        pub fn $fixed_columns<const CVIEW: usize>($me: $Me, first_col: usize)
            -> $MatrixView<'_, T, R, Const<CVIEW>, S::RStride, S::CStride> {

            $me.$columns_generic(first_col, Const::<CVIEW>)
        }

        /// Extracts a compile-time number of columns from this matrix, regularly skipping `step` columns.
        ///
        /// This combines the benefits of compile-time sizing with strided access. The number of
        /// columns is known at compile time (specified by `CVIEW`), while the stride is a runtime
        /// parameter.
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        /// - `step`: The number of columns to skip between each selected column (0 means consecutive).
        ///
        /// # Type Parameters
        /// - `CVIEW`: The number of columns to extract (known at compile time).
        ///
        /// # Panics
        /// Panics if the selected columns would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x6;
        ///
        /// let m = Matrix3x6::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        /// );
        ///
        /// // Extract exactly 2 columns starting at column 1, skipping 1 column each time
        /// let strided = m.fixed_columns_with_step::<2>(1, 1);
        /// assert_eq!(strided.ncols(), 2);
        /// assert_eq!(strided[(0, 0)], 2.0);  // Column 1
        /// assert_eq!(strided[(0, 1)], 4.0);  // Column 3
        /// ```
        ///
        /// # See Also
        /// - [`fixed_columns`](Self::fixed_columns) - Consecutive compile-time columns
        /// - [`columns_with_step`](Self::columns_with_step) - Strided columns with runtime size
        /// - [`fixed_rows_with_step`](Self::fixed_rows_with_step) - Strided compile-time rows
        #[inline]
        pub fn $fixed_columns_with_step<const CVIEW: usize>($me: $Me, first_col: usize, step: usize)
            -> $MatrixView<'_, T, R, Const<CVIEW>, S::RStride, Dyn> {

            $me.$columns_generic_with_step(first_col, Const::<CVIEW>, step)
        }

        /// Extracts a view of `ncols` consecutive columns from this matrix.
        ///
        /// This is the most generic column extraction method. The number of columns can be specified
        /// either at compile time (using `Const<N>`) or at runtime (using `Dyn`). This function
        /// is typically used internally by other more convenient methods like [`columns`](Self::columns)
        /// and [`fixed_columns`](Self::fixed_columns).
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        /// - `ncols`: The number of columns to extract (can be compile-time or runtime dimension).
        ///
        /// # Type Parameters
        /// - `CView`: A dimension type (either `Const<N>` or `Dyn`) specifying the number of columns.
        ///
        /// # Panics
        /// Panics if `first_col + ncols` exceeds the number of columns in the matrix.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix3x5, Dyn, Const};
        ///
        /// let m = Matrix3x5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        /// );
        ///
        /// // Using runtime dimension
        /// let dynamic_cols = m.columns_generic(1, Dyn(2));
        /// assert_eq!(dynamic_cols.ncols(), 2);
        ///
        /// // Using compile-time dimension
        /// let static_cols = m.columns_generic(1, Const::<2>);
        /// assert_eq!(static_cols.ncols(), 2);
        /// ```
        ///
        /// # See Also
        /// - [`columns`](Self::columns) - Simpler interface for runtime-sized extraction
        /// - [`fixed_columns`](Self::fixed_columns) - Simpler interface for compile-time-sized extraction
        /// - [`columns_generic_with_step`](Self::columns_generic_with_step) - Generic strided column extraction
        #[inline]
        pub fn $columns_generic<CView: Dim>($me: $Me, first_col: usize, ncols: CView)
            -> $MatrixView<'_, T, R, CView, S::RStride, S::CStride> {

            let my_shape = $me.shape_generic();
            $me.assert_view_index((0, first_col), (my_shape.0.value(), ncols.value()), (0, 0));
            let shape = (my_shape.0, ncols);

            unsafe {
                let data = $ViewStorage::new_unchecked($data, (0, first_col), shape);
                Matrix::from_data_statically_unchecked(data)
            }
        }


        /// Extracts a view of `ncols` columns from this matrix, regularly skipping `step` columns.
        ///
        /// This is the most generic strided column extraction method. The number of columns can be
        /// specified either at compile time (using `Const<N>`) or at runtime (using `Dyn`).
        /// This function is typically used internally by other methods like
        /// [`columns_with_step`](Self::columns_with_step) and [`fixed_columns_with_step`](Self::fixed_columns_with_step).
        ///
        /// # Parameters
        /// - `first_col`: The zero-based index of the first column to include.
        /// - `ncols`: The number of columns to extract (can be compile-time or runtime dimension).
        /// - `step`: The number of columns to skip between each selected column (0 means consecutive).
        ///
        /// # Type Parameters
        /// - `CView`: A dimension type (either `Const<N>` or `Dyn`) specifying the number of columns.
        ///
        /// # Panics
        /// Panics if the selected columns would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix3x6, Dyn, Const};
        ///
        /// let m = Matrix3x6::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        /// );
        ///
        /// // Using runtime dimension: select 3 columns with step 1 (columns 0, 2, 4)
        /// let dynamic_strided = m.columns_generic_with_step(0, Dyn(3), 1);
        /// assert_eq!(dynamic_strided[(0, 0)], 1.0);
        /// assert_eq!(dynamic_strided[(0, 1)], 3.0);
        ///
        /// // Using compile-time dimension
        /// let static_strided = m.columns_generic_with_step(0, Const::<3>, 1);
        /// assert_eq!(static_strided[(0, 2)], 5.0);
        /// ```
        ///
        /// # See Also
        /// - [`columns_with_step`](Self::columns_with_step) - Simpler interface for runtime-sized strided extraction
        /// - [`fixed_columns_with_step`](Self::fixed_columns_with_step) - Simpler interface for compile-time strided extraction
        /// - [`columns_generic`](Self::columns_generic) - Generic consecutive column extraction
        #[inline]
        pub fn $columns_generic_with_step<CView: Dim>($me: $Me, first_col: usize, ncols: CView, step: usize)
            -> $MatrixView<'_, T, R, CView, S::RStride, Dyn> {

            let my_shape   = $me.shape_generic();
            let my_strides = $me.data.strides();

            $me.assert_view_index((0, first_col), (my_shape.0.value(), ncols.value()), (0, step));

            let strides = (my_strides.0, Dyn((step + 1) * my_strides.1.value()));
            let shape   = (my_shape.0, ncols);

            unsafe {
                let data = $ViewStorage::new_with_strides_unchecked($data, (0, first_col), shape, strides);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /*
         *
         * General slicing.
         *
         */
        /// Slices this matrix starting at its component `(irow, icol)` and with `(nrows, ncols)`
        /// consecutive elements.
        #[inline]
        #[deprecated = slice_deprecation_note!($view)]
        pub fn $slice($me: $Me, start: (usize, usize), shape: (usize, usize))
            -> $MatrixView<'_, T, Dyn, Dyn, S::RStride, S::CStride> {
            $me.$view(start, shape)
        }

        /// Returns a view of a rectangular sub-matrix of this matrix.
        ///
        /// A matrix view (or slice) is a non-copying reference to a portion of the matrix.
        /// This is one of the most general and commonly used view creation methods, allowing
        /// you to extract an arbitrary rectangular region from the matrix.
        ///
        /// # Parameters
        /// - `start`: A tuple `(row, col)` specifying the top-left corner of the sub-matrix (zero-based).
        /// - `shape`: A tuple `(nrows, ncols)` specifying the size of the sub-matrix.
        ///
        /// # Panics
        /// Panics if the specified region extends beyond the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix4x5;
        ///
        /// let m = Matrix4x5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        ///     16.0, 17.0, 18.0, 19.0, 20.0,
        /// );
        ///
        /// // Extract a 2x3 sub-matrix starting at position (1, 1)
        /// let sub = m.view((1, 1), (2, 3));
        /// assert_eq!(sub.nrows(), 2);
        /// assert_eq!(sub.ncols(), 3);
        /// assert_eq!(sub[(0, 0)], 7.0);
        /// assert_eq!(sub[(1, 2)], 14.0);
        /// ```
        ///
        /// # See Also
        /// - [`fixed_view`](Self::fixed_view) - Extract a view with compile-time dimensions
        /// - [`view_with_steps`](Self::view_with_steps) - Extract a strided view
        /// - [`rows`](Self::rows) - Extract complete rows
        /// - [`columns`](Self::columns) - Extract complete columns
        #[inline]
        pub fn $view($me: $Me, start: (usize, usize), shape: (usize, usize))
            -> $MatrixView<'_, T, Dyn, Dyn, S::RStride, S::CStride> {

            $me.assert_view_index(start, shape, (0, 0));
            let shape = (Dyn(shape.0), Dyn(shape.1));

            unsafe {
                let data = $ViewStorage::new_unchecked($data, start, shape);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /// Slices this matrix starting at its component `(start.0, start.1)` and with
        /// `(shape.0, shape.1)` components. Each row (resp. column) of the sliced matrix is
        /// separated by `steps.0` (resp. `steps.1`) ignored rows (resp. columns) of the
        /// original matrix.
        #[inline]
        #[deprecated = slice_deprecation_note!($view_with_steps)]
        pub fn $slice_with_steps($me: $Me, start: (usize, usize), shape: (usize, usize), steps: (usize, usize))
            -> $MatrixView<'_, T, Dyn, Dyn, Dyn, Dyn> {
            $me.$view_with_steps(start, shape, steps)
        }

        /// Returns a strided view of this matrix with both row and column gaps.
        ///
        /// This creates a view that samples the matrix at regular row and column intervals,
        /// allowing you to extract non-contiguous sub-matrices. The `steps` parameter controls
        /// how many elements to skip between each selected row or column.
        ///
        /// # Parameters
        /// - `start`: A tuple `(row, col)` specifying the top-left corner (zero-based).
        /// - `shape`: A tuple `(nrows, ncols)` specifying the size of the view.
        /// - `steps`: A tuple `(row_step, col_step)` specifying gaps between selected elements.
        ///   A step of 0 means consecutive elements, 1 means skip one element, etc.
        ///
        /// # Panics
        /// Panics if the selected region would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix6;
        ///
        /// let m = Matrix6::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ///     19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ///     25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        ///     31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        /// );
        ///
        /// // Extract a 2x2 view starting at (0,0), skipping 1 row and 1 column each time
        /// let strided = m.view_with_steps((0, 0), (2, 2), (1, 1));
        /// assert_eq!(strided[(0, 0)], 1.0);   // Row 0, Col 0
        /// assert_eq!(strided[(0, 1)], 3.0);   // Row 0, Col 2
        /// assert_eq!(strided[(1, 0)], 13.0);  // Row 2, Col 0
        /// assert_eq!(strided[(1, 1)], 15.0);  // Row 2, Col 2
        /// ```
        ///
        /// # See Also
        /// - [`view`](Self::view) - Extract a contiguous view
        /// - [`rows_with_step`](Self::rows_with_step) - Strided row extraction
        /// - [`columns_with_step`](Self::columns_with_step) - Strided column extraction
        #[inline]
        pub fn $view_with_steps($me: $Me, start: (usize, usize), shape: (usize, usize), steps: (usize, usize))
            -> $MatrixView<'_, T, Dyn, Dyn, Dyn, Dyn> {
            let shape = (Dyn(shape.0), Dyn(shape.1));
            $me.$generic_view_with_steps(start, shape, steps)
        }

        /// Slices this matrix starting at its component `(irow, icol)` and with `(RVIEW, CVIEW)`
        /// consecutive components.
        #[inline]
        #[deprecated = slice_deprecation_note!($fixed_view)]
        pub fn $fixed_slice<const RVIEW: usize, const CVIEW: usize>($me: $Me, irow: usize, icol: usize)
            -> $MatrixView<'_, T, Const<RVIEW>, Const<CVIEW>, S::RStride, S::CStride> {
            $me.$fixed_view(irow, icol)
        }

        /// Returns a view of a rectangular sub-matrix with compile-time dimensions.
        ///
        /// This is similar to [`view`](Self::view), but the size of the sub-matrix is known at
        /// compile time, which can enable better compiler optimizations and type checking.
        ///
        /// # Parameters
        /// - `irow`: The zero-based row index of the top-left corner.
        /// - `icol`: The zero-based column index of the top-left corner.
        ///
        /// # Type Parameters
        /// - `RVIEW`: The number of rows in the view (known at compile time).
        /// - `CVIEW`: The number of columns in the view (known at compile time).
        ///
        /// # Panics
        /// Panics if the specified region extends beyond the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix5;
        ///
        /// let m = Matrix5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        ///     16.0, 17.0, 18.0, 19.0, 20.0,
        ///     21.0, 22.0, 23.0, 24.0, 25.0,
        /// );
        ///
        /// // Extract a 2x3 sub-matrix starting at position (1, 1)
        /// let sub = m.fixed_view::<2, 3>(1, 1);
        /// assert_eq!(sub.nrows(), 2);
        /// assert_eq!(sub.ncols(), 3);
        /// assert_eq!(sub[(0, 0)], 7.0);
        /// assert_eq!(sub[(1, 2)], 14.0);
        /// ```
        ///
        /// # See Also
        /// - [`view`](Self::view) - Extract a view with runtime dimensions
        /// - [`fixed_view_with_steps`](Self::fixed_view_with_steps) - Extract a fixed-size strided view
        /// - [`fixed_rows`](Self::fixed_rows) - Extract complete rows with compile-time size
        /// - [`fixed_columns`](Self::fixed_columns) - Extract complete columns with compile-time size
        #[inline]
        pub fn $fixed_view<const RVIEW: usize, const CVIEW: usize>($me: $Me, irow: usize, icol: usize)
            -> $MatrixView<'_, T, Const<RVIEW>, Const<CVIEW>, S::RStride, S::CStride> {

            $me.assert_view_index((irow, icol), (RVIEW, CVIEW), (0, 0));
            let shape = (Const::<RVIEW>, Const::<CVIEW>);

            unsafe {
                let data = $ViewStorage::new_unchecked($data, (irow, icol), shape);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /// Slices this matrix starting at its component `(start.0, start.1)` and with
        /// `(RVIEW, CVIEW)` components. Each row (resp. column) of the sliced
        /// matrix is separated by `steps.0` (resp. `steps.1`) ignored rows (resp. columns) of
        /// the original matrix.
        #[inline]
        #[deprecated = slice_deprecation_note!($fixed_view_with_steps)]
        pub fn $fixed_slice_with_steps<const RVIEW: usize, const CVIEW: usize>($me: $Me, start: (usize, usize), steps: (usize, usize))
            -> $MatrixView<'_, T, Const<RVIEW>, Const<CVIEW>, Dyn, Dyn> {
            $me.$fixed_view_with_steps(start, steps)
        }

        /// Returns a strided view with compile-time dimensions and runtime strides.
        ///
        /// This combines compile-time sizing with runtime-specified strides, allowing you to
        /// extract a fixed-size sub-matrix with gaps between rows and/or columns.
        ///
        /// # Parameters
        /// - `start`: A tuple `(row, col)` specifying the top-left corner (zero-based).
        /// - `steps`: A tuple `(row_step, col_step)` specifying gaps between selected elements.
        ///
        /// # Type Parameters
        /// - `RVIEW`: The number of rows in the view (known at compile time).
        /// - `CVIEW`: The number of columns in the view (known at compile time).
        ///
        /// # Panics
        /// Panics if the selected region would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix6;
        ///
        /// let m = Matrix6::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ///     19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ///     25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        ///     31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        /// );
        ///
        /// // Extract a 2x2 view starting at (0,0), skipping 1 row and 1 column
        /// let strided = m.fixed_view_with_steps::<2, 2>((0, 0), (1, 1));
        /// assert_eq!(strided[(0, 0)], 1.0);   // Row 0, Col 0
        /// assert_eq!(strided[(0, 1)], 3.0);   // Row 0, Col 2
        /// assert_eq!(strided[(1, 0)], 13.0);  // Row 2, Col 0
        /// ```
        ///
        /// # See Also
        /// - [`fixed_view`](Self::fixed_view) - Extract a contiguous fixed-size view
        /// - [`view_with_steps`](Self::view_with_steps) - Extract a strided view with runtime dimensions
        #[inline]
        pub fn $fixed_view_with_steps<const RVIEW: usize, const CVIEW: usize>($me: $Me, start: (usize, usize), steps: (usize, usize))
            -> $MatrixView<'_, T, Const<RVIEW>, Const<CVIEW>, Dyn, Dyn> {
            let shape = (Const::<RVIEW>, Const::<CVIEW>);
            $me.$generic_view_with_steps(start, shape, steps)
        }

        /// Creates a slice that may or may not have a fixed size and stride.
        #[inline]
        #[deprecated = slice_deprecation_note!($generic_view)]
        pub fn $generic_slice<RView, CView>($me: $Me, start: (usize, usize), shape: (RView, CView))
            -> $MatrixView<'_, T, RView, CView, S::RStride, S::CStride>
            where RView: Dim,
                  CView: Dim {
            $me.$generic_view(start, shape)
        }

        /// Creates a matrix view with generic (compile-time or runtime) dimensions.
        ///
        /// This is the most generic view creation method. Both the starting position and the
        /// size can be specified using either compile-time dimensions (`Const<N>`) or runtime
        /// dimensions (`Dyn`). This method is typically used internally by other more convenient
        /// methods like [`view`](Self::view) and [`fixed_view`](Self::fixed_view).
        ///
        /// # Parameters
        /// - `start`: A tuple `(row, col)` specifying the top-left corner (zero-based).
        /// - `shape`: A tuple of dimension types specifying the view size.
        ///
        /// # Type Parameters
        /// - `RView`: The row dimension type (`Const<N>` or `Dyn`).
        /// - `CView`: The column dimension type (`Const<N>` or `Dyn`).
        ///
        /// # Panics
        /// Panics if the specified region extends beyond the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix5, Dyn, Const};
        ///
        /// let m = Matrix5::identity();
        ///
        /// // Using runtime dimensions
        /// let dynamic_view = m.generic_view((1, 1), (Dyn(2), Dyn(2)));
        /// assert_eq!(dynamic_view.nrows(), 2);
        ///
        /// // Using compile-time dimensions
        /// let static_view = m.generic_view((1, 1), (Const::<2>, Const::<2>));
        /// assert_eq!(static_view.nrows(), 2);
        /// ```
        ///
        /// # See Also
        /// - [`view`](Self::view) - Simpler interface for runtime dimensions
        /// - [`fixed_view`](Self::fixed_view) - Simpler interface for compile-time dimensions
        /// - [`generic_view_with_steps`](Self::generic_view_with_steps) - Generic strided view
        #[inline]
        pub fn $generic_view<RView, CView>($me: $Me, start: (usize, usize), shape: (RView, CView))
            -> $MatrixView<'_, T, RView, CView, S::RStride, S::CStride>
            where RView: Dim,
                  CView: Dim {

            $me.assert_view_index(start, (shape.0.value(), shape.1.value()), (0, 0));

            unsafe {
                let data = $ViewStorage::new_unchecked($data, start, shape);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /// Creates a slice that may or may not have a fixed size and stride.
        #[inline]
        #[deprecated = slice_deprecation_note!($generic_view_with_steps)]
        pub fn $generic_slice_with_steps<RView, CView>($me: $Me,
                                                         start: (usize, usize),
                                                         shape: (RView, CView),
                                                         steps: (usize, usize))
            -> $MatrixView<'_, T, RView, CView, Dyn, Dyn>
            where RView: Dim,
                  CView: Dim {
            $me.$generic_view_with_steps(start, shape, steps)
        }

        /// Creates a strided matrix view with generic (compile-time or runtime) dimensions.
        ///
        /// This is the most generic strided view creation method. The size can be specified using
        /// either compile-time dimensions (`Const<N>`) or runtime dimensions (`Dyn`), while strides
        /// are always runtime parameters. This method is typically used internally by other methods
        /// like [`view_with_steps`](Self::view_with_steps) and [`fixed_view_with_steps`](Self::fixed_view_with_steps).
        ///
        /// # Parameters
        /// - `start`: A tuple `(row, col)` specifying the top-left corner (zero-based).
        /// - `shape`: A tuple of dimension types specifying the view size.
        /// - `steps`: A tuple `(row_step, col_step)` specifying gaps between selected elements.
        ///
        /// # Type Parameters
        /// - `RView`: The row dimension type (`Const<N>` or `Dyn`).
        /// - `CView`: The column dimension type (`Const<N>` or `Dyn`).
        ///
        /// # Panics
        /// Panics if the selected region would exceed the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::{Matrix6, Dyn, Const};
        ///
        /// let m = Matrix6::from_fn(|i, j| (i * 6 + j + 1) as f64);
        ///
        /// // Using runtime dimensions
        /// let dynamic_view = m.generic_view_with_steps((0, 0), (Dyn(2), Dyn(2)), (1, 1));
        /// assert_eq!(dynamic_view[(0, 0)], 1.0);
        /// assert_eq!(dynamic_view[(1, 1)], 15.0);
        ///
        /// // Using compile-time dimensions
        /// let static_view = m.generic_view_with_steps((0, 0), (Const::<2>, Const::<2>), (1, 1));
        /// assert_eq!(static_view[(0, 0)], 1.0);
        /// ```
        ///
        /// # See Also
        /// - [`generic_view`](Self::generic_view) - Generic contiguous view
        /// - [`view_with_steps`](Self::view_with_steps) - Simpler interface for runtime dimensions
        /// - [`fixed_view_with_steps`](Self::fixed_view_with_steps) - Simpler interface for compile-time dimensions
        #[inline]
        pub fn $generic_view_with_steps<RView, CView>($me: $Me,
                                                         start: (usize, usize),
                                                         shape: (RView, CView),
                                                         steps: (usize, usize))
            -> $MatrixView<'_, T, RView, CView, Dyn, Dyn>
            where RView: Dim,
                  CView: Dim {

            $me.assert_view_index(start, (shape.0.value(), shape.1.value()), steps);

            let my_strides = $me.data.strides();
            let strides    = (Dyn((steps.0 + 1) * my_strides.0.value()),
                              Dyn((steps.1 + 1) * my_strides.1.value()));

            unsafe {
                let data = $ViewStorage::new_with_strides_unchecked($data, start, shape, strides);
                Matrix::from_data_statically_unchecked(data)
            }
        }

        /*
         *
         * Splitting.
         *
         */
        /// Splits this matrix into two row views using non-overlapping row ranges.
        ///
        /// This is useful for simultaneously accessing two different sets of rows from the same
        /// matrix. The ranges must not overlap, as this would violate Rust's aliasing rules.
        /// However, the ranges don't need to be adjacent or cover all rows.
        ///
        /// # Parameters
        /// - `r1`: The first row range (can be a `usize`, `Range`, `RangeFrom`, `RangeTo`, or `RangeFull`).
        /// - `r2`: The second row range (same types as `r1`).
        ///
        /// # Returns
        /// A tuple of two matrix views `(view1, view2)` corresponding to the two row ranges.
        ///
        /// # Panics
        /// - Panics if the ranges overlap.
        /// - Panics if either range extends beyond the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix5x3;
        ///
        /// let m = Matrix5x3::new(
        ///     1.0, 2.0, 3.0,
        ///     4.0, 5.0, 6.0,
        ///     7.0, 8.0, 9.0,
        ///     10.0, 11.0, 12.0,
        ///     13.0, 14.0, 15.0,
        /// );
        ///
        /// // Split into rows 0-1 and rows 3-4
        /// let (top, bottom) = m.rows_range_pair(0..2, 3..5);
        /// assert_eq!(top.nrows(), 2);
        /// assert_eq!(bottom.nrows(), 2);
        /// assert_eq!(top[(0, 0)], 1.0);
        /// assert_eq!(bottom[(0, 0)], 10.0);
        /// ```
        ///
        /// # See Also
        /// - [`columns_range_pair`](Self::columns_range_pair) - Split by column ranges
        /// - [`rows_range`](Self::rows_range) - Extract a single row range
        #[inline]
        pub fn $rows_range_pair<Range1: DimRange<R>, Range2: DimRange<R>>($me: $Me, r1: Range1, r2: Range2)
            -> ($MatrixView<'_, T, Range1::Size, C, S::RStride, S::CStride>,
                $MatrixView<'_, T, Range2::Size, C, S::RStride, S::CStride>) {

            let (nrows, ncols) = $me.shape_generic();
            let strides        = $me.data.strides();

            let start1 = r1.begin(nrows);
            let start2 = r2.begin(nrows);

            let end1 = r1.end(nrows);
            let end2 = r2.end(nrows);

            let nrows1 = r1.size(nrows);
            let nrows2 = r2.size(nrows);

            assert!(start2 >= end1 || start1 >= end2, "Rows range pair: the ranges must not overlap.");
            assert!(end2 <= nrows.value(), "Rows range pair: index out of range.");

            unsafe {
                let ptr1 = $data.$get_addr(start1, 0);
                let ptr2 = $data.$get_addr(start2, 0);

                let data1  = $ViewStorage::from_raw_parts(ptr1, (nrows1, ncols), strides);
                let data2  = $ViewStorage::from_raw_parts(ptr2, (nrows2, ncols), strides);
                let view1 = Matrix::from_data_statically_unchecked(data1);
                let view2 = Matrix::from_data_statically_unchecked(data2);

                (view1, view2)
            }
        }

        /// Splits this matrix into two column views using non-overlapping column ranges.
        ///
        /// This is useful for simultaneously accessing two different sets of columns from the same
        /// matrix. The ranges must not overlap, as this would violate Rust's aliasing rules.
        /// However, the ranges don't need to be adjacent or cover all columns.
        ///
        /// # Parameters
        /// - `r1`: The first column range (can be a `usize`, `Range`, `RangeFrom`, `RangeTo`, or `RangeFull`).
        /// - `r2`: The second column range (same types as `r1`).
        ///
        /// # Returns
        /// A tuple of two matrix views `(view1, view2)` corresponding to the two column ranges.
        ///
        /// # Panics
        /// - Panics if the ranges overlap.
        /// - Panics if either range extends beyond the matrix bounds.
        ///
        /// # Example
        /// ```
        /// use nalgebra::Matrix3x5;
        ///
        /// let m = Matrix3x5::new(
        ///     1.0, 2.0, 3.0, 4.0, 5.0,
        ///     6.0, 7.0, 8.0, 9.0, 10.0,
        ///     11.0, 12.0, 13.0, 14.0, 15.0,
        /// );
        ///
        /// // Split into columns 0-1 and columns 3-4
        /// let (left, right) = m.columns_range_pair(0..2, 3..5);
        /// assert_eq!(left.ncols(), 2);
        /// assert_eq!(right.ncols(), 2);
        /// assert_eq!(left[(0, 0)], 1.0);
        /// assert_eq!(right[(0, 0)], 4.0);
        /// ```
        ///
        /// # See Also
        /// - [`rows_range_pair`](Self::rows_range_pair) - Split by row ranges
        /// - [`columns_range`](Self::columns_range) - Extract a single column range
        #[inline]
        pub fn $columns_range_pair<Range1: DimRange<C>, Range2: DimRange<C>>($me: $Me, r1: Range1, r2: Range2)
            -> ($MatrixView<'_, T, R, Range1::Size, S::RStride, S::CStride>,
                $MatrixView<'_, T, R, Range2::Size, S::RStride, S::CStride>) {

            let (nrows, ncols) = $me.shape_generic();
            let strides        = $me.data.strides();

            let start1 = r1.begin(ncols);
            let start2 = r2.begin(ncols);

            let end1 = r1.end(ncols);
            let end2 = r2.end(ncols);

            let ncols1 = r1.size(ncols);
            let ncols2 = r2.size(ncols);

            assert!(start2 >= end1 || start1 >= end2, "Columns range pair: the ranges must not overlap.");
            assert!(end2 <= ncols.value(), "Columns range pair: index out of range.");

            unsafe {
                let ptr1 = $data.$get_addr(0, start1);
                let ptr2 = $data.$get_addr(0, start2);

                let data1  = $ViewStorage::from_raw_parts(ptr1, (nrows, ncols1), strides);
                let data2  = $ViewStorage::from_raw_parts(ptr2, (nrows, ncols2), strides);
                let view1 = Matrix::from_data_statically_unchecked(data1);
                let view2 = Matrix::from_data_statically_unchecked(data2);

                (view1, view2)
            }
        }
    }
);

/// A matrix slice.
///
/// This type alias exists only for legacy purposes and is deprecated. It will be removed
/// in a future release. Please use [`MatrixView`] instead.
/// See [issue #1076](https://github.com/dimforge/nalgebra/issues/1076)
/// for the rationale.
#[deprecated = "Use MatrixView instead."]
pub type MatrixSlice<'a, T, R, C, RStride = U1, CStride = R> =
    MatrixView<'a, T, R, C, RStride, CStride>;

/// A matrix view.
pub type MatrixView<'a, T, R, C, RStride = U1, CStride = R> =
    Matrix<T, R, C, ViewStorage<'a, T, R, C, RStride, CStride>>;

/// A mutable matrix slice.
///
/// This type alias exists only for legacy purposes and is deprecated. It will be removed
/// in a future release. Please use [`MatrixViewMut`] instead.
/// See [issue #1076](https://github.com/dimforge/nalgebra/issues/1076)
/// for the rationale.
#[deprecated = "Use MatrixViewMut instead."]
pub type MatrixSliceMut<'a, T, R, C, RStride = U1, CStride = R> =
    MatrixViewMut<'a, T, R, C, RStride, CStride>;

/// A mutable matrix view.
pub type MatrixViewMut<'a, T, R, C, RStride = U1, CStride = R> =
    Matrix<T, R, C, ViewStorageMut<'a, T, R, C, RStride, CStride>>;

/// # Views based on index and length
impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    matrix_view_impl!(
     self: &Self, MatrixView, ViewStorage, RawStorage.get_address_unchecked(), &self.data;
     row,
     row_part,
     rows,
     rows_with_step,
     fixed_rows,
     fixed_rows_with_step,
     rows_generic,
     rows_generic_with_step,
     column,
     column_part,
     columns,
     columns_with_step,
     fixed_columns,
     fixed_columns_with_step,
     columns_generic,
     columns_generic_with_step,
     slice => view,
     slice_with_steps => view_with_steps,
     fixed_slice => fixed_view,
     fixed_slice_with_steps => fixed_view_with_steps,
     generic_slice => generic_view,
     generic_slice_with_steps => generic_view_with_steps,
     rows_range_pair,
     columns_range_pair);
}

/// # Mutable views based on index and length
impl<T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    matrix_view_impl!(
     self: &mut Self, MatrixViewMut, ViewStorageMut, RawStorageMut.get_address_unchecked_mut(), &mut self.data;
     row_mut,
     row_part_mut,
     rows_mut,
     rows_with_step_mut,
     fixed_rows_mut,
     fixed_rows_with_step_mut,
     rows_generic_mut,
     rows_generic_with_step_mut,
     column_mut,
     column_part_mut,
     columns_mut,
     columns_with_step_mut,
     fixed_columns_mut,
     fixed_columns_with_step_mut,
     columns_generic_mut,
     columns_generic_with_step_mut,
     slice_mut => view_mut,
     slice_with_steps_mut => view_with_steps_mut,
     fixed_slice_mut => fixed_view_mut,
     fixed_slice_with_steps_mut => fixed_view_with_steps_mut,
     generic_slice_mut => generic_view_mut,
     generic_slice_with_steps_mut => generic_view_with_steps_mut,
     rows_range_pair_mut,
     columns_range_pair_mut);
}

/// A range with a size that may be known at compile-time.
///
/// This may be:
/// * A single `usize` index, e.g., `4`
/// * A left-open range `std::ops::RangeTo`, e.g., `.. 4`
/// * A right-open range `std::ops::RangeFrom`, e.g., `4 ..`
/// * A full range `std::ops::RangeFull`, e.g., `..`
pub trait DimRange<D: Dim> {
    /// Type of the range size. May be a type-level integer.
    type Size: Dim;

    /// The start index of the range.
    fn begin(&self, shape: D) -> usize;
    // NOTE: this is the index immediately after the last index.
    /// The index immediately after the last index inside of the range.
    fn end(&self, shape: D) -> usize;
    /// The number of elements of the range, i.e., `self.end - self.begin`.
    fn size(&self, shape: D) -> Self::Size;
}

/// A range with a size that may be known at compile-time.
///
/// This is merely a legacy trait alias to minimize breakage. Use the [`DimRange`] trait instead.
#[deprecated = slice_deprecation_note!(DimRange)]
pub trait SliceRange<D: Dim>: DimRange<D> {}

#[allow(deprecated)]
impl<R: DimRange<D>, D: Dim> SliceRange<D> for R {}

impl<D: Dim> DimRange<D> for usize {
    type Size = U1;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        *self
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        *self + 1
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Const::<1>
    }
}

impl<D: Dim> DimRange<D> for Range<usize> {
    type Size = Dyn;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        self.end
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dyn(self.end - self.start)
    }
}

impl<D: Dim> DimRange<D> for RangeFrom<usize> {
    type Size = Dyn;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn end(&self, dim: D) -> usize {
        dim.value()
    }

    #[inline(always)]
    fn size(&self, dim: D) -> Self::Size {
        Dyn(dim.value() - self.start)
    }
}

impl<D: Dim> DimRange<D> for RangeTo<usize> {
    type Size = Dyn;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        self.end
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dyn(self.end)
    }
}

impl<D: Dim> DimRange<D> for RangeFull {
    type Size = D;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn end(&self, dim: D) -> usize {
        dim.value()
    }

    #[inline(always)]
    fn size(&self, dim: D) -> Self::Size {
        dim
    }
}

impl<D: Dim> DimRange<D> for RangeInclusive<usize> {
    type Size = Dyn;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        *self.start()
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        *self.end() + 1
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dyn(*self.end() + 1 - *self.start())
    }
}

// TODO: see how much of this overlaps with the general indexing
// methods from indexing.rs.
impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Slices a sub-matrix containing the rows indexed by the range `rows` and the columns indexed
    /// by the range `cols`.
    #[inline]
    #[must_use]
    #[deprecated = slice_deprecation_note!(view_range)]
    pub fn slice_range<RowRange, ColRange>(
        &self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixView<'_, T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: DimRange<R>,
        ColRange: DimRange<C>,
    {
        let (nrows, ncols) = self.shape_generic();
        self.generic_view(
            (rows.begin(nrows), cols.begin(ncols)),
            (rows.size(nrows), cols.size(ncols)),
        )
    }

    /// Returns a view containing the rows and columns indexed by the given ranges.
    ///
    /// This provides a convenient way to extract a rectangular sub-matrix using Rust's
    /// range syntax. You can use various range types including `start..end`, `start..`,
    /// `..end`, `..`, or even single indices.
    ///
    /// # Parameters
    /// - `rows`: A range specifying which rows to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    /// - `cols`: A range specifying which columns to include (same syntax as rows).
    ///
    /// # Returns
    /// A matrix view referencing the specified sub-matrix.
    ///
    /// # Panics
    /// Panics if either range extends beyond the matrix bounds.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix5;
    ///
    /// let m = Matrix5::new(
    ///     1.0, 2.0, 3.0, 4.0, 5.0,
    ///     6.0, 7.0, 8.0, 9.0, 10.0,
    ///     11.0, 12.0, 13.0, 14.0, 15.0,
    ///     16.0, 17.0, 18.0, 19.0, 20.0,
    ///     21.0, 22.0, 23.0, 24.0, 25.0,
    /// );
    ///
    /// // Extract rows 1-2 (inclusive of 1, exclusive of 3) and columns 2-4
    /// let sub = m.view_range(1..3, 2..5);
    /// assert_eq!(sub.nrows(), 2);
    /// assert_eq!(sub.ncols(), 3);
    /// assert_eq!(sub[(0, 0)], 8.0);
    ///
    /// // Use open-ended ranges
    /// let bottom_right = m.view_range(3.., 3..);
    /// assert_eq!(bottom_right.nrows(), 2);
    /// assert_eq!(bottom_right[(0, 0)], 19.0);
    /// ```
    ///
    /// # See Also
    /// - [`view`](Self::view) - Extract a view using explicit start position and size
    /// - [`rows_range`](Self::rows_range) - Extract only specific rows (all columns)
    /// - [`columns_range`](Self::columns_range) - Extract only specific columns (all rows)
    #[inline]
    #[must_use]
    pub fn view_range<RowRange, ColRange>(
        &self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixView<'_, T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: DimRange<R>,
        ColRange: DimRange<C>,
    {
        let (nrows, ncols) = self.shape_generic();
        self.generic_view(
            (rows.begin(nrows), cols.begin(ncols)),
            (rows.size(nrows), cols.size(ncols)),
        )
    }

    /// Returns a view containing all rows indexed by the given range.
    ///
    /// This extracts a horizontal band from the matrix, including all columns but only
    /// the rows specified by the range. It's a convenient shorthand for
    /// `view_range(rows, ..)`.
    ///
    /// # Parameters
    /// - `rows`: A range specifying which rows to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    ///
    /// # Returns
    /// A matrix view containing the specified rows and all columns.
    ///
    /// # Panics
    /// Panics if the range extends beyond the number of rows in the matrix.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix5x3;
    ///
    /// let m = Matrix5x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    ///     10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0,
    /// );
    ///
    /// // Extract rows 1-3
    /// let middle = m.rows_range(1..4);
    /// assert_eq!(middle.nrows(), 3);
    /// assert_eq!(middle.ncols(), 3);
    /// assert_eq!(middle[(0, 0)], 4.0);
    /// assert_eq!(middle[(2, 2)], 12.0);
    ///
    /// // Extract from row 3 to the end
    /// let bottom = m.rows_range(3..);
    /// assert_eq!(bottom.nrows(), 2);
    /// ```
    ///
    /// # See Also
    /// - [`rows`](Self::rows) - Extract rows using explicit start and count
    /// - [`columns_range`](Self::columns_range) - Extract columns by range
    /// - [`view_range`](Self::view_range) - Extract both rows and columns by range
    #[inline]
    #[must_use]
    pub fn rows_range<RowRange: DimRange<R>>(
        &self,
        rows: RowRange,
    ) -> MatrixView<'_, T, RowRange::Size, C, S::RStride, S::CStride> {
        self.view_range(rows, ..)
    }

    /// Returns a view containing all columns indexed by the given range.
    ///
    /// This extracts a vertical band from the matrix, including all rows but only
    /// the columns specified by the range. It's a convenient shorthand for
    /// `view_range(.., cols)`.
    ///
    /// # Parameters
    /// - `cols`: A range specifying which columns to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    ///
    /// # Returns
    /// A matrix view containing all rows and the specified columns.
    ///
    /// # Panics
    /// Panics if the range extends beyond the number of columns in the matrix.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix3x5;
    ///
    /// let m = Matrix3x5::new(
    ///     1.0, 2.0, 3.0, 4.0, 5.0,
    ///     6.0, 7.0, 8.0, 9.0, 10.0,
    ///     11.0, 12.0, 13.0, 14.0, 15.0,
    /// );
    ///
    /// // Extract columns 1-3
    /// let middle = m.columns_range(1..4);
    /// assert_eq!(middle.nrows(), 3);
    /// assert_eq!(middle.ncols(), 3);
    /// assert_eq!(middle[(0, 0)], 2.0);
    /// assert_eq!(middle[(2, 2)], 14.0);
    ///
    /// // Extract the first 2 columns
    /// let left = m.columns_range(..2);
    /// assert_eq!(left.ncols(), 2);
    /// ```
    ///
    /// # See Also
    /// - [`columns`](Self::columns) - Extract columns using explicit start and count
    /// - [`rows_range`](Self::rows_range) - Extract rows by range
    /// - [`view_range`](Self::view_range) - Extract both rows and columns by range
    #[inline]
    #[must_use]
    pub fn columns_range<ColRange: DimRange<C>>(
        &self,
        cols: ColRange,
    ) -> MatrixView<'_, T, R, ColRange::Size, S::RStride, S::CStride> {
        self.view_range(.., cols)
    }
}

// TODO: see how much of this overlaps with the general indexing
// methods from indexing.rs.
impl<T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Slices a mutable sub-matrix containing the rows indexed by the range `rows` and the columns
    /// indexed by the range `cols`.
    #[deprecated = slice_deprecation_note!(view_range_mut)]
    pub fn slice_range_mut<RowRange, ColRange>(
        &mut self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixViewMut<'_, T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: DimRange<R>,
        ColRange: DimRange<C>,
    {
        self.view_range_mut(rows, cols)
    }

    /// Returns a mutable view containing the rows and columns indexed by the given ranges.
    ///
    /// This is the mutable version of [`view_range`](Self::view_range). It provides a way to
    /// extract and modify a rectangular sub-matrix using Rust's range syntax.
    ///
    /// # Parameters
    /// - `rows`: A range specifying which rows to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    /// - `cols`: A range specifying which columns to include (same syntax as rows).
    ///
    /// # Returns
    /// A mutable matrix view referencing the specified sub-matrix.
    ///
    /// # Panics
    /// Panics if either range extends beyond the matrix bounds.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix5;
    ///
    /// let mut m = Matrix5::new(
    ///     1.0, 2.0, 3.0, 4.0, 5.0,
    ///     6.0, 7.0, 8.0, 9.0, 10.0,
    ///     11.0, 12.0, 13.0, 14.0, 15.0,
    ///     16.0, 17.0, 18.0, 19.0, 20.0,
    ///     21.0, 22.0, 23.0, 24.0, 25.0,
    /// );
    ///
    /// // Extract and modify a 2x3 sub-matrix
    /// let mut sub = m.view_range_mut(1..3, 2..5);
    /// sub[(0, 0)] = 100.0;
    ///
    /// // The change is reflected in the original matrix
    /// assert_eq!(m[(1, 2)], 100.0);
    /// ```
    ///
    /// # See Also
    /// - [`view_range`](Self::view_range) - Immutable version
    /// - [`view_mut`](Self::view_mut) - Extract a mutable view using explicit coordinates
    /// - [`rows_range_mut`](Self::rows_range_mut) - Extract mutable rows by range
    /// - [`columns_range_mut`](Self::columns_range_mut) - Extract mutable columns by range
    pub fn view_range_mut<RowRange, ColRange>(
        &mut self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixViewMut<'_, T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: DimRange<R>,
        ColRange: DimRange<C>,
    {
        let (nrows, ncols) = self.shape_generic();
        self.generic_view_mut(
            (rows.begin(nrows), cols.begin(ncols)),
            (rows.size(nrows), cols.size(ncols)),
        )
    }

    /// Returns a mutable view containing all rows indexed by the given range.
    ///
    /// This is the mutable version of [`rows_range`](Self::rows_range). It extracts a
    /// horizontal band from the matrix that can be modified.
    ///
    /// # Parameters
    /// - `rows`: A range specifying which rows to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    ///
    /// # Returns
    /// A mutable matrix view containing the specified rows and all columns.
    ///
    /// # Panics
    /// Panics if the range extends beyond the number of rows in the matrix.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix5x3;
    ///
    /// let mut m = Matrix5x3::zeros();
    ///
    /// // Modify rows 1-3
    /// let mut middle = m.rows_range_mut(1..4);
    /// middle.fill(5.0);
    ///
    /// // Verify the changes
    /// assert_eq!(m[(0, 0)], 0.0);  // Row 0 unchanged
    /// assert_eq!(m[(1, 0)], 5.0);  // Row 1 modified
    /// assert_eq!(m[(3, 0)], 5.0);  // Row 3 modified
    /// assert_eq!(m[(4, 0)], 0.0);  // Row 4 unchanged
    /// ```
    ///
    /// # See Also
    /// - [`rows_range`](Self::rows_range) - Immutable version
    /// - [`rows_mut`](Self::rows_mut) - Extract mutable rows using explicit start and count
    /// - [`columns_range_mut`](Self::columns_range_mut) - Extract mutable columns by range
    #[inline]
    pub fn rows_range_mut<RowRange: DimRange<R>>(
        &mut self,
        rows: RowRange,
    ) -> MatrixViewMut<'_, T, RowRange::Size, C, S::RStride, S::CStride> {
        self.view_range_mut(rows, ..)
    }

    /// Returns a mutable view containing all columns indexed by the given range.
    ///
    /// This is the mutable version of [`columns_range`](Self::columns_range). It extracts a
    /// vertical band from the matrix that can be modified.
    ///
    /// # Parameters
    /// - `cols`: A range specifying which columns to include (e.g., `1..4`, `2..`, `..3`, or `..`).
    ///
    /// # Returns
    /// A mutable matrix view containing all rows and the specified columns.
    ///
    /// # Panics
    /// Panics if the range extends beyond the number of columns in the matrix.
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix3x5;
    ///
    /// let mut m = Matrix3x5::zeros();
    ///
    /// // Modify columns 2-4
    /// let mut middle = m.columns_range_mut(2..5);
    /// middle.fill(7.0);
    ///
    /// // Verify the changes
    /// assert_eq!(m[(0, 1)], 0.0);  // Column 1 unchanged
    /// assert_eq!(m[(0, 2)], 7.0);  // Column 2 modified
    /// assert_eq!(m[(2, 4)], 7.0);  // Column 4 modified
    /// ```
    ///
    /// # See Also
    /// - [`columns_range`](Self::columns_range) - Immutable version
    /// - [`columns_mut`](Self::columns_mut) - Extract mutable columns using explicit start and count
    /// - [`rows_range_mut`](Self::rows_range_mut) - Extract mutable rows by range
    #[inline]
    pub fn columns_range_mut<ColRange: DimRange<C>>(
        &mut self,
        cols: ColRange,
    ) -> MatrixViewMut<'_, T, R, ColRange::Size, S::RStride, S::CStride> {
        self.view_range_mut(.., cols)
    }
}

impl<'a, T, R, C, RStride, CStride> From<MatrixViewMut<'a, T, R, C, RStride, CStride>>
    for MatrixView<'a, T, R, C, RStride, CStride>
where
    R: Dim,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
{
    fn from(view_mut: MatrixViewMut<'a, T, R, C, RStride, CStride>) -> Self {
        let data = ViewStorage {
            ptr: view_mut.data.ptr,
            shape: view_mut.data.shape,
            strides: view_mut.data.strides,
            _phantoms: PhantomData,
        };

        unsafe { Matrix::from_data_statically_unchecked(data) }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Returns this matrix as a view.
    ///
    /// The returned view type is generally ambiguous unless specified.
    /// This is particularly useful when working with functions or methods that take
    /// matrix views as input.
    ///
    /// # Panics
    /// Panics if the dimensions of the view and the matrix are not compatible and this cannot
    /// be proven at compile-time. This might happen, for example, when constructing a static
    /// view of size 3x3 from a dynamically sized matrix of dimension 5x5.
    ///
    /// # Examples
    /// ```
    /// use nalgebra::{DMatrixSlice, SMatrixView};
    ///
    /// fn consume_view(_: DMatrixSlice<f64>) {}
    ///
    /// let matrix = nalgebra::Matrix3::zeros();
    /// consume_view(matrix.as_view());
    ///
    /// let dynamic_view: DMatrixSlice<f64> = matrix.as_view();
    /// let static_view_from_dyn: SMatrixView<f64, 3, 3> = dynamic_view.as_view();
    /// ```
    pub fn as_view<RView, CView, RViewStride, CViewStride>(
        &self,
    ) -> MatrixView<'_, T, RView, CView, RViewStride, CViewStride>
    where
        RView: Dim,
        CView: Dim,
        RViewStride: Dim,
        CViewStride: Dim,
        ShapeConstraint: DimEq<R, RView>
            + DimEq<C, CView>
            + DimEq<RViewStride, S::RStride>
            + DimEq<CViewStride, S::CStride>,
    {
        // Defer to (&matrix).into()
        self.into()
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    /// Returns this matrix as a mutable view.
    ///
    /// The returned view type is generally ambiguous unless specified.
    /// This is particularly useful when working with functions or methods that take
    /// matrix views as input.
    ///
    /// # Panics
    /// Panics if the dimensions of the view and the matrix are not compatible and this cannot
    /// be proven at compile-time. This might happen, for example, when constructing a static
    /// view of size 3x3 from a dynamically sized matrix of dimension 5x5.
    ///
    /// # Examples
    /// ```
    /// use nalgebra::{DMatrixViewMut, SMatrixViewMut};
    ///
    /// fn consume_view(_: DMatrixViewMut<f64>) {}
    ///
    /// let mut matrix = nalgebra::Matrix3::zeros();
    /// consume_view(matrix.as_view_mut());
    ///
    /// let mut dynamic_view: DMatrixViewMut<f64> = matrix.as_view_mut();
    /// let static_view_from_dyn: SMatrixViewMut<f64, 3, 3> = dynamic_view.as_view_mut();
    /// ```
    pub fn as_view_mut<RView, CView, RViewStride, CViewStride>(
        &mut self,
    ) -> MatrixViewMut<'_, T, RView, CView, RViewStride, CViewStride>
    where
        RView: Dim,
        CView: Dim,
        RViewStride: Dim,
        CViewStride: Dim,
        ShapeConstraint: DimEq<R, RView>
            + DimEq<C, CView>
            + DimEq<RViewStride, S::RStride>
            + DimEq<CViewStride, S::CStride>,
    {
        // Defer to (&mut matrix).into()
        self.into()
    }
}

// TODO: Arbitrary strides?
impl<'a, T, R1, C1, R2, C2> ReshapableStorage<T, R1, C1, R2, C2>
    for ViewStorage<'a, T, R1, C1, U1, R1>
where
    T: Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
{
    type Output = ViewStorage<'a, T, R2, C2, U1, R2>;

    fn reshape_generic(self, nrows: R2, ncols: C2) -> Self::Output {
        let (r1, c1) = self.shape();
        assert_eq!(nrows.value() * ncols.value(), r1.value() * c1.value());
        let ptr = self.ptr();
        let new_shape = (nrows, ncols);
        let strides = (U1::name(), nrows);
        unsafe { ViewStorage::from_raw_parts(ptr, new_shape, strides) }
    }
}

// TODO: Arbitrary strides?
impl<'a, T, R1, C1, R2, C2> ReshapableStorage<T, R1, C1, R2, C2>
    for ViewStorageMut<'a, T, R1, C1, U1, R1>
where
    T: Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
{
    type Output = ViewStorageMut<'a, T, R2, C2, U1, R2>;

    fn reshape_generic(mut self, nrows: R2, ncols: C2) -> Self::Output {
        let (r1, c1) = self.shape();
        assert_eq!(nrows.value() * ncols.value(), r1.value() * c1.value());
        let ptr = self.ptr_mut();
        let new_shape = (nrows, ncols);
        let strides = (U1::name(), nrows);
        unsafe { ViewStorageMut::from_raw_parts(ptr, new_shape, strides) }
    }
}
