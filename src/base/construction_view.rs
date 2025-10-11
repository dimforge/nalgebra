use crate::base::dimension::{Const, Dim, DimName, Dyn};
use crate::base::matrix_view::{ViewStorage, ViewStorageMut};
use crate::base::{MatrixView, MatrixViewMut, Scalar};

use num_rational::Ratio;

/// # Creating matrix views from `&[T]`
impl<'a, T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixView<'a, T, R, C, RStride, CStride>
{
    /// Creates, without bounds checking, a matrix view from a slice with custom dimensions and strides.
    ///
    /// This is a low-level function for creating matrix views with full control over memory layout.
    /// It allows you to specify both the dimensions (rows and columns) and strides (memory spacing
    /// between elements) using generic type parameters.
    ///
    /// # What are strides?
    ///
    /// Strides control how elements are spaced in memory:
    /// - `rstride` (row stride): Number of elements to skip between consecutive rows
    /// - `cstride` (column stride): Number of elements to skip between consecutive columns
    ///
    /// For a standard column-major matrix, `rstride` is typically 1 (elements in a column are
    /// contiguous) and `cstride` equals the number of rows (columns are separated by that many elements).
    ///
    /// # Parameters
    ///
    /// - `data`: The slice containing the matrix data
    /// - `start`: Index in the slice where the matrix data begins
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    /// - `rstride`: Row stride (type `RStride` - can be `Const<N>` or `Dyn`)
    /// - `cstride`: Column stride (type `CStride` - can be `Const<N>` or `Dyn`)
    ///
    /// # Safety
    ///
    /// This method is unsafe because:
    /// - The input data array is not checked to contain enough elements
    /// - Accessing elements using the provided strides must not go out of bounds
    /// - The caller must ensure: `data.len() >= start + (nrows-1)*rstride + (ncols-1)*cstride + 1`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixView, Dyn, Const};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// // Create a 2x2 view starting at index 1 with custom strides
    /// // rstride=1, cstride=2 means: column-major with columns 2 elements apart
    /// let view = unsafe {
    ///     MatrixView::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic_unchecked(
    ///         &data,
    ///         1,           // start at index 1 (element = 2.0)
    ///         Dyn(2),      // 2 rows
    ///         Dyn(2),      // 2 columns
    ///         Dyn(1),      // row stride = 1
    ///         Dyn(2),      // column stride = 2
    ///     )
    /// };
    ///
    /// // This creates the matrix:
    /// // [2.0, 4.0]
    /// // [3.0, 5.0]
    /// assert_eq!(view[(0, 0)], 2.0);
    /// assert_eq!(view[(0, 1)], 4.0);
    /// assert_eq!(view[(1, 0)], 3.0);
    /// assert_eq!(view[(1, 1)], 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Safe version with bounds checking
    /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Simpler version without custom strides
    /// - [`from_slice_with_strides`](Self::from_slice_with_strides) - Runtime-sized version
    #[inline]
    pub const unsafe fn from_slice_with_strides_generic_unchecked(
        data: &'a [T],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        unsafe {
            let data = ViewStorage::from_raw_parts(
                data.as_ptr().add(start),
                (nrows, ncols),
                (rstride, cstride),
            );
            Self::from_data(data)
        }
    }

    /// Creates a matrix view from a slice with custom dimensions and strides.
    ///
    /// This function creates a view into a slice with full control over the matrix dimensions
    /// and memory layout through stride parameters. It performs bounds checking to ensure
    /// the slice is large enough for the requested view.
    ///
    /// # What are strides?
    ///
    /// Strides control how elements are spaced in memory:
    /// - `rstride` (row stride): Number of elements to skip between consecutive rows
    /// - `cstride` (column stride): Number of elements to skip between consecutive columns
    ///
    /// For a standard column-major matrix, `rstride` is 1 (elements in a column are
    /// contiguous) and `cstride` equals the number of rows.
    ///
    /// # Parameters
    ///
    /// - `data`: The slice containing the matrix data
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    /// - `rstride`: Row stride (type `RStride` - can be `Const<N>` or `Dyn`)
    /// - `cstride`: Column stride (type `CStride` - can be `Const<N>` or `Dyn`)
    ///
    /// # Panics
    ///
    /// Panics if the input data slice does not contain enough elements for the specified
    /// dimensions and strides. Specifically, requires:
    /// `data.len() >= (ncols-1)*cstride + (nrows-1)*rstride + 1`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixView, Dyn, Const};
    ///
    /// // Create a view of a 2x3 matrix from a slice
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// // Standard column-major layout: rstride=1, cstride=nrows
    /// let view = MatrixView::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic(
    ///     &data,
    ///     Dyn(2),      // 2 rows
    ///     Dyn(3),      // 3 columns
    ///     Dyn(1),      // row stride = 1
    ///     Dyn(2),      // column stride = 2
    /// );
    ///
    /// assert_eq!(view.nrows(), 2);
    /// assert_eq!(view.ncols(), 3);
    /// assert_eq!(view[(0, 0)], 1.0);
    /// assert_eq!(view[(1, 2)], 6.0);
    /// ```
    ///
    /// Custom strides for non-contiguous data:
    ///
    /// ```
    /// use nalgebra::{MatrixView, Dyn};
    ///
    /// // Extract every other element
    /// let data = vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0];
    ///
    /// let view = MatrixView::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic(
    ///     &data,
    ///     Dyn(2),      // 2 rows
    ///     Dyn(2),      // 2 columns
    ///     Dyn(2),      // row stride = 2 (skip padding)
    ///     Dyn(4),      // column stride = 4
    /// );
    ///
    /// // View contains: [1.0, 3.0]
    /// //                [2.0, 4.0]
    /// assert_eq!(view[(0, 0)], 1.0);
    /// assert_eq!(view[(1, 0)], 2.0);
    /// assert_eq!(view[(0, 1)], 3.0);
    /// assert_eq!(view[(1, 1)], 4.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Unchecked version
    /// - [`from_slice_generic`](Self::from_slice_generic) - Simpler version with default strides
    /// - [`from_slice_with_strides`](Self::from_slice_with_strides) - Runtime-sized version with strides
    #[inline]
    pub fn from_slice_with_strides_generic(
        data: &'a [T],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix view: input data buffer too small."
        );

        unsafe {
            Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim> MatrixView<'a, T, R, C> {
    /// Creates, without bounds checking, a matrix view from a slice with specified dimensions.
    ///
    /// This is a simplified version of `from_slice_with_strides_generic_unchecked` that uses
    /// standard column-major strides (rstride=1, cstride=nrows). This is the most common
    /// layout for nalgebra matrices where elements in each column are stored contiguously.
    ///
    /// # Parameters
    ///
    /// - `data`: The slice containing the matrix data in column-major order
    /// - `start`: Index in the slice where the matrix data begins
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    ///
    /// # Safety
    ///
    /// This method is unsafe because:
    /// - The input data array is not checked to contain enough elements
    /// - The caller must ensure: `data.len() >= start + nrows * ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixView, Dyn};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    ///
    /// // Create a 2x3 view starting at index 2
    /// let view = unsafe {
    ///     MatrixView::<_, Dyn, Dyn>::from_slice_generic_unchecked(
    ///         &data,
    ///         2,       // start at index 2 (element = 3.0)
    ///         Dyn(2),  // 2 rows
    ///         Dyn(3),  // 3 columns
    ///     )
    /// };
    ///
    /// // Column-major layout: [3.0, 5.0, 7.0]
    /// //                      [4.0, 6.0, 8.0]
    /// assert_eq!(view[(0, 0)], 3.0);
    /// assert_eq!(view[(1, 0)], 4.0);
    /// assert_eq!(view[(0, 1)], 5.0);
    /// assert_eq!(view[(1, 1)], 6.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_generic`](Self::from_slice_generic) - Safe version with bounds checking
    /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Version with custom strides
    /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Runtime-sized version
    #[inline]
    pub unsafe fn from_slice_generic_unchecked(
        data: &'a [T],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        unsafe {
            Self::from_slice_with_strides_generic_unchecked(
                data, start, nrows, ncols, Const::<1>, nrows,
            )
        }
    }

    /// Creates a matrix view from a slice with specified dimensions.
    ///
    /// This is the primary safe function for creating matrix views from slices using generic
    /// dimension parameters. It assumes standard column-major layout where elements in each
    /// column are stored contiguously in memory.
    ///
    /// # What is column-major layout?
    ///
    /// In column-major layout (used by nalgebra), matrix elements are stored column by column:
    /// - A 2x3 matrix `[a, b, c; d, e, f]` is stored as `[a, d, b, e, c, f]`
    /// - Elements in the same column are adjacent in memory
    ///
    /// # Parameters
    ///
    /// - `data`: The slice containing matrix data in column-major order
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    ///
    /// # Panics
    ///
    /// Panics if the input data slice does not contain enough elements.
    /// Requires: `data.len() >= nrows * ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixView, Dyn, Const};
    ///
    /// // Create a 2x3 matrix view from a slice
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// let view = MatrixView::<_, Dyn, Dyn>::from_slice_generic(
    ///     &data,
    ///     Dyn(2),  // 2 rows
    ///     Dyn(3),  // 3 columns
    /// );
    ///
    /// // Column-major layout creates:
    /// // [1.0, 3.0, 5.0]
    /// // [2.0, 4.0, 6.0]
    /// assert_eq!(view[(0, 0)], 1.0);
    /// assert_eq!(view[(1, 0)], 2.0);
    /// assert_eq!(view[(0, 1)], 3.0);
    /// assert_eq!(view[(0, 2)], 5.0);
    /// ```
    ///
    /// Using compile-time dimensions with `Const`:
    ///
    /// ```
    /// use nalgebra::{MatrixView, Const};
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    ///
    /// let view = MatrixView::<_, Const<2>, Const<2>>::from_slice_generic(
    ///     &data,
    ///     Const::<2>,  // 2 rows (compile-time constant)
    ///     Const::<2>,  // 2 columns (compile-time constant)
    /// );
    ///
    /// assert_eq!(view.nrows(), 2);
    /// assert_eq!(view.ncols(), 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Unchecked version
    /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Version with custom strides
    /// - [`from_slice`](Self::from_slice) - Simpler runtime-sized version
    #[inline]
    pub fn from_slice_generic(data: &'a [T], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, Const::<1>, nrows)
    }
}

macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr_2021),*; $($args: ident),*) => {
        impl<'a, T: Scalar, $($DimIdent: $DimBound),*> MatrixView<'a, T, $($Dims),*> {
            /// Creates a matrix view from a slice.
            ///
            /// This is the most convenient method for creating matrix views from slices with
            /// runtime dimensions. The slice should contain the matrix data in column-major order
            /// (elements within each column are contiguous).
            ///
            /// # What is column-major layout?
            ///
            /// In column-major layout (used by nalgebra), matrix elements are stored column by column:
            /// - A 2x3 matrix `[a, b, c; d, e, f]` is stored as `[a, d, b, e, c, f]`
            /// - Elements in the same column are adjacent in memory
            ///
            /// # Panics
            ///
            /// Panics if `data` does not contain enough elements for the requested dimensions.
            /// Requires: `data.len() >= nrows * ncols`
            ///
            /// # Examples
            ///
            /// Creating a dynamic matrix view:
            ///
            /// ```
            /// use nalgebra::DMatrixView;
            ///
            /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a 2x3 view
            /// let view = DMatrixView::from_slice(&data, 2, 3);
            ///
            /// // Column-major: [1.0, 3.0, 5.0]
            /// //               [2.0, 4.0, 6.0]
            /// assert_eq!(view[(0, 0)], 1.0);
            /// assert_eq!(view[(1, 0)], 2.0);
            /// assert_eq!(view[(0, 1)], 3.0);
            /// assert_eq!(view.nrows(), 2);
            /// assert_eq!(view.ncols(), 3);
            /// ```
            ///
            /// Creating a fixed-size matrix view:
            ///
            /// ```
            /// use nalgebra::MatrixView2x3;
            ///
            /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // No dimensions needed - they're in the type!
            /// let view = MatrixView2x3::from_slice(&data);
            ///
            /// assert_eq!(view.nrows(), 2);
            /// assert_eq!(view.ncols(), 3);
            /// assert_eq!(view[(1, 2)], 6.0);
            /// ```
            ///
            /// Creating a partially dynamic view:
            ///
            /// ```
            /// use nalgebra::MatrixView2xX;
            ///
            /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // 2 rows (fixed), dynamic columns
            /// let view = MatrixView2xX::from_slice(&data, 3);
            ///
            /// assert_eq!(view.nrows(), 2);
            /// assert_eq!(view.ncols(), 3);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Unchecked version
            /// - [`from_slice_with_strides`](Self::from_slice_with_strides) - Version with custom strides
            /// - [`from_slice_generic`](Self::from_slice_generic) - Generic version with type-level dimensions
            #[inline]
            pub fn from_slice(data: &'a [T], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bounds checking, a matrix view from a slice.
            ///
            /// This is the unchecked version of [`from_slice`](Self::from_slice). It skips bounds
            /// checking and allows specifying a starting offset in the slice.
            ///
            /// # Parameters
            ///
            /// - `data`: The slice containing the matrix data in column-major order
            /// - `start`: Index in the slice where the matrix data begins
            /// - Additional parameters depend on the matrix type (dimensions for dynamic matrices)
            ///
            /// # Safety
            ///
            /// The caller must ensure:
            /// - `data.len() >= start + nrows * ncols`
            /// - The start index and dimensions don't cause out-of-bounds access
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::DMatrixView;
            ///
            /// let data = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a 2x3 view starting at index 1 (skipping the first element)
            /// let view = unsafe {
            ///     DMatrixView::from_slice_unchecked(&data, 1, 2, 3)
            /// };
            ///
            /// // View contains elements 1.0 through 6.0
            /// assert_eq!(view[(0, 0)], 1.0);
            /// assert_eq!(view[(1, 2)], 6.0);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice`](Self::from_slice) - Safe version with bounds checking
            /// - [`from_slice_with_strides_unchecked`](Self::from_slice_with_strides_unchecked) - Version with custom strides
            /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Generic version
            #[inline]
            pub unsafe fn from_slice_unchecked(data: &'a [T], start: usize, $($args: usize),*) -> Self { unsafe {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }}
        }

        impl<'a, T: Scalar, $($DimIdent: $DimBound, )*> MatrixView<'a, T, $($Dims,)* Dyn, Dyn> {
            /// Creates a matrix view with custom strides from a slice.
            ///
            /// This function allows you to create a matrix view with non-standard memory layout
            /// by specifying custom row and column strides. This is useful for viewing
            /// non-contiguous data or data with padding.
            ///
            /// # What are strides?
            ///
            /// Strides control how elements are spaced in memory:
            /// - `rstride` (row stride): Number of elements to skip between consecutive rows
            /// - `cstride` (column stride): Number of elements to skip between consecutive columns
            ///
            /// For a standard column-major matrix, `rstride=1` and `cstride=nrows`.
            ///
            /// # Panics
            ///
            /// Panics if `data` does not contain enough elements for the specified dimensions
            /// and strides.
            ///
            /// # Examples
            ///
            /// Standard usage with custom strides:
            ///
            /// ```
            /// use nalgebra::DMatrixView;
            ///
            /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a 2x2 view with column stride = 3
            /// let view = DMatrixView::from_slice_with_strides(&data, 2, 2, 1, 3);
            ///
            /// // This creates: [1.0, 4.0]
            /// //               [2.0, 5.0]
            /// assert_eq!(view[(0, 0)], 1.0);
            /// assert_eq!(view[(1, 0)], 2.0);
            /// assert_eq!(view[(0, 1)], 4.0);
            /// assert_eq!(view[(1, 1)], 5.0);
            /// ```
            ///
            /// Extracting every other row:
            ///
            /// ```
            /// use nalgebra::DMatrixView;
            ///
            /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            ///
            /// // Extract rows 0 and 2 by using row stride = 2
            /// let view = DMatrixView::from_slice_with_strides(&data, 2, 2, 2, 4);
            ///
            /// // This creates: [1.0, 5.0]
            /// //               [3.0, 7.0]
            /// assert_eq!(view[(0, 0)], 1.0);
            /// assert_eq!(view[(1, 0)], 3.0);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice`](Self::from_slice) - Simpler version with default strides
            /// - [`from_slice_with_strides_unchecked`](Self::from_slice_with_strides_unchecked) - Unchecked version
            /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Generic version
            #[inline]
            pub fn from_slice_with_strides(data: &'a [T], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(data, $($gargs,)* Dyn(rstride), Dyn(cstride))
            }

            /// Creates, without bounds checking, a matrix view with custom strides from a slice.
            ///
            /// This is the unchecked version of [`from_slice_with_strides`](Self::from_slice_with_strides).
            /// It skips bounds checking and allows specifying a starting offset in the slice.
            ///
            /// # Parameters
            ///
            /// - `data`: The slice containing the matrix data
            /// - `start`: Index in the slice where the matrix data begins
            /// - Additional dimension parameters (nrows, ncols for fully dynamic matrices)
            /// - `rstride`: Row stride (elements to skip between consecutive rows)
            /// - `cstride`: Column stride (elements to skip between consecutive columns)
            ///
            /// # Safety
            ///
            /// The caller must ensure:
            /// - `data.len() >= start + (nrows-1)*rstride + (ncols-1)*cstride + 1`
            /// - The strides don't cause out-of-bounds access when accessing matrix elements
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::DMatrixView;
            ///
            /// let data = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a 2x2 view starting at index 1 with custom strides
            /// let view = unsafe {
            ///     DMatrixView::from_slice_with_strides_unchecked(
            ///         &data,
            ///         1,    // start
            ///         2, 2, // dimensions
            ///         1, 3, // strides
            ///     )
            /// };
            ///
            /// // Creates: [1.0, 4.0]
            /// //          [2.0, 5.0]
            /// assert_eq!(view[(0, 0)], 1.0);
            /// assert_eq!(view[(0, 1)], 4.0);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice_with_strides`](Self::from_slice_with_strides) - Safe version with bounds checking
            /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Simpler version without custom strides
            /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Generic version
            #[inline]
            pub unsafe fn from_slice_with_strides_unchecked(data: &'a [T], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self { unsafe {
                Self::from_slice_with_strides_generic_unchecked(data, start, $($gargs,)* Dyn(rstride), Dyn(cstride))
            }}
        }
    }
);

// TODO: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<T, ..., S>
=> R: DimName, => C: DimName; // Type parameters for impl<T, ..., S>
R::name(), C::name();         // Arguments for `_generic` constructors.
); // Arguments for non-generic constructors.

impl_constructors!(R, Dyn;
                   => R: DimName;
                   R::name(), Dyn(ncols);
                   ncols);

impl_constructors!(Dyn, C;
                   => C: DimName;
                   Dyn(nrows), C::name();
                   nrows);

impl_constructors!(Dyn, Dyn;
                   ;
                   Dyn(nrows), Dyn(ncols);
                   nrows, ncols);

/// # Creating mutable matrix views from `&mut [T]`
impl<'a, T: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixViewMut<'a, T, R, C, RStride, CStride>
{
    /// Creates, without bounds checking, a mutable matrix view from a slice with custom dimensions and strides.
    ///
    /// This is a low-level function for creating mutable matrix views with full control over memory layout.
    /// It allows you to specify both the dimensions (rows and columns) and strides (memory spacing
    /// between elements) using generic type parameters.
    ///
    /// # What are strides?
    ///
    /// Strides control how elements are spaced in memory:
    /// - `rstride` (row stride): Number of elements to skip between consecutive rows
    /// - `cstride` (column stride): Number of elements to skip between consecutive columns
    ///
    /// For a standard column-major matrix, `rstride` is typically 1 (elements in a column are
    /// contiguous) and `cstride` equals the number of rows (columns are separated by that many elements).
    ///
    /// # Parameters
    ///
    /// - `data`: The mutable slice containing the matrix data
    /// - `start`: Index in the slice where the matrix data begins
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    /// - `rstride`: Row stride (type `RStride` - can be `Const<N>` or `Dyn`)
    /// - `cstride`: Column stride (type `CStride` - can be `Const<N>` or `Dyn`)
    ///
    /// # Safety
    ///
    /// This method is unsafe because:
    /// - The input data array is not checked to contain enough elements
    /// - Accessing elements using the provided strides must not go out of bounds
    /// - The caller must ensure: `data.len() >= start + (nrows-1)*rstride + (ncols-1)*cstride + 1`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Dyn};
    ///
    /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// // Create a mutable 2x2 view starting at index 1 with custom strides
    /// let mut view = unsafe {
    ///     MatrixViewMut::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic_unchecked(
    ///         &mut data,
    ///         1,           // start at index 1
    ///         Dyn(2),      // 2 rows
    ///         Dyn(2),      // 2 columns
    ///         Dyn(1),      // row stride = 1
    ///         Dyn(2),      // column stride = 2
    ///     )
    /// };
    ///
    /// // Modify the view
    /// view[(0, 0)] = 99.0;
    /// view[(1, 1)] = 88.0;
    ///
    /// // Original data is modified
    /// assert_eq!(data[1], 99.0);  // element at (0,0) in the view
    /// assert_eq!(data[4], 88.0);  // element at (1,1) in the view
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Safe version with bounds checking
    /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Simpler version without custom strides
    /// - [`from_slice_with_strides_unchecked`](Self::from_slice_with_strides_unchecked) - Runtime-sized version
    #[inline]
    pub const unsafe fn from_slice_with_strides_generic_unchecked(
        data: &'a mut [T],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        unsafe {
            let data = ViewStorageMut::from_raw_parts(
                data.as_mut_ptr().add(start),
                (nrows, ncols),
                (rstride, cstride),
            );
            Self::from_data(data)
        }
    }

    /// Creates a mutable matrix view from a slice with custom dimensions and strides.
    ///
    /// This function creates a mutable view into a slice with full control over the matrix dimensions
    /// and memory layout through stride parameters. It performs bounds checking to ensure
    /// the slice is large enough for the requested view.
    ///
    /// # What are strides?
    ///
    /// Strides control how elements are spaced in memory:
    /// - `rstride` (row stride): Number of elements to skip between consecutive rows
    /// - `cstride` (column stride): Number of elements to skip between consecutive columns
    ///
    /// For a standard column-major matrix, `rstride` is 1 and `cstride` equals the number of rows.
    ///
    /// # Parameters
    ///
    /// - `data`: The mutable slice containing the matrix data
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    /// - `rstride`: Row stride (type `RStride` - can be `Const<N>` or `Dyn`)
    /// - `cstride`: Column stride (type `CStride` - can be `Const<N>` or `Dyn`)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The input data slice does not contain enough elements for the specified dimensions and strides
    /// - The dimensions and strides would result in aliased indices (multiple matrix elements mapping to the same memory location)
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Dyn, Const};
    ///
    /// // Create a mutable view of a 2x3 matrix from a slice
    /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// // Standard column-major layout: rstride=1, cstride=nrows
    /// let mut view = MatrixViewMut::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic(
    ///     &mut data,
    ///     Dyn(2),      // 2 rows
    ///     Dyn(3),      // 3 columns
    ///     Dyn(1),      // row stride = 1
    ///     Dyn(2),      // column stride = 2
    /// );
    ///
    /// // Modify through the view
    /// view[(0, 0)] = 10.0;
    /// view[(1, 2)] = 60.0;
    ///
    /// assert_eq!(view[(0, 0)], 10.0);
    /// assert_eq!(data[0], 10.0);  // Original data is modified
    /// assert_eq!(data[5], 60.0);
    /// ```
    ///
    /// Custom strides for selective modification:
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Dyn};
    ///
    /// // Modify every other element
    /// let mut data = vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0];
    ///
    /// let mut view = MatrixViewMut::<_, Dyn, Dyn, Dyn, Dyn>::from_slice_with_strides_generic(
    ///     &mut data,
    ///     Dyn(2),      // 2 rows
    ///     Dyn(2),      // 2 columns
    ///     Dyn(2),      // row stride = 2 (skip padding)
    ///     Dyn(4),      // column stride = 4
    /// );
    ///
    /// // Multiply all viewed elements by 10
    /// view[(0, 0)] *= 10.0;
    /// view[(1, 0)] *= 10.0;
    /// view[(0, 1)] *= 10.0;
    /// view[(1, 1)] *= 10.0;
    ///
    /// // Only the non-99 values are modified
    /// assert_eq!(data[0], 10.0);
    /// assert_eq!(data[1], 99.0);  // Unchanged
    /// assert_eq!(data[2], 20.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Unchecked version
    /// - [`from_slice_generic`](Self::from_slice_generic) - Simpler version with default strides
    /// - [`from_slice_with_strides_mut`](Self::from_slice_with_strides_mut) - Runtime-sized version
    #[inline]
    pub fn from_slice_with_strides_generic(
        data: &'a mut [T],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix view: input data buffer too small."
        );

        assert!(
            {
                let nrows = nrows.value();
                let ncols = ncols.value();
                let rstride = rstride.value();
                let cstride = cstride.value();

                nrows * ncols <= 1
                    || match (rstride, cstride) {
                        (0, 0) => false,      // otherwise: matrix[(0, 0)] == index[(nrows - 1, ncols - 1)],
                        (0, _) => nrows <= 1, // otherwise: matrix[(0, 0)] == index[(nrows - 1, 0)],
                        (_, 0) => ncols <= 1, // otherwise: matrix[(0, 0)] == index[(0, ncols - 1)],
                        (_, _) => {
                            // otherwise: matrix[(0, numer)] == index[(denom, 0)]
                            let ratio = Ratio::new(rstride, cstride);
                            nrows <= *ratio.denom() || ncols <= *ratio.numer()
                        }
                    }
            },
            "Matrix view: dimensions and strides result in aliased indices."
        );

        unsafe {
            Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim> MatrixViewMut<'a, T, R, C> {
    /// Creates, without bounds checking, a mutable matrix view from a slice with specified dimensions.
    ///
    /// This is a simplified version of `from_slice_with_strides_generic_unchecked` that uses
    /// standard column-major strides (rstride=1, cstride=nrows). This is the most common
    /// layout for nalgebra matrices where elements in each column are stored contiguously.
    ///
    /// # Parameters
    ///
    /// - `data`: The mutable slice containing the matrix data in column-major order
    /// - `start`: Index in the slice where the matrix data begins
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    ///
    /// # Safety
    ///
    /// This method is unsafe because:
    /// - The input data array is not checked to contain enough elements
    /// - The caller must ensure: `data.len() >= start + nrows * ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Dyn};
    ///
    /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    ///
    /// // Create a mutable 2x3 view starting at index 2
    /// let mut view = unsafe {
    ///     MatrixViewMut::<_, Dyn, Dyn>::from_slice_generic_unchecked(
    ///         &mut data,
    ///         2,       // start at index 2
    ///         Dyn(2),  // 2 rows
    ///         Dyn(3),  // 3 columns
    ///     )
    /// };
    ///
    /// // Modify the view
    /// view[(0, 0)] = 100.0;
    /// view[(1, 2)] = 800.0;
    ///
    /// // Column-major layout: [100.0, 5.0, 7.0]
    /// //                      [4.0,   6.0, 800.0]
    /// assert_eq!(data[2], 100.0);  // Modified
    /// assert_eq!(data[7], 800.0);  // Modified
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_generic`](Self::from_slice_generic) - Safe version with bounds checking
    /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Version with custom strides
    /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Runtime-sized version
    #[inline]
    pub unsafe fn from_slice_generic_unchecked(
        data: &'a mut [T],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        unsafe {
            Self::from_slice_with_strides_generic_unchecked(
                data, start, nrows, ncols, Const::<1>, nrows,
            )
        }
    }

    /// Creates a mutable matrix view from a slice with specified dimensions.
    ///
    /// This is the primary safe function for creating mutable matrix views from slices using generic
    /// dimension parameters. It assumes standard column-major layout where elements in each
    /// column are stored contiguously in memory.
    ///
    /// # What is column-major layout?
    ///
    /// In column-major layout (used by nalgebra), matrix elements are stored column by column:
    /// - A 2x3 matrix `[a, b, c; d, e, f]` is stored as `[a, d, b, e, c, f]`
    /// - Elements in the same column are adjacent in memory
    ///
    /// # Parameters
    ///
    /// - `data`: The mutable slice containing matrix data in column-major order
    /// - `nrows`: Number of rows (type `R` - can be `Const<N>` or `Dyn`)
    /// - `ncols`: Number of columns (type `C` - can be `Const<N>` or `Dyn`)
    ///
    /// # Panics
    ///
    /// Panics if the input data slice does not contain enough elements.
    /// Requires: `data.len() >= nrows * ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Dyn, Const};
    ///
    /// // Create a mutable 2x3 matrix view from a slice
    /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ///
    /// let mut view = MatrixViewMut::<_, Dyn, Dyn>::from_slice_generic(
    ///     &mut data,
    ///     Dyn(2),  // 2 rows
    ///     Dyn(3),  // 3 columns
    /// );
    ///
    /// // Modify the matrix through the view
    /// view[(0, 0)] = 10.0;
    /// view[(1, 2)] = 60.0;
    ///
    /// // Column-major layout: [10.0, 3.0, 5.0]
    /// //                      [2.0,  4.0, 60.0]
    /// assert_eq!(view[(0, 0)], 10.0);
    /// assert_eq!(data[0], 10.0);  // Original data is modified
    /// assert_eq!(data[5], 60.0);
    /// ```
    ///
    /// Using compile-time dimensions with `Const`:
    ///
    /// ```
    /// use nalgebra::{MatrixViewMut, Const};
    ///
    /// let mut data = [1.0, 2.0, 3.0, 4.0];
    ///
    /// let mut view = MatrixViewMut::<_, Const<2>, Const<2>>::from_slice_generic(
    ///     &mut data,
    ///     Const::<2>,  // 2 rows (compile-time constant)
    ///     Const::<2>,  // 2 columns (compile-time constant)
    /// );
    ///
    /// // Fill diagonal with zeros
    /// view[(0, 0)] = 0.0;
    /// view[(1, 1)] = 0.0;
    ///
    /// assert_eq!(data[0], 0.0);
    /// assert_eq!(data[3], 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Unchecked version
    /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Version with custom strides
    /// - [`from_slice`](Self::from_slice) - Simpler runtime-sized version
    #[inline]
    pub fn from_slice_generic(data: &'a mut [T], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, Const::<1>, nrows)
    }
}

macro_rules! impl_constructors_mut(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr_2021),*; $($args: ident),*) => {
        impl<'a, T: Scalar, $($DimIdent: $DimBound),*> MatrixViewMut<'a, T, $($Dims),*> {
            /// Creates a mutable matrix view from a slice.
            ///
            /// This is the most convenient method for creating mutable matrix views from slices.
            /// The slice should contain the matrix data in column-major order (elements within
            /// each column are contiguous), and modifications to the view will modify the original slice.
            ///
            /// # What is column-major layout?
            ///
            /// In column-major layout (used by nalgebra), matrix elements are stored column by column:
            /// - A 2x3 matrix `[a, b, c; d, e, f]` is stored as `[a, d, b, e, c, f]`
            /// - Elements in the same column are adjacent in memory
            ///
            /// # Panics
            ///
            /// Panics if `data` does not contain enough elements for the requested dimensions.
            /// Requires: `data.len() >= nrows * ncols`
            ///
            /// # Examples
            ///
            /// Creating and modifying a dynamic matrix view:
            ///
            /// ```
            /// use nalgebra::DMatrixViewMut;
            ///
            /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a mutable 2x3 view
            /// let mut view = DMatrixViewMut::from_slice(&mut data, 2, 3);
            ///
            /// // Modify through the view
            /// view[(0, 0)] = 10.0;
            /// view[(1, 2)] = 60.0;
            ///
            /// // Original data is modified
            /// assert_eq!(data[0], 10.0);
            /// assert_eq!(data[5], 60.0);
            /// ```
            ///
            /// Creating a fixed-size mutable matrix view:
            ///
            /// ```
            /// use nalgebra::MatrixViewMut2x3;
            ///
            /// let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // No dimensions needed - they're in the type!
            /// let mut view = MatrixViewMut2x3::from_slice(&mut data);
            ///
            /// // Multiply all elements by 2
            /// for elem in view.iter_mut() {
            ///     *elem *= 2.0;
            /// }
            ///
            /// assert_eq!(data[0], 2.0);
            /// assert_eq!(data[5], 12.0);
            /// ```
            ///
            /// Modifying a partially dynamic view:
            ///
            /// ```
            /// use nalgebra::MatrixViewMut2xX;
            ///
            /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // 2 rows (fixed), dynamic columns
            /// let mut view = MatrixViewMut2xX::from_slice(&mut data, 3);
            ///
            /// // Set first column to zeros
            /// view.column_mut(0).fill(0.0);
            ///
            /// assert_eq!(data[0], 0.0);
            /// assert_eq!(data[1], 0.0);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Unchecked version
            /// - [`from_slice_with_strides_mut`](Self::from_slice_with_strides_mut) - Version with custom strides
            /// - [`from_slice_generic`](Self::from_slice_generic) - Generic version with type-level dimensions
            #[inline]
            pub fn from_slice(data: &'a mut [T], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bounds checking, a mutable matrix view from a slice.
            ///
            /// This is the unchecked version of [`from_slice`](Self::from_slice). It skips bounds
            /// checking and allows specifying a starting offset in the slice. Modifications to the
            /// view will modify the original slice.
            ///
            /// # Parameters
            ///
            /// - `data`: The mutable slice containing the matrix data in column-major order
            /// - `start`: Index in the slice where the matrix data begins
            /// - Additional parameters depend on the matrix type (dimensions for dynamic matrices)
            ///
            /// # Safety
            ///
            /// The caller must ensure:
            /// - `data.len() >= start + nrows * ncols`
            /// - The start index and dimensions don't cause out-of-bounds access
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::DMatrixViewMut;
            ///
            /// let mut data = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a mutable 2x3 view starting at index 1 (skipping the first element)
            /// let mut view = unsafe {
            ///     DMatrixViewMut::from_slice_unchecked(&mut data, 1, 2, 3)
            /// };
            ///
            /// // Modify through the view
            /// view[(0, 0)] = 100.0;
            /// view[(1, 2)] = 600.0;
            ///
            /// // Original data is modified (skipping first element)
            /// assert_eq!(data[0], 99.0);   // Unchanged
            /// assert_eq!(data[1], 100.0);  // Modified
            /// assert_eq!(data[6], 600.0);  // Modified
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice`](Self::from_slice) - Safe version with bounds checking
            /// - [`from_slice_with_strides_unchecked`](Self::from_slice_with_strides_unchecked) - Version with custom strides
            /// - [`from_slice_generic_unchecked`](Self::from_slice_generic_unchecked) - Generic version
            #[inline]
            pub unsafe fn from_slice_unchecked(data: &'a mut [T], start: usize, $($args: usize),*) -> Self { unsafe {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }}
        }

        impl<'a, T: Scalar, $($DimIdent: $DimBound, )*> MatrixViewMut<'a, T, $($Dims,)* Dyn, Dyn> {
            /// Creates a mutable matrix view with custom strides from a slice.
            ///
            /// This function allows you to create a mutable matrix view with non-standard memory layout
            /// by specifying custom row and column strides. This is useful for modifying
            /// non-contiguous data or data with padding. Modifications to the view will modify
            /// the original slice.
            ///
            /// # What are strides?
            ///
            /// Strides control how elements are spaced in memory:
            /// - `rstride` (row stride): Number of elements to skip between consecutive rows
            /// - `cstride` (column stride): Number of elements to skip between consecutive columns
            ///
            /// For a standard column-major matrix, `rstride=1` and `cstride=nrows`.
            ///
            /// # Panics
            ///
            /// Panics if:
            /// - `data` does not contain enough elements for the specified dimensions and strides
            /// - The dimensions and strides would result in aliased indices (multiple matrix elements
            ///   mapping to the same memory location)
            ///
            /// # Examples
            ///
            /// Standard usage with custom strides:
            ///
            /// ```
            /// use nalgebra::DMatrixViewMut;
            ///
            /// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a mutable 2x2 view with column stride = 3
            /// let mut view = DMatrixViewMut::from_slice_with_strides_mut(&mut data, 2, 2, 1, 3);
            ///
            /// // Double all viewed elements
            /// for elem in view.iter_mut() {
            ///     *elem *= 2.0;
            /// }
            ///
            /// // View was: [1.0, 4.0]  -> Now: [2.0, 8.0]
            /// //           [2.0, 5.0]         [4.0, 10.0]
            /// assert_eq!(data[0], 2.0);
            /// assert_eq!(data[1], 4.0);
            /// assert_eq!(data[3], 8.0);
            /// assert_eq!(data[4], 10.0);
            /// ```
            ///
            /// Modifying every other element:
            ///
            /// ```
            /// use nalgebra::DMatrixViewMut;
            ///
            /// let mut data = vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0];
            ///
            /// // Extract and modify every other element using strides
            /// let mut view = DMatrixViewMut::from_slice_with_strides_mut(&mut data, 2, 2, 2, 4);
            ///
            /// // Set all viewed elements to zero
            /// view.fill(0.0);
            ///
            /// // Only non-99 values were modified
            /// assert_eq!(data[0], 0.0);
            /// assert_eq!(data[1], 99.0);  // Unchanged
            /// assert_eq!(data[2], 0.0);
            /// assert_eq!(data[3], 99.0);  // Unchanged
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice`](Self::from_slice) - Simpler version with default strides
            /// - [`from_slice_with_strides_unchecked`](Self::from_slice_with_strides_unchecked) - Unchecked version
            /// - [`from_slice_with_strides_generic`](Self::from_slice_with_strides_generic) - Generic version
            #[inline]
            pub fn from_slice_with_strides_mut(data: &'a mut [T], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(
                    data, $($gargs,)* Dyn(rstride), Dyn(cstride))
            }

            /// Creates, without bounds checking, a mutable matrix view with custom strides from a slice.
            ///
            /// This is the unchecked version of [`from_slice_with_strides_mut`](Self::from_slice_with_strides_mut).
            /// It skips bounds checking and allows specifying a starting offset in the slice.
            /// Modifications to the view will modify the original slice.
            ///
            /// # Parameters
            ///
            /// - `data`: The mutable slice containing the matrix data
            /// - `start`: Index in the slice where the matrix data begins
            /// - Additional dimension parameters (nrows, ncols for fully dynamic matrices)
            /// - `rstride`: Row stride (elements to skip between consecutive rows)
            /// - `cstride`: Column stride (elements to skip between consecutive columns)
            ///
            /// # Safety
            ///
            /// The caller must ensure:
            /// - `data.len() >= start + (nrows-1)*rstride + (ncols-1)*cstride + 1`
            /// - The strides don't cause out-of-bounds access when accessing matrix elements
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::DMatrixViewMut;
            ///
            /// let mut data = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            ///
            /// // Create a mutable 2x2 view starting at index 1 with custom strides
            /// let mut view = unsafe {
            ///     DMatrixViewMut::from_slice_with_strides_unchecked(
            ///         &mut data,
            ///         1,    // start
            ///         2, 2, // dimensions
            ///         1, 3, // strides
            ///     )
            /// };
            ///
            /// // Add 100 to all viewed elements
            /// for elem in view.iter_mut() {
            ///     *elem += 100.0;
            /// }
            ///
            /// // Creates: [101.0, 104.0]
            /// //          [102.0, 105.0]
            /// assert_eq!(data[1], 101.0);
            /// assert_eq!(data[4], 104.0);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`from_slice_with_strides_mut`](Self::from_slice_with_strides_mut) - Safe version with bounds checking
            /// - [`from_slice_unchecked`](Self::from_slice_unchecked) - Simpler version without custom strides
            /// - [`from_slice_with_strides_generic_unchecked`](Self::from_slice_with_strides_generic_unchecked) - Generic version
            #[inline]
            pub unsafe fn from_slice_with_strides_unchecked(data: &'a mut [T], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self { unsafe {
                Self::from_slice_with_strides_generic_unchecked(
                    data, start, $($gargs,)* Dyn(rstride), Dyn(cstride))
            }}
        }
    }
);

// TODO: this is not very pretty. We could find a better call syntax.
impl_constructors_mut!(R, C;                         // Arguments for Matrix<T, ..., S>
=> R: DimName, => C: DimName; // Type parameters for impl<T, ..., S>
R::name(), C::name();         // Arguments for `_generic` constructors.
); // Arguments for non-generic constructors.

impl_constructors_mut!(R, Dyn;
                       => R: DimName;
                       R::name(), Dyn(ncols);
                       ncols);

impl_constructors_mut!(Dyn, C;
                       => C: DimName;
                       Dyn(nrows), C::name();
                       nrows);

impl_constructors_mut!(Dyn, Dyn;
                       ;
                       Dyn(nrows), Dyn(ncols);
                       nrows, ncols);
