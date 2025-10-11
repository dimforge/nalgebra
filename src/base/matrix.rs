// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use num::{One, Zero};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "rkyv-serialize-no-std")]
use super::rkyv_wrappers::CustomPhantom;
#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;
#[cfg(feature = "rkyv-serialize-no-std")]
use rkyv::{Archive, Archived, with::With};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, Field, SupersetOf};
use simba::simd::SimdPartialOrd;

use crate::base::allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR};
use crate::base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, DimAdd, DimSum, IsNotStaticOne, U1, U2, U3};
use crate::base::iter::{
    ColumnIter, ColumnIterMut, MatrixIter, MatrixIterMut, RowIter, RowIterMut,
};
use crate::base::storage::{Owned, RawStorage, RawStorageMut, SameShapeStorage};
use crate::base::{Const, DefaultAllocator, OMatrix, OVector, Scalar, Unit};
use crate::{ArrayStorage, SMatrix, SimdComplexField, Storage, UninitMatrix};

use crate::storage::IsContiguous;
use crate::uninit::{Init, InitStatus, Uninit};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::{DMatrix, DVector, Dyn, RowDVector, VecStorage};
use std::mem::MaybeUninit;

/// A square matrix.
pub type SquareMatrix<T, D, S> = Matrix<T, D, D, S>;

/// A matrix with one column and `D` rows.
pub type Vector<T, D, S> = Matrix<T, D, U1, S>;

/// A matrix with one row and `D` columns .
pub type RowVector<T, D, S> = Matrix<T, U1, D, S>;

/// The type of the result of a matrix sum.
pub type MatrixSum<T, R1, C1, R2, C2> =
    Matrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<T, R1, C1, R2, C2>>;

/// The type of the result of a matrix sum.
pub type VectorSum<T, R1, R2> =
    Matrix<T, SameShapeR<R1, R2>, U1, SameShapeStorage<T, R1, U1, R2, U1>>;

/// The type of the result of a matrix cross product.
pub type MatrixCross<T, R1, C1, R2, C2> =
    Matrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<T, R1, C1, R2, C2>>;

/// The most generic column-major matrix (and vector) type.
///
/// # Methods summary
/// Because `Matrix` is the most generic types used as a common representation of all matrices and
/// vectors of **nalgebra** this documentation page contains every single matrix/vector-related
/// method. In order to make browsing this page simpler, the next subsections contain direct links
/// to groups of methods related to a specific topic.
///
/// #### Vector and matrix construction
/// - [Constructors of statically-sized vectors or statically-sized matrices](#constructors-of-statically-sized-vectors-or-statically-sized-matrices)
///   (`Vector3`, `Matrix3x6`…)
/// - [Constructors of fully dynamic matrices](#constructors-of-fully-dynamic-matrices) (`DMatrix`)
/// - [Constructors of dynamic vectors and matrices with a dynamic number of rows](#constructors-of-dynamic-vectors-and-matrices-with-a-dynamic-number-of-rows)
///   (`DVector`, `MatrixXx3`…)
/// - [Constructors of matrices with a dynamic number of columns](#constructors-of-matrices-with-a-dynamic-number-of-columns)
///   (`Matrix2xX`…)
/// - [Generic constructors](#generic-constructors)
///   (For code generic wrt. the vectors or matrices dimensions.)
///
/// #### Computer graphics utilities for transformations
/// - [2D transformations as a Matrix3 <span style="float:right;">`new_rotation`…</span>](#2d-transformations-as-a-matrix3)
/// - [3D transformations as a Matrix4 <span style="float:right;">`new_rotation`, `new_perspective`, `look_at_rh`…</span>](#3d-transformations-as-a-matrix4)
/// - [Translation and scaling in any dimension <span style="float:right;">`new_scaling`, `new_translation`…</span>](#translation-and-scaling-in-any-dimension)
/// - [Append/prepend translation and scaling <span style="float:right;">`append_scaling`, `prepend_translation_mut`…</span>](#appendprepend-translation-and-scaling)
/// - [Transformation of vectors and points <span style="float:right;">`transform_vector`, `transform_point`…</span>](#transformation-of-vectors-and-points)
///
/// #### Common math operations
/// - [Componentwise operations <span style="float:right;">`component_mul`, `component_div`, `inf`…</span>](#componentwise-operations)
/// - [Special multiplications <span style="float:right;">`tr_mul`, `ad_mul`, `kronecker`…</span>](#special-multiplications)
/// - [Dot/scalar product <span style="float:right;">`dot`, `dotc`, `tr_dot`…</span>](#dotscalar-product)
/// - [Cross product <span style="float:right;">`cross`, `perp`…</span>](#cross-product)
/// - [Magnitude and norms <span style="float:right;">`norm`, `normalize`, `metric_distance`…</span>](#magnitude-and-norms)
/// - [In-place normalization <span style="float:right;">`normalize_mut`, `try_normalize_mut`…</span>](#in-place-normalization)
/// - [Interpolation <span style="float:right;">`lerp`, `slerp`…</span>](#interpolation)
/// - [BLAS functions <span style="float:right;">`gemv`, `gemm`, `syger`…</span>](#blas-functions)
/// - [Swizzling <span style="float:right;">`xx`, `yxz`…</span>](#swizzling)
/// - [Triangular matrix extraction <span style="float:right;">`upper_triangle`, `lower_triangle`</span>](#triangular-matrix-extraction)
///
/// #### Statistics
/// - [Common operations <span style="float:right;">`row_sum`, `column_mean`, `variance`…</span>](#common-statistics-operations)
/// - [Find the min and max components <span style="float:right;">`min`, `max`, `amin`, `amax`, `camin`, `cmax`…</span>](#find-the-min-and-max-components)
/// - [Find the min and max components (vector-specific methods) <span style="float:right;">`argmin`, `argmax`, `icamin`, `icamax`…</span>](#find-the-min-and-max-components-vector-specific-methods)
///
/// #### Iteration, map, and fold
/// - [Iteration on components, rows, and columns <span style="float:right;">`iter`, `column_iter`…</span>](#iteration-on-components-rows-and-columns)
/// - [Parallel iterators using rayon <span style="float:right;">`par_column_iter`, `par_column_iter_mut`…</span>](#parallel-iterators-using-rayon)
/// - [Elementwise mapping and folding <span style="float:right;">`map`, `fold`, `zip_map`…</span>](#elementwise-mapping-and-folding)
/// - [Folding or columns and rows <span style="float:right;">`compress_rows`, `compress_columns`…</span>](#folding-on-columns-and-rows)
///
/// #### Vector and matrix views
/// - [Creating matrix views from `&[T]` <span style="float:right;">`from_slice`, `from_slice_with_strides`…</span>](#creating-matrix-views-from-t)
/// - [Creating mutable matrix views from `&mut [T]` <span style="float:right;">`from_slice_mut`, `from_slice_with_strides_mut`…</span>](#creating-mutable-matrix-views-from-mut-t)
/// - [Views based on index and length <span style="float:right;">`row`, `columns`, `view`…</span>](#views-based-on-index-and-length)
/// - [Mutable views based on index and length <span style="float:right;">`row_mut`, `columns_mut`, `view_mut`…</span>](#mutable-views-based-on-index-and-length)
/// - [Views based on ranges <span style="float:right;">`rows_range`, `columns_range`…</span>](#views-based-on-ranges)
/// - [Mutable views based on ranges <span style="float:right;">`rows_range_mut`, `columns_range_mut`…</span>](#mutable-views-based-on-ranges)
///
/// #### In-place modification of a single matrix or vector
/// - [In-place filling <span style="float:right;">`fill`, `fill_diagonal`, `fill_with_identity`…</span>](#in-place-filling)
/// - [In-place swapping <span style="float:right;">`swap`, `swap_columns`…</span>](#in-place-swapping)
/// - [Set rows, columns, and diagonal <span style="float:right;">`set_column`, `set_diagonal`…</span>](#set-rows-columns-and-diagonal)
///
/// #### Vector and matrix size modification
/// - [Rows and columns insertion <span style="float:right;">`insert_row`, `insert_column`…</span>](#rows-and-columns-insertion)
/// - [Rows and columns removal <span style="float:right;">`remove_row`, `remove column`…</span>](#rows-and-columns-removal)
/// - [Rows and columns extraction <span style="float:right;">`select_rows`, `select_columns`…</span>](#rows-and-columns-extraction)
/// - [Resizing and reshaping <span style="float:right;">`resize`, `reshape_generic`…</span>](#resizing-and-reshaping)
/// - [In-place resizing <span style="float:right;">`resize_mut`, `resize_vertically_mut`…</span>](#in-place-resizing)
///
/// #### Matrix decomposition
/// - [Rectangular matrix decomposition <span style="float:right;">`qr`, `lu`, `svd`…</span>](#rectangular-matrix-decomposition)
/// - [Square matrix decomposition <span style="float:right;">`cholesky`, `symmetric_eigen`…</span>](#square-matrix-decomposition)
///
/// #### Vector basis computation
/// - [Basis and orthogonalization <span style="float:right;">`orthonormal_subspace_basis`, `orthonormalize`…</span>](#basis-and-orthogonalization)
///
/// # Type parameters
/// The generic `Matrix` type has four type parameters:
/// - `T`: for the matrix components scalar type.
/// - `R`: for the matrix number of rows.
/// - `C`: for the matrix number of columns.
/// - `S`: for the matrix data storage, i.e., the buffer that actually contains the matrix
///   components.
///
/// The matrix dimensions parameters `R` and `C` can either be:
/// - type-level unsigned integer constants (e.g. `U1`, `U124`) from the `nalgebra::` root module.
///   All numbers from 0 to 127 are defined that way.
/// - type-level unsigned integer constants (e.g. `U1024`, `U10000`) from the `typenum::` crate.
///   Using those, you will not get error messages as nice as for numbers smaller than 128 defined on
///   the `nalgebra::` module.
/// - the special value `Dyn` from the `nalgebra::` root module. This indicates that the
///   specified dimension is not known at compile-time. Note that this will generally imply that the
///   matrix data storage `S` performs a dynamic allocation and contains extra metadata for the
///   matrix shape.
///
/// Note that mixing `Dyn` with type-level unsigned integers is allowed. Actually, a
/// dynamically-sized column vector should be represented as a `Matrix<T, Dyn, U1, S>` (given
/// some concrete types for `T` and a compatible data storage type `S`).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Matrix<T::Archived, R, C, S::Archived>",
        bound(archive = "
        T: Archive,
        S: Archive,
        With<PhantomData<(T, R, C)>, CustomPhantom<(Archived<T>, R, C)>>: Archive<Archived = PhantomData<(Archived<T>, R, C)>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Matrix<T, R, C, S> {
    /// The data storage that contains all the matrix components. Disappointed?
    ///
    /// Well, if you came here to see how you can access the matrix components,
    /// you may be in luck: you can access the individual components of all vectors with compile-time
    /// dimensions <= 6 using field notation like this:
    /// `vec.x`, `vec.y`, `vec.z`, `vec.w`, `vec.a`, `vec.b`. Reference and assignation work too:
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec = Vector3::new(1.0, 2.0, 3.0);
    /// vec.x = 10.0;
    /// vec.y += 30.0;
    /// assert_eq!(vec.x, 10.0);
    /// assert_eq!(vec.y + 100.0, 132.0);
    /// ```
    /// Similarly, for matrices with compile-time dimensions <= 6, you can use field notation
    /// like this: `mat.m11`, `mat.m42`, etc. The first digit identifies the row to address
    /// and the second digit identifies the column to address. So `mat.m13` identifies the component
    /// at the first row and third column (note that the count of rows and columns start at 1 instead
    /// of 0 here. This is so we match the mathematical notation).
    ///
    /// For all matrices and vectors, independently from their size, individual components can
    /// be accessed and modified using indexing: `vec[20]`, `mat[(20, 19)]`. Here the indexing
    /// starts at 0 as you would expect.
    pub data: S,

    // NOTE: the fact that this field is private is important because
    //       this prevents the user from constructing a matrix with
    //       dimensions R, C that don't match the dimension of the
    //       storage S. Instead they have to use the unsafe function
    //       from_data_statically_unchecked.
    //       Note that it would probably make sense to just have
    //       the type `Matrix<S>`, and have `T, R, C` be associated-types
    //       of the `RawStorage` trait. However, because we don't have
    //       specialization, this is not possible because these `T, R, C`
    //       allows us to desambiguate a lot of configurations.
    #[cfg_attr(feature = "rkyv-serialize-no-std", with(CustomPhantom<(T::Archived, R, C)>))]
    _phantoms: PhantomData<(T, R, C)>,
}

impl<T, R: Dim, C: Dim, S: fmt::Debug> fmt::Debug for Matrix<T, R, C, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.data.fmt(formatter)
    }
}

impl<T, R, C, S> Default for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: Default,
{
    fn default() -> Self {
        Matrix {
            data: Default::default(),
            _phantoms: PhantomData,
        }
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T, R, C, S> Serialize for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de, T, R, C, S> Deserialize<'de> for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        S::deserialize(deserializer).map(|x| Matrix {
            data: x,
            _phantoms: PhantomData,
        })
    }
}

#[cfg(feature = "compare")]
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> matrixcompare_core::Matrix<T>
    for Matrix<T, R, C, S>
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> matrixcompare_core::Access<'_, T> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "compare")]
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> matrixcompare_core::DenseAccess<T>
    for Matrix<T, R, C, S>
{
    fn fetch_single(&self, row: usize, col: usize) -> T {
        self.index((row, col)).clone()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> bytemuck::Zeroable
    for Matrix<T, R, C, S>
where
    S: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> bytemuck::Pod for Matrix<T, R, C, S>
where
    S: bytemuck::Pod,
    Self: Copy,
{
}

impl<T, R, C, S> Matrix<T, R, C, S> {
    /// Creates a new matrix with the given data without statically checking that the matrix
    /// dimension matches the storage dimension.
    ///
    /// # Safety
    ///
    /// The storage dimension must match the given dimensions.
    #[inline(always)]
    pub const unsafe fn from_data_statically_unchecked(data: S) -> Matrix<T, R, C, S> {
        Matrix {
            data,
            _phantoms: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> SMatrix<T, R, C> {
    /// Creates a new statically-allocated matrix from the given [`ArrayStorage`].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    #[inline(always)]
    pub const fn from_array_storage(storage: ArrayStorage<T, R, C>) -> Self {
        // This is sound because the row and column types are exactly the same as that of the
        // storage, so there can be no mismatch
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

// TODO: Consider removing/deprecating `from_vec_storage` once we are able to make
// `from_data` const fn compatible
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> DMatrix<T> {
    /// Creates a new heap-allocated matrix from the given [`VecStorage`].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    pub const fn from_vec_storage(storage: VecStorage<T, Dyn, Dyn>) -> Self {
        // This is sound because the dimensions of the matrix and the storage are guaranteed
        // to be the same
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

// TODO: Consider removing/deprecating `from_vec_storage` once we are able to make
// `from_data` const fn compatible
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> DVector<T> {
    /// Creates a new heap-allocated matrix from the given [`VecStorage`].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    pub const fn from_vec_storage(storage: VecStorage<T, Dyn, U1>) -> Self {
        // This is sound because the dimensions of the matrix and the storage are guaranteed
        // to be the same
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

// TODO: Consider removing/deprecating `from_vec_storage` once we are able to make
// `from_data` const fn compatible
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> RowDVector<T> {
    /// Creates a new heap-allocated matrix from the given [`VecStorage`].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    pub const fn from_vec_storage(storage: VecStorage<T, U1, Dyn>) -> Self {
        // This is sound because the dimensions of the matrix and the storage are guaranteed
        // to be the same
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

impl<T: Scalar, R: Dim, C: Dim> UninitMatrix<T, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    /// Assumes a matrix's entries to be initialized. This operation should be near zero-cost.
    ///
    /// # Safety
    /// The user must make sure that every single entry of the buffer has been initialized,
    /// or Undefined Behavior will immediately occur.
    #[inline(always)]
    pub unsafe fn assume_init(self) -> OMatrix<T, R, C> {
        unsafe {
            OMatrix::from_data(<DefaultAllocator as Allocator<R, C>>::assume_init(
                self.data,
            ))
        }
    }
}

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Creates a new matrix with the given data.
    #[inline(always)]
    pub const fn from_data(data: S) -> Self {
        unsafe { Self::from_data_statically_unchecked(data) }
    }

    /// Returns the shape of this matrix as a tuple `(nrows, ncols)`.
    ///
    /// This method returns the dimensions of the matrix: the first element is the number
    /// of rows, and the second element is the number of columns. The values are returned
    /// as runtime `usize` values, regardless of whether the dimensions are known at
    /// compile-time or dynamically allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, DMatrix};
    /// // Static size matrix
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.shape(), (3, 4));
    ///
    /// // Dynamic size matrix
    /// let dmat = DMatrix::<f64>::zeros(5, 7);
    /// assert_eq!(dmat.shape(), (5, 7));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`shape_generic`](#method.shape_generic) - Returns dimensions as type-level values
    /// - [`nrows`](#method.nrows) - Returns only the number of rows
    /// - [`ncols`](#method.ncols) - Returns only the number of columns
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.shape_generic();
        (nrows.value(), ncols.value())
    }

    /// Returns the shape of this matrix as type-level dimensions.
    ///
    /// Unlike [`shape`](#method.shape), which returns runtime `usize` values, this method
    /// returns the dimensions wrapped in their compile-time type representations: either
    /// `Const<N>` for statically-sized dimensions, or `Dyn` for dynamically-sized dimensions.
    ///
    /// This is useful for type-level programming and when you need to preserve compile-time
    /// dimension information for type safety.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, DMatrix, Const, Dyn};
    /// // Static size matrix - returns (Const<3>, Const<4>)
    /// let mat = Matrix3x4::<f32>::zeros();
    /// let (rows, cols) = mat.shape_generic();
    /// assert_eq!(rows.value(), 3);
    /// assert_eq!(cols.value(), 4);
    ///
    /// // Dynamic size matrix - returns (Dyn, Dyn)
    /// let dmat = DMatrix::<f64>::zeros(5, 7);
    /// let (rows, cols) = dmat.shape_generic();
    /// assert_eq!(rows.value(), 5);
    /// assert_eq!(cols.value(), 7);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`shape`](#method.shape) - Returns dimensions as runtime `usize` values
    /// - [`nrows`](#method.nrows) - Returns only the number of rows
    /// - [`ncols`](#method.ncols) - Returns only the number of columns
    #[inline]
    #[must_use]
    pub fn shape_generic(&self) -> (R, C) {
        self.data.shape()
    }

    /// Returns the number of rows in this matrix.
    ///
    /// This is a convenience method that extracts just the row count from the matrix's shape.
    /// It's equivalent to calling `self.shape().0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, DMatrix, Vector5};
    /// // Rectangular matrix
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.nrows(), 3);
    ///
    /// // Column vector
    /// let vec = Vector5::<f64>::zeros();
    /// assert_eq!(vec.nrows(), 5);
    ///
    /// // Dynamic matrix
    /// let dmat = DMatrix::<i32>::zeros(10, 5);
    /// assert_eq!(dmat.nrows(), 10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ncols`](#method.ncols) - Returns the number of columns
    /// - [`shape`](#method.shape) - Returns both dimensions as a tuple
    /// - [`len`](#method.len) - Returns the total number of elements
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    /// Returns the number of columns in this matrix.
    ///
    /// This is a convenience method that extracts just the column count from the matrix's shape.
    /// It's equivalent to calling `self.shape().1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, DMatrix, RowVector6};
    /// // Rectangular matrix
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.ncols(), 4);
    ///
    /// // Row vector
    /// let vec = RowVector6::<f64>::zeros();
    /// assert_eq!(vec.ncols(), 6);
    ///
    /// // Dynamic matrix
    /// let dmat = DMatrix::<i32>::zeros(10, 5);
    /// assert_eq!(dmat.ncols(), 5);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`nrows`](#method.nrows) - Returns the number of rows
    /// - [`shape`](#method.shape) - Returns both dimensions as a tuple
    /// - [`len`](#method.len) - Returns the total number of elements
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    /// Returns the strides between elements in memory as a tuple `(row_stride, column_stride)`.
    ///
    /// The strides indicate how many elements to skip in memory to move to the next row or column.
    /// - **Row stride**: Number of elements between consecutive rows (within the same column)
    /// - **Column stride**: Number of elements between consecutive columns (within the same row)
    ///
    /// For a standard column-major matrix with `nrows` rows:
    /// - Row stride is typically `1` (elements in the same column are contiguous)
    /// - Column stride is typically `nrows` (columns are stored sequentially)
    ///
    /// Matrix views or slices may have different strides.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, DMatrix};
    /// // Standard matrix has row_stride=1, col_stride=nrows
    /// let mat = Matrix3::<f32>::zeros();
    /// assert_eq!(mat.strides(), (1, 3));
    ///
    /// // Dynamic matrix
    /// let dmat = DMatrix::<f64>::zeros(10, 5);
    /// assert_eq!(dmat.strides(), (1, 10));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`shape`](#method.shape) - Returns the dimensions of the matrix
    /// - [`as_slice`](#method.as_slice) - Access contiguous matrix data as a slice
    #[inline]
    #[must_use]
    pub fn strides(&self) -> (usize, usize) {
        let (srows, scols) = self.data.strides();
        (srows.value(), scols.value())
    }

    /// Converts a linear vector index to matrix row and column coordinates.
    ///
    /// When treating the matrix as a flat vector (in column-major order), this method converts
    /// a linear index into the corresponding (row, column) coordinates in the matrix.
    ///
    /// Elements are indexed in column-major order: the first column is indexed from 0 to nrows-1,
    /// the second column from nrows to 2*nrows-1, and so on.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// // Matrix in column-major order: [1, 3, 2, 4]
    /// assert_eq!(m.vector_to_matrix_index(0), (0, 0)); // element 1
    /// assert_eq!(m.vector_to_matrix_index(1), (1, 0)); // element 3
    /// assert_eq!(m.vector_to_matrix_index(2), (0, 1)); // element 2
    /// assert_eq!(m.vector_to_matrix_index(3), (1, 1)); // element 4
    ///
    /// // Verify the conversion
    /// let i = m.vector_to_matrix_index(3);
    /// assert_eq!(m[i], m[3]);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(1, 2,
    ///                        3, 4,
    ///                        5, 6);
    /// assert_eq!(m.vector_to_matrix_index(4), (1, 1)); // element 4
    /// ```
    ///
    /// # See Also
    ///
    /// - [`len`](#method.len) - Returns the total number of elements
    /// - Indexing with `matrix[index]` for linear element access
    #[inline]
    #[must_use]
    pub fn vector_to_matrix_index(&self, i: usize) -> (usize, usize) {
        let (nrows, ncols) = self.shape();

        // Two most common uses that should be optimized by the compiler for statically-sized
        // matrices.
        if nrows == 1 {
            (0, i)
        } else if ncols == 1 {
            (i, 0)
        } else {
            (i % nrows, i / nrows)
        }
    }

    /// Returns a raw pointer to the first element of the matrix.
    ///
    /// This provides direct access to the underlying memory. The matrix data is stored in
    /// column-major order, meaning elements within each column are contiguous in memory.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be properly aligned and non-null.
    ///
    /// # Safety
    ///
    /// While this method itself is safe, dereferencing the returned pointer requires `unsafe` code.
    /// The caller must ensure:
    /// - The pointer is only used while the matrix is valid and hasn't been moved
    /// - Proper bounds are maintained (use [`len`](#method.len) to check the element count)
    /// - Concurrent access follows Rust's aliasing rules
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let ptr = m.as_ptr();
    ///
    /// // Elements are stored in column-major order: [1, 3, 2, 4]
    /// unsafe {
    ///     assert_eq!(*ptr, 1);
    ///     assert_eq!(*ptr.add(1), 3);
    ///     assert_eq!(*ptr.add(2), 2);
    ///     assert_eq!(*ptr.add(3), 4);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`as_mut_ptr`](#method.as_mut_ptr) - Returns a mutable pointer
    /// - [`as_slice`](#method.as_slice) - Returns a safe slice for contiguous matrices
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.ptr()
    }

    /// Tests whether two matrices are approximately equal using relative and absolute tolerances.
    ///
    /// This method compares each corresponding pair of elements and returns `true` if all pairs
    /// satisfy the relative equality test. This is useful when working with floating-point numbers
    /// where exact equality is often not achievable due to rounding errors.
    ///
    /// The comparison uses two tolerance parameters:
    /// - `eps`: The absolute tolerance (epsilon) for small values
    /// - `max_relative`: The relative tolerance for larger values
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// let b = Matrix2::new(1.0000001, 2.0,
    ///                      3.0, 4.0);
    ///
    /// // Not exactly equal
    /// assert_ne!(a, b);
    ///
    /// // But approximately equal with tolerance
    /// assert!(a.relative_eq(&b, 1e-5, 1e-5));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`abs_diff_eq`](#method.abs_diff_eq) - Simpler absolute difference comparison
    /// - The `approx` crate for more comparison options
    #[inline]
    #[must_use]
    pub fn relative_eq<R2, C2, SB>(
        &self,
        other: &Matrix<T, R2, C2, SB>,
        eps: T::Epsilon,
        max_relative: T::Epsilon,
    ) -> bool
    where
        T: RelativeEq + Scalar,
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        T::Epsilon: Clone,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.relative_eq(b, eps.clone(), max_relative.clone()))
    }

    /// Tests whether two matrices are exactly equal element-wise.
    ///
    /// This method compares each corresponding pair of elements and returns `true` if all pairs
    /// are exactly equal according to the `PartialEq` trait. Unlike the standard `==` operator,
    /// this method works with matrices of different storage types.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, DMatrix};
    /// let a = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let b = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let c = Matrix2::new(1, 2,
    ///                      3, 5);
    ///
    /// assert!(a.eq(&b));
    /// assert!(!a.eq(&c));
    /// ```
    ///
    /// ```
    /// # use nalgebra::{Matrix2, DMatrix};
    /// // Compare matrices with different storage types
    /// let static_mat = Matrix2::new(1.0, 2.0,
    ///                               3.0, 4.0);
    /// let dynamic_mat = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// assert!(static_mat.eq(&dynamic_mat));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`relative_eq`](#method.relative_eq) - Approximate equality with tolerance
    /// - The standard `==` operator for matrices with the same storage type
    #[inline]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn eq<R2, C2, SB>(&self, other: &Matrix<T, R2, C2, SB>) -> bool
    where
        T: PartialEq,
        R2: Dim,
        C2: Dim,
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| *a == *b)
    }

    /// Converts this matrix into one that owns its data.
    ///
    /// If the matrix already owns its data (e.g., it's a `Matrix` with `ArrayStorage` or `VecStorage`),
    /// this is essentially a no-op. If it's a matrix view (e.g., a slice), this creates a new owned
    /// matrix with a copy of the data.
    ///
    /// This is useful when you need to store a matrix beyond the lifetime of borrowed data, or when
    /// you need to modify a matrix that was originally a view.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, DMatrix};
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Convert a matrix view to an owned matrix
    /// let row_view = m.row(0);
    /// let owned = row_view.into_owned();
    /// // Now `owned` can outlive `m`
    /// drop(m);
    /// assert_eq!(owned[(0, 0)], 1);
    /// ```
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// // Useful for storing views in data structures
    /// let m = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// let col = m.column(1).into_owned();
    /// assert_eq!(col.len(), 3);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`clone_owned`](#method.clone_owned) - Non-consuming version
    /// - [`into_owned_sum`](#method.into_owned_sum) - For type-level storage combination
    #[inline]
    pub fn into_owned(self) -> OMatrix<T, R, C>
    where
        T: Scalar,
        S: Storage<T, R, C>,
        DefaultAllocator: Allocator<R, C>,
    {
        Matrix::from_data(self.data.into_owned())
    }

    // TODO: this could probably benefit from specialization.
    // XXX: bad name.
    /// Converts this matrix into one that owns its data, with storage type suitable for addition.
    ///
    /// This method is similar to [`into_owned`](#method.into_owned), but the resulting storage
    /// type is determined by the type-level storage combination rules used for matrix addition.
    /// This is useful in generic code where you need to ensure the result is compatible with
    /// addition operations involving another matrix of dimensions `R2 × C2`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, U2};
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let row_view = m.row(0);
    ///
    /// // Convert to owned with storage type compatible for addition
    /// let owned = row_view.into_owned_sum::<U2, U2>();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`into_owned`](#method.into_owned) - Simpler version without type-level storage rules
    /// - [`clone_owned_sum`](#method.clone_owned_sum) - Non-consuming version
    #[inline]
    pub fn into_owned_sum<R2, C2>(self) -> MatrixSum<T, R, C, R2, C2>
    where
        T: Scalar,
        S: Storage<T, R, C>,
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        if TypeId::of::<SameShapeStorage<T, R, C, R2, C2>>() == TypeId::of::<Owned<T, R, C>>() {
            // We can just return `self.into_owned()`.

            unsafe {
                // TODO: check that those copies are optimized away by the compiler.
                let owned = self.into_owned();
                let res = mem::transmute_copy(&owned);
                mem::forget(owned);
                res
            }
        } else {
            self.clone_owned_sum()
        }
    }

    /// Clones this matrix, creating one that owns its data.
    ///
    /// This is similar to [`into_owned`](#method.into_owned) but doesn't consume the original matrix.
    /// It always creates a new owned matrix with a copy of the data, regardless of whether the
    /// original matrix owns its data or is a view.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Clone a view into an owned matrix
    /// let row_view = m.row(0);
    /// let owned = row_view.clone_owned();
    ///
    /// // Both the original and the clone are still accessible
    /// assert_eq!(row_view[(0, 0)], 1);
    /// assert_eq!(owned[(0, 0)], 1);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = v.clone_owned();
    /// assert_eq!(v, v2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`into_owned`](#method.into_owned) - Consuming version
    /// - [`clone_owned_sum`](#method.clone_owned_sum) - For type-level storage combination
    #[inline]
    #[must_use]
    pub fn clone_owned(&self) -> OMatrix<T, R, C>
    where
        T: Scalar,
        S: Storage<T, R, C>,
        DefaultAllocator: Allocator<R, C>,
    {
        Matrix::from_data(self.data.clone_owned())
    }

    /// Clones this matrix, creating one that owns its data with storage type suitable for addition.
    ///
    /// This method is similar to [`clone_owned`](#method.clone_owned), but the resulting storage
    /// type is determined by the type-level storage combination rules used for matrix addition.
    /// This is useful in generic code where you need to ensure the result is compatible with
    /// addition operations involving another matrix of dimensions `R2 × C2`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, U2};
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let row_view = m.row(0);
    ///
    /// // Clone to owned with storage type compatible for addition
    /// let owned = row_view.clone_owned_sum::<U2, U2>();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`clone_owned`](#method.clone_owned) - Simpler version without type-level storage rules
    /// - [`into_owned_sum`](#method.into_owned_sum) - Consuming version
    #[inline]
    #[must_use]
    pub fn clone_owned_sum<R2, C2>(&self) -> MatrixSum<T, R, C, R2, C2>
    where
        T: Scalar,
        S: Storage<T, R, C>,
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();
        let nrows: SameShapeR<R, R2> = Dim::from_usize(nrows);
        let ncols: SameShapeC<C, C2> = Dim::from_usize(ncols);

        let mut res = Matrix::uninit(nrows, ncols);

        unsafe {
            // TODO: use copy_from?
            for j in 0..res.ncols() {
                for i in 0..res.nrows() {
                    *res.get_unchecked_mut((i, j)) =
                        MaybeUninit::new(self.get_unchecked((i, j)).clone());
                }
            }

            // SAFETY: the output has been initialized above.
            res.assume_init()
        }
    }

    /// Transposes `self` and store the result into `out`.
    #[inline]
    fn transpose_to_uninit<Status, R2, C2, SB>(
        &self,
        _status: Status,
        out: &mut Matrix<Status::Value, R2, C2, SB>,
    ) where
        Status: InitStatus<T>,
        T: Scalar,
        R2: Dim,
        C2: Dim,
        SB: RawStorageMut<Status::Value, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transposition."
        );

        // TODO: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                // Safety: the indices are in range.
                unsafe {
                    Status::init(
                        out.get_unchecked_mut((j, i)),
                        self.get_unchecked((i, j)).clone(),
                    );
                }
            }
        }
    }

    /// Transposes `self` and stores the result into the pre-allocated matrix `out`.
    ///
    /// This method is useful when you want to avoid allocating a new matrix for the transpose
    /// operation. The output matrix must have dimensions matching the transpose (i.e., if `self`
    /// is `m × n`, then `out` must be `n × m`).
    ///
    /// # Panics
    ///
    /// Panics if the output matrix dimensions don't match the expected transpose dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// let m = Matrix2x3::new(1, 2, 3,
    ///                        4, 5, 6);
    ///
    /// let mut out = Matrix3x2::zeros();
    /// m.transpose_to(&mut out);
    ///
    /// assert_eq!(out, Matrix3x2::new(1, 4,
    ///                                2, 5,
    ///                                3, 6));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transpose`](#method.transpose) - Allocates and returns a new transposed matrix
    /// - [`transpose_mut`](#method.transpose_mut) - In-place transpose for square matrices
    #[inline]
    pub fn transpose_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        T: Scalar,
        R2: Dim,
        C2: Dim,
        SB: RawStorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        self.transpose_to_uninit(Init, out)
    }

    /// Returns the transpose of this matrix.
    ///
    /// The transpose of a matrix swaps its rows and columns. If the original matrix has
    /// dimensions `m × n`, the transposed matrix will have dimensions `n × m`.
    ///
    /// This method creates a new matrix. For in-place transposition of square matrices,
    /// use [`transpose_mut`](#method.transpose_mut).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(1, 2, 3,
    ///                        4, 5, 6);
    ///
    /// let mt = m.transpose();
    /// assert_eq!(mt.shape(), (3, 2));
    /// assert_eq!(mt[(0, 0)], 1);
    /// assert_eq!(mt[(1, 0)], 2);
    /// assert_eq!(mt[(2, 0)], 3);
    /// assert_eq!(mt[(0, 1)], 4);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Transpose a column vector to a row vector
    /// let col = Vector3::new(1, 2, 3);
    /// let row = col.transpose();
    /// assert_eq!(row.shape(), (1, 3));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transpose_mut`](#method.transpose_mut) - In-place transpose for square matrices
    /// - [`adjoint`](#method.adjoint) - Conjugate transpose (for complex matrices)
    #[inline]
    #[must_use = "Did you mean to use transpose_mut()?"]
    pub fn transpose(&self) -> OMatrix<T, C, R>
    where
        T: Scalar,
        DefaultAllocator: Allocator<C, R>,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut res = Matrix::uninit(ncols, nrows);
        self.transpose_to_uninit(Uninit, &mut res);
        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }
}

/// # Elementwise mapping and folding
impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a new matrix with the function `f` applied to each element.
    ///
    /// This method transforms each element of the matrix using the provided function.
    /// The function receives each element value and returns a potentially different type.
    /// The resulting matrix has the same dimensions as the original.
    ///
    /// This is useful for element-wise transformations such as scaling, absolute values,
    /// type conversions, or any other per-element operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Double each element
    /// let doubled = m.map(|x| x * 2);
    /// assert_eq!(doubled, Matrix2::new(2, 4, 6, 8));
    ///
    /// // Convert to floating point and compute square
    /// let squared = m.map(|x| (x as f64).powi(2));
    /// assert_eq!(squared, Matrix2::new(1.0, 4.0, 9.0, 16.0));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.5, 2.7, -3.2);
    ///
    /// // Apply absolute value to each element
    /// let abs = v.map(|x| x.abs());
    /// assert_eq!(abs, Vector3::new(1.5, 2.7, 3.2));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map_with_location`](#method.map_with_location) - Map with row/column indices
    /// - [`cast`](#method.cast) - Type conversion for matrices
    /// - [`apply`](#method.apply) - In-place element-wise transformation
    #[inline]
    #[must_use]
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, mut f: F) -> OMatrix<T2, R, C>
    where
        T: Scalar,
        DefaultAllocator: Allocator<R, C>,
    {
        let (nrows, ncols) = self.shape_generic();
        let mut res = Matrix::uninit(nrows, ncols);

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                // Safety: all indices are in range.
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(a));
                }
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Converts each element of this matrix to another type.
    ///
    /// This method performs type conversion on each element of the matrix, creating a new
    /// matrix with elements of the target type. The conversion must be valid according to
    /// the `SupersetOf` trait relationship.
    ///
    /// This is commonly used to convert between different numeric types (e.g., `f64` to `f32`,
    /// `i32` to `f64`, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Vector3, Matrix2};
    /// // Convert from f64 to f32
    /// let v = Vector3::new(1.0f64, 2.0, 3.0);
    /// let v_f32 = v.cast::<f32>();
    /// assert_eq!(v_f32, Vector3::new(1.0f32, 2.0, 3.0));
    ///
    /// // Convert from integers to floats
    /// let m = Matrix2::new(1, 2, 3, 4);
    /// let m_float = m.cast::<f64>();
    /// assert_eq!(m_float, Matrix2::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_cast`](#method.try_cast) - Fallible version that returns `Option`
    /// - [`map`](#method.map) - General element-wise transformation
    pub fn cast<T2: Scalar>(self) -> OMatrix<T2, R, C>
    where
        T: Scalar,
        OMatrix<T2, R, C>: SupersetOf<Self>,
        DefaultAllocator: Allocator<R, C>,
    {
        crate::convert(self)
    }

    /// Attempts to convert each element of this matrix to another type, returning `None` on failure.
    ///
    /// This is the fallible version of [`cast`](#method.cast). It returns `Some` if all elements
    /// can be successfully converted, or `None` if any conversion fails. This is useful when
    /// converting from floating-point to integer types where overflow is possible, or when
    /// converting between types with different ranges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Successful conversion from float to int
    /// let v = Vector3::new(1.0f64, 2.0, 3.0);
    /// let v_int = v.try_cast::<i32>();
    /// assert_eq!(v_int, Some(Vector3::new(1, 2, 3)));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Failed conversion due to non-integer values
    /// let v = Vector3::new(1.5f64, 2.7, 3.2);
    /// let v_int = v.try_cast::<i32>();
    /// assert_eq!(v_int, None);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`cast`](#method.cast) - Infallible version (may panic on failure)
    /// - [`map`](#method.map) - General element-wise transformation
    pub fn try_cast<T2: Scalar>(self) -> Option<OMatrix<T2, R, C>>
    where
        T: Scalar,
        Self: SupersetOf<OMatrix<T2, R, C>>,
        DefaultAllocator: Allocator<R, C>,
    {
        crate::try_convert(self)
    }

    /// Folds (reduces) the matrix elements using a closure to determine the initial value.
    ///
    /// This is similar to `self.iter().fold(init, f)` except that the initial value is computed
    /// by a closure rather than being provided directly. This is useful when the initial value
    /// depends on the first element or when you need different behavior for empty matrices.
    ///
    /// The initialization closure is given the first component of this matrix:
    /// - If the matrix has no components (0 rows or 0 columns), `init_f` is called with `None`
    ///   and its return value is the value returned by this method.
    /// - If the matrix has at least one component, `init_f` is called with the first component
    ///   to compute the initial value. Folding then continues on all remaining components.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Use the first element as the initial value for summing
    /// let sum = m.fold_with(
    ///     |first| first.copied().unwrap_or(0),
    ///     |acc, &x| acc + x
    /// );
    /// assert_eq!(sum, 10);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(2.0, 8.0, 4.0);
    ///
    /// // Find maximum value
    /// let max = v.fold_with(
    ///     |first| *first.unwrap(),
    ///     |acc, &x| acc.max(x)
    /// );
    /// assert_eq!(max, 8.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`fold`](#method.fold) - Standard fold with explicit initial value
    /// - [`zip_fold`](#method.zip_fold) - Fold over two matrices simultaneously
    #[inline]
    #[must_use]
    pub fn fold_with<T2>(
        &self,
        init_f: impl FnOnce(Option<&T>) -> T2,
        f: impl FnMut(T2, &T) -> T2,
    ) -> T2
    where
        T: Scalar,
    {
        let mut it = self.iter();
        let init = init_f(it.next());
        it.fold(init, f)
    }

    /// Returns a new matrix with a function applied to each element along with its position.
    ///
    /// This is similar to [`map`](#method.map), but the function also receives the row and column
    /// indices of each element. The signature is `f(row, col, value)`. This is useful when the
    /// transformation depends on the element's position in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Add row and column indices to each element
    /// let result = m.map_with_location(|i, j, val| val + (i + j) as i32);
    /// assert_eq!(result, Matrix2::new(1, 3,
    ///                                 4, 6));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// // Create a matrix where each element is its position encoded as row * 10 + col
    /// let m = Matrix3::from_element(0);
    /// let positions = m.map_with_location(|i, j, _| (i * 10 + j) as i32);
    /// assert_eq!(positions[(1, 2)], 12);
    /// assert_eq!(positions[(2, 1)], 21);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](#method.map) - Map without position information
    /// - [`zip_map`](#method.zip_map) - Map over two matrices simultaneously
    #[inline]
    #[must_use]
    pub fn map_with_location<T2: Scalar, F: FnMut(usize, usize, T) -> T2>(
        &self,
        mut f: F,
    ) -> OMatrix<T2, R, C>
    where
        T: Scalar,
        DefaultAllocator: Allocator<R, C>,
    {
        let (nrows, ncols) = self.shape_generic();
        let mut res = Matrix::uninit(nrows, ncols);

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                // Safety: all indices are in range.
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(i, j, a));
                }
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Returns a new matrix by applying a function to corresponding elements of two matrices.
    ///
    /// This method combines elements from `self` and `rhs` pairwise using the provided function.
    /// The matrices must have the same dimensions. The function signature is `f(self_elem, rhs_elem)`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let b = Matrix2::new(10, 20,
    ///                      30, 40);
    ///
    /// // Add corresponding elements
    /// let sum = a.zip_map(&b, |x, y| x + y);
    /// assert_eq!(sum, Matrix2::new(11, 22, 33, 44));
    ///
    /// // Multiply corresponding elements
    /// let product = a.zip_map(&b, |x, y| x * y);
    /// assert_eq!(product, Matrix2::new(10, 40, 90, 160));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(2.0, 3.0, 4.0);
    ///
    /// // Compute element-wise maximum
    /// let max = v1.zip_map(&v2, |a, b| a.max(b));
    /// assert_eq!(max, Vector3::new(2.0, 3.0, 4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](#method.map) - Map over a single matrix
    /// - [`zip_zip_map`](#method.zip_zip_map) - Map over three matrices
    /// - [`zip_apply`](#method.zip_apply) - In-place version
    #[inline]
    #[must_use]
    pub fn zip_map<T2, N3, S2, F>(&self, rhs: &Matrix<T2, R, C, S2>, mut f: F) -> OMatrix<N3, R, C>
    where
        T: Scalar,
        T2: Scalar,
        N3: Scalar,
        S2: RawStorage<T2, R, C>,
        F: FnMut(T, T2) -> N3,
        DefaultAllocator: Allocator<R, C>,
    {
        let (nrows, ncols) = self.shape_generic();
        let mut res = Matrix::uninit(nrows, ncols);

        assert_eq!(
            (nrows.value(), ncols.value()),
            rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                // Safety: all indices are in range.
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    let b = rhs.data.get_unchecked(i, j).clone();
                    *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(a, b))
                }
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Returns a new matrix by applying a function to corresponding elements of three matrices.
    ///
    /// This method combines elements from `self`, `b`, and `c` using the provided function.
    /// All matrices must have the same dimensions. The function signature is `f(self_elem, b_elem, c_elem)`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let b = Matrix2::new(10, 20,
    ///                      30, 40);
    /// let c = Matrix2::new(100, 200,
    ///                      300, 400);
    ///
    /// // Compute a weighted sum: a + 2*b + 3*c
    /// let result = a.zip_zip_map(&b, &c, |x, y, z| x + 2*y + 3*z);
    /// assert_eq!(result, Matrix2::new(321, 642, 963, 1284));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v1 = Vector3::new(1.0, 5.0, 3.0);
    /// let v2 = Vector3::new(2.0, 4.0, 6.0);
    /// let v3 = Vector3::new(3.0, 2.0, 1.0);
    ///
    /// // Find the middle value (median) among three values
    /// let median = v1.zip_zip_map(&v2, &v3, |a, b, c| {
    ///     let mut arr = [a, b, c];
    ///     arr.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ///     arr[1]
    /// });
    /// assert_eq!(median, Vector3::new(2.0, 4.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](#method.map) - Map over a single matrix
    /// - [`zip_map`](#method.zip_map) - Map over two matrices
    /// - [`zip_zip_apply`](#method.zip_zip_apply) - In-place version
    #[inline]
    #[must_use]
    pub fn zip_zip_map<T2, N3, N4, S2, S3, F>(
        &self,
        b: &Matrix<T2, R, C, S2>,
        c: &Matrix<N3, R, C, S3>,
        mut f: F,
    ) -> OMatrix<N4, R, C>
    where
        T: Scalar,
        T2: Scalar,
        N3: Scalar,
        N4: Scalar,
        S2: RawStorage<T2, R, C>,
        S3: RawStorage<N3, R, C>,
        F: FnMut(T, T2, N3) -> N4,
        DefaultAllocator: Allocator<R, C>,
    {
        let (nrows, ncols) = self.shape_generic();
        let mut res = Matrix::uninit(nrows, ncols);

        assert_eq!(
            (nrows.value(), ncols.value()),
            b.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );
        assert_eq!(
            (nrows.value(), ncols.value()),
            c.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                // Safety: all indices are in range.
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    let b = b.data.get_unchecked(i, j).clone();
                    let c = c.data.get_unchecked(i, j).clone();
                    *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(a, b, c))
                }
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Folds (reduces) all matrix elements into a single value using a function.
    ///
    /// This method iterates through all elements of the matrix in column-major order,
    /// repeatedly applying the function to combine the accumulator with each element.
    /// The function signature is `f(accumulator, element)`.
    ///
    /// This is equivalent to `self.iter().fold(init, f)` but optimized for matrix layout.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// // Sum all elements
    /// let sum = m.fold(0, |acc, x| acc + x);
    /// assert_eq!(sum, 10);
    ///
    /// // Find maximum element
    /// let max = m.fold(i32::MIN, |acc, x| acc.max(x));
    /// assert_eq!(max, 4);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(2.0, 3.0, 4.0);
    ///
    /// // Compute product of all elements
    /// let product = v.fold(1.0, |acc, x| acc * x);
    /// assert_eq!(product, 24.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`fold_with`](#method.fold_with) - Fold with computed initial value
    /// - [`zip_fold`](#method.zip_fold) - Fold over two matrices simultaneously
    #[inline]
    #[must_use]
    pub fn fold<Acc>(&self, init: Acc, mut f: impl FnMut(Acc, T) -> Acc) -> Acc
    where
        T: Scalar,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut res = init;

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                // Safety: all indices are in range.
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    res = f(res, a)
                }
            }
        }

        res
    }

    /// Folds (reduces) corresponding elements from two matrices using a function.
    ///
    /// This method iterates through corresponding elements of `self` and `rhs` in column-major order,
    /// repeatedly applying the function to combine the accumulator with each pair of elements.
    /// The function signature is `f(accumulator, self_elem, rhs_elem)`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let b = Matrix2::new(10, 20,
    ///                      30, 40);
    ///
    /// // Compute sum of products of corresponding elements
    /// let dot_product = a.zip_fold(&b, 0, |acc, x, y| acc + x * y);
    /// assert_eq!(dot_product, 1*10 + 2*20 + 3*30 + 4*40);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(2.0, 3.0, 1.0);
    ///
    /// // Count how many corresponding elements satisfy a condition
    /// let count = v1.zip_fold(&v2, 0, |acc, x, y| {
    ///     if x > y { acc + 1 } else { acc }
    /// });
    /// assert_eq!(count, 1); // Only 3.0 > 1.0
    /// ```
    ///
    /// # See Also
    ///
    /// - [`fold`](#method.fold) - Fold over a single matrix
    /// - [`zip_map`](#method.zip_map) - Map over two matrices
    #[inline]
    #[must_use]
    pub fn zip_fold<T2, R2, C2, S2, Acc>(
        &self,
        rhs: &Matrix<T2, R2, C2, S2>,
        init: Acc,
        mut f: impl FnMut(Acc, T, T2) -> Acc,
    ) -> Acc
    where
        T: Scalar,
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: RawStorage<T2, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut res = init;

        assert_eq!(
            (nrows.value(), ncols.value()),
            rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).clone();
                    let b = rhs.data.get_unchecked(i, j).clone();
                    res = f(res, a, b)
                }
            }
        }

        res
    }

    /// Applies a function to each element of the matrix, modifying it in-place.
    ///
    /// This is the in-place version of [`map`](#method.map). The function receives a mutable
    /// reference to each element and can modify it directly. Elements are processed in
    /// column-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut m = Matrix2::new(1, 2,
    ///                          3, 4);
    ///
    /// // Double each element in-place
    /// m.apply(|x| *x *= 2);
    /// assert_eq!(m, Matrix2::new(2, 4, 6, 8));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::new(-1.0, 2.5, -3.7);
    ///
    /// // Take absolute value of each element
    /// v.apply(|x| *x = x.abs());
    /// assert_eq!(v, Vector3::new(1.0, 2.5, 3.7));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](#method.map) - Non-mutating version that returns a new matrix
    /// - [`apply_into`](#method.apply_into) - Chainable version
    /// - [`zip_apply`](#method.zip_apply) - Apply with elements from another matrix
    #[inline]
    pub fn apply<F: FnMut(&mut T)>(&mut self, mut f: F)
    where
        S: RawStorageMut<T, R, C>,
    {
        let (nrows, ncols) = self.shape();

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    f(e)
                }
            }
        }
    }

    /// Modifies each element in-place using corresponding elements from another matrix.
    ///
    /// This is the in-place version of [`zip_map`](#method.zip_map). The function receives
    /// a mutable reference to each element of `self` and the corresponding element from `rhs`.
    /// The function signature is `f(&mut self_elem, rhs_elem)`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut a = Matrix2::new(1, 2,
    ///                          3, 4);
    /// let b = Matrix2::new(10, 20,
    ///                      30, 40);
    ///
    /// // Add corresponding elements from b to a
    /// a.zip_apply(&b, |x, y| *x += y);
    /// assert_eq!(a, Matrix2::new(11, 22, 33, 44));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(2.0, 3.0, 1.0);
    ///
    /// // Keep maximum of each pair
    /// v1.zip_apply(&v2, |x, y| *x = (*x).max(y));
    /// assert_eq!(v1, Vector3::new(2.0, 3.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`zip_map`](#method.zip_map) - Non-mutating version that returns a new matrix
    /// - [`apply`](#method.apply) - Apply to a single matrix
    /// - [`zip_zip_apply`](#method.zip_zip_apply) - Apply with three matrices
    #[inline]
    pub fn zip_apply<T2, R2, C2, S2>(
        &mut self,
        rhs: &Matrix<T2, R2, C2, S2>,
        mut f: impl FnMut(&mut T, T2),
    ) where
        S: RawStorageMut<T, R, C>,
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: RawStorage<T2, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();

        assert_eq!(
            (nrows, ncols),
            rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    let rhs = rhs.get_unchecked((i, j)).clone();
                    f(e, rhs)
                }
            }
        }
    }

    /// Modifies each element in-place using corresponding elements from two other matrices.
    ///
    /// This is the in-place version of [`zip_zip_map`](#method.zip_zip_map). The function receives
    /// a mutable reference to each element of `self` and the corresponding elements from `b` and `c`.
    /// The function signature is `f(&mut self_elem, b_elem, c_elem)`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut a = Matrix2::new(1, 2,
    ///                          3, 4);
    /// let b = Matrix2::new(10, 20,
    ///                      30, 40);
    /// let c = Matrix2::new(100, 200,
    ///                      300, 400);
    ///
    /// // Compute: a = a + 2*b + 3*c
    /// a.zip_zip_apply(&b, &c, |x, y, z| *x = *x + 2*y + 3*z);
    /// assert_eq!(a, Matrix2::new(321, 642, 963, 1284));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v1 = Vector3::new(1.0, 5.0, 3.0);
    /// let v2 = Vector3::new(2.0, 4.0, 6.0);
    /// let v3 = Vector3::new(3.0, 2.0, 1.0);
    ///
    /// // Find and store the maximum among three values
    /// v1.zip_zip_apply(&v2, &v3, |x, y, z| *x = (*x).max(y).max(z));
    /// assert_eq!(v1, Vector3::new(3.0, 5.0, 6.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`zip_zip_map`](#method.zip_zip_map) - Non-mutating version that returns a new matrix
    /// - [`zip_apply`](#method.zip_apply) - Apply with two matrices
    /// - [`apply`](#method.apply) - Apply to a single matrix
    #[inline]
    pub fn zip_zip_apply<T2, R2, C2, S2, N3, R3, C3, S3>(
        &mut self,
        b: &Matrix<T2, R2, C2, S2>,
        c: &Matrix<N3, R3, C3, S3>,
        mut f: impl FnMut(&mut T, T2, N3),
    ) where
        S: RawStorageMut<T, R, C>,
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: RawStorage<T2, R2, C2>,
        N3: Scalar,
        R3: Dim,
        C3: Dim,
        S3: RawStorage<N3, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();

        assert_eq!(
            (nrows, ncols),
            b.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );
        assert_eq!(
            (nrows, ncols),
            c.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    let b = b.get_unchecked((i, j)).clone();
                    let c = c.get_unchecked((i, j)).clone();
                    f(e, b, c)
                }
            }
        }
    }
}

/// # Iteration on components, rows, and columns
impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns an iterator over all matrix elements in column-major order.
    ///
    /// The iterator yields references to each element, traversing the matrix column by column.
    /// This is the standard iteration order for matrices in nalgebra, optimized for column-major
    /// storage layout.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(11, 12, 13,
    ///                          21, 22, 23);
    ///
    /// // Iterate in column-major order: first column, then second, then third
    /// let mut it = mat.iter();
    /// assert_eq!(*it.next().unwrap(), 11);
    /// assert_eq!(*it.next().unwrap(), 21);
    /// assert_eq!(*it.next().unwrap(), 12);
    /// assert_eq!(*it.next().unwrap(), 22);
    /// assert_eq!(*it.next().unwrap(), 13);
    /// assert_eq!(*it.next().unwrap(), 23);
    /// assert!(it.next().is_none());
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(1, 2, 3);
    /// let sum: i32 = v.iter().sum();
    /// assert_eq!(sum, 6);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`iter_mut`](#method.iter_mut) - Mutable iterator
    /// - [`column_iter`](#method.column_iter) - Iterate over columns
    /// - [`row_iter`](#method.row_iter) - Iterate over rows
    #[inline]
    pub fn iter(&self) -> MatrixIter<'_, T, R, C, S> {
        MatrixIter::new(&self.data)
    }

    /// Iterate through the rows of this matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, row) in a.row_iter().enumerate() {
    ///     assert_eq!(row, a.row(i))
    /// }
    /// ```
    #[inline]
    pub const fn row_iter(&self) -> RowIter<'_, T, R, C, S> {
        RowIter::new(self)
    }

    /// Returns an iterator over the columns of the matrix.
    ///
    /// Each iteration yields a column view, allowing you to process the matrix one column at a time.
    /// This is efficient for column-major matrices as columns are stored contiguously in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, column) in a.column_iter().enumerate() {
    ///     assert_eq!(column, a.column(i))
    /// }
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::identity();
    /// let col_sums: Vec<f64> = m.column_iter()
    ///     .map(|col| col.iter().sum())
    ///     .collect();
    /// assert_eq!(col_sums, vec![1.0, 1.0, 1.0]);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`column_iter_mut`](#method.column_iter_mut) - Mutable column iterator
    /// - [`row_iter`](#method.row_iter) - Iterate over rows
    /// - [`iter`](#method.iter) - Iterate over individual elements
    #[inline]
    pub fn column_iter(&self) -> ColumnIter<'_, T, R, C, S> {
        ColumnIter::new(self)
    }

    /// Returns a mutable iterator over all matrix elements in column-major order.
    ///
    /// The iterator yields mutable references to each element, allowing in-place modifications
    /// while traversing the matrix column by column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    ///
    /// // Double each element
    /// for elem in m.iter_mut() {
    ///     *elem *= 2;
    /// }
    /// assert_eq!(m, Matrix2x3::new(2, 4, 6, 8, 10, 12));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::new(1.0, 2.0, 3.0);
    /// v.iter_mut().for_each(|x| *x = x.sqrt());
    /// assert_eq!(v, Vector3::new(1.0, 2.0f64.sqrt(), 3.0f64.sqrt()));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`iter`](#method.iter) - Immutable iterator
    /// - [`column_iter_mut`](#method.column_iter_mut) - Mutable iterator over columns
    /// - [`apply`](#method.apply) - Apply a function to each element in-place
    #[inline]
    pub fn iter_mut(&mut self) -> MatrixIterMut<'_, T, R, C, S>
    where
        S: RawStorageMut<T, R, C>,
    {
        MatrixIterMut::new(&mut self.data)
    }

    /// Mutably iterates through this matrix rows.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, mut row) in a.row_iter_mut().enumerate() {
    ///     row *= (i + 1) * 10;
    /// }
    ///
    /// let expected = Matrix2x3::new(10, 20, 30,
    ///                               80, 100, 120);
    /// assert_eq!(a, expected);
    /// ```
    #[inline]
    pub const fn row_iter_mut(&mut self) -> RowIterMut<'_, T, R, C, S>
    where
        S: RawStorageMut<T, R, C>,
    {
        RowIterMut::new(self)
    }

    /// Returns a mutable iterator over the columns of the matrix.
    ///
    /// Each iteration yields a mutable column view, allowing you to modify the matrix
    /// one column at a time. This is efficient for column-major matrices as columns
    /// are stored contiguously in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, mut col) in a.column_iter_mut().enumerate() {
    ///     col *= (i + 1) * 10;
    /// }
    ///
    /// let expected = Matrix2x3::new(10, 40, 90,
    ///                               40, 100, 180);
    /// assert_eq!(a, expected);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// // Set each column to increasing values
    /// for (i, mut col) in m.column_iter_mut().enumerate() {
    ///     col.fill((i + 1) as f64);
    /// }
    /// assert_eq!(m.column(0).sum(), 3.0);
    /// assert_eq!(m.column(1).sum(), 6.0);
    /// assert_eq!(m.column(2).sum(), 9.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`column_iter`](#method.column_iter) - Immutable column iterator
    /// - [`row_iter_mut`](#method.row_iter_mut) - Mutable row iterator
    /// - [`iter_mut`](#method.iter_mut) - Mutable iterator over individual elements
    #[inline]
    pub fn column_iter_mut(&mut self) -> ColumnIterMut<'_, T, R, C, S>
    where
        S: RawStorageMut<T, R, C>,
    {
        ColumnIterMut::new(self)
    }
}

impl<T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a mutable raw pointer to the first element of the matrix.
    ///
    /// This provides direct mutable access to the underlying memory. The matrix data is stored
    /// in column-major order, meaning elements within each column are contiguous in memory.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be properly aligned and non-null.
    ///
    /// # Safety
    ///
    /// While this method itself is safe, dereferencing the returned pointer requires `unsafe` code.
    /// The caller must ensure:
    /// - The pointer is only used while the matrix is valid and hasn't been moved
    /// - Proper bounds are maintained (use [`len`](#method.len) to check the element count)
    /// - No other references (mutable or immutable) to the matrix data exist while using this pointer
    /// - Modifications through the pointer maintain the matrix's invariants
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut m = Matrix2::new(1, 2,
    ///                          3, 4);
    /// let ptr = m.as_mut_ptr();
    ///
    /// // Modify the first element (stored in column-major order)
    /// unsafe {
    ///     *ptr = 10;
    /// }
    ///
    /// assert_eq!(m[(0, 0)], 10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`as_ptr`](#method.as_ptr) - Returns an immutable pointer
    /// - [`as_mut_slice`](#method.as_mut_slice) - Returns a safe mutable slice for contiguous matrices
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.ptr_mut()
    }

    /// Swaps two entries without bound-checking.
    ///
    /// # Safety
    ///
    /// Both `(r, c)` must have `r < nrows(), c < ncols()`.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        unsafe {
            debug_assert!(row_cols1.0 < self.nrows() && row_cols1.1 < self.ncols());
            debug_assert!(row_cols2.0 < self.nrows() && row_cols2.1 < self.ncols());
            self.data.swap_unchecked(row_cols1, row_cols2)
        }
    }

    /// Swaps two elements in the matrix.
    ///
    /// The elements are specified by their `(row, column)` coordinates. This method
    /// can swap any two elements, even if they're in the same row or column.
    ///
    /// # Panics
    ///
    /// Panics if either coordinate pair is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut m = Matrix2::new(1, 2,
    ///                          3, 4);
    ///
    /// // Swap elements at (0, 0) and (1, 1)
    /// m.swap((0, 0), (1, 1));
    /// assert_eq!(m, Matrix2::new(4, 2,
    ///                            3, 1));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::new(1, 2, 3);
    /// v.swap((0, 0), (2, 0));
    /// assert_eq!(v, Vector3::new(3, 2, 1));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`swap_rows`](#method.swap_rows) - Swap entire rows
    /// - [`swap_columns`](#method.swap_columns) - Swap entire columns
    #[inline]
    pub fn swap(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        let (nrows, ncols) = self.shape();
        assert!(
            row_cols1.0 < nrows && row_cols1.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        assert!(
            row_cols2.0 < nrows && row_cols2.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        unsafe { self.swap_unchecked(row_cols1, row_cols2) }
    }

    /// Copies elements from a slice into this matrix.
    ///
    /// The slice must contain exactly as many elements as the matrix (nrows × ncols).
    /// Elements from the slice are copied in column-major order: the first `nrows` elements
    /// fill the first column, the next `nrows` elements fill the second column, and so on.
    ///
    /// # Panics
    ///
    /// Panics if the slice length doesn't match the matrix size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::zeros();
    /// let data = [1, 4, 2, 5, 3, 6];  // Column-major order
    ///
    /// m.copy_from_slice(&data);
    /// assert_eq!(m, Matrix2x3::new(1, 2, 3,
    ///                              4, 5, 6));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::zeros();
    /// v.copy_from_slice(&[10, 20, 30]);
    /// assert_eq!(v, Vector3::new(10, 20, 30));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`copy_from`](#method.copy_from) - Copy from another matrix
    /// - [`from_row_slice`](#method.from_row_slice) - Create a matrix from row-major data
    /// - [`from_column_slice`](#method.from_column_slice) - Create a matrix from column-major data
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Scalar,
    {
        let (nrows, ncols) = self.shape();

        assert!(
            nrows * ncols == slice.len(),
            "The slice must contain the same number of elements as the matrix."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = slice.get_unchecked(i + j * nrows).clone();
                }
            }
        }
    }

    /// Copies all elements from another matrix into this one.
    ///
    /// Both matrices must have the same dimensions. This is useful for copying data between
    /// matrices with different storage types (e.g., from a dynamic matrix to a static one,
    /// or from a matrix view to an owned matrix).
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, DMatrix};
    /// let source = Matrix2::new(1, 2,
    ///                           3, 4);
    /// let mut dest = Matrix2::zeros();
    ///
    /// dest.copy_from(&source);
    /// assert_eq!(dest, source);
    /// ```
    ///
    /// ```
    /// # use nalgebra::{Matrix3, DMatrix};
    /// // Copy from dynamic to static
    /// let dyn_mat = DMatrix::from_row_slice(2, 2, &[1, 2, 3, 4]);
    /// let mut static_mat = Matrix3::zeros();
    /// static_mat.view_mut((0, 0), (2, 2)).copy_from(&dyn_mat);
    /// assert_eq!(static_mat[(0, 0)], 1);
    /// assert_eq!(static_mat[(1, 1)], 4);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`copy_from_slice`](#method.copy_from_slice) - Copy from a slice
    /// - [`tr_copy_from`](#method.tr_copy_from) - Copy the transpose of another matrix
    /// - [`clone_from`](#method.clone_from) - Standard library clone method
    #[inline]
    pub fn copy_from<R2, C2, SB>(&mut self, other: &Matrix<T, R2, C2, SB>)
    where
        T: Scalar,
        R2: Dim,
        C2: Dim,
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(
            self.shape() == other.shape(),
            "Unable to copy from a matrix with a different shape."
        );

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((i, j)).clone();
                }
            }
        }
    }

    /// Copies the transpose of another matrix into this one.
    ///
    /// If `other` has dimensions `m × n`, then `self` must have dimensions `n × m`.
    /// This is equivalent to `self.copy_from(&other.transpose())` but more efficient
    /// as it avoids allocating a temporary transposed matrix.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible (i.e., if `self.shape()` ≠ `(other.ncols(), other.nrows())`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// let source = Matrix2x3::new(1, 2, 3,
    ///                             4, 5, 6);
    /// let mut dest = Matrix3x2::zeros();
    ///
    /// dest.tr_copy_from(&source);
    /// assert_eq!(dest, Matrix3x2::new(1, 4,
    ///                                 2, 5,
    ///                                 3, 6));
    /// ```
    ///
    /// ```
    /// # use nalgebra::{Matrix2, RowVector2, Vector2};
    /// let col = Vector2::new(1, 2);
    /// let mut row = RowVector2::zeros();
    ///
    /// row.tr_copy_from(&col);
    /// assert_eq!(row, RowVector2::new(1, 2));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`copy_from`](#method.copy_from) - Copy without transposing
    /// - [`transpose_to`](#method.transpose_to) - Transpose into a pre-allocated matrix
    #[inline]
    pub fn tr_copy_from<R2, C2, SB>(&mut self, other: &Matrix<T, R2, C2, SB>)
    where
        T: Scalar,
        R2: Dim,
        C2: Dim,
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: DimEq<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == other.shape(),
            "Unable to copy from a matrix with incompatible shape."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((j, i)).clone();
                }
            }
        }
    }

    // TODO: rename `apply` to `apply_mut` and `apply_into` to `apply`?
    /// Applies a function to each element and returns the modified matrix.
    ///
    /// This is a chainable version of [`apply`](#method.apply). It consumes `self`, applies
    /// the function to each element in-place, and returns the modified matrix. This is useful
    /// for method chaining and functional-style programming.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    ///
    /// let doubled = m.apply_into(|x| *x *= 2);
    /// assert_eq!(doubled, Matrix2::new(2, 4, 6, 8));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Chain multiple operations
    /// let v = Vector3::new(1.0, 4.0, 9.0)
    ///     .apply_into(|x| *x = x.sqrt())
    ///     .apply_into(|x| *x *= 2.0);
    /// assert_eq!(v, Vector3::new(2.0, 4.0, 6.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`apply`](#method.apply) - Non-chainable in-place version
    /// - [`map`](#method.map) - Creates a new matrix instead of modifying
    #[inline]
    pub fn apply_into<F: FnMut(&mut T)>(mut self, f: F) -> Self {
        self.apply(f);
        self
    }
}

impl<T, D: Dim, S: RawStorage<T, D>> Vector<T, D, S> {
    /// Gets a reference to the i-th element of this column vector without bound checking.
    /// # Safety
    /// `i` must be less than `D`.
    #[inline]
    #[must_use]
    pub unsafe fn vget_unchecked(&self, i: usize) -> &T {
        unsafe {
            debug_assert!(i < self.nrows(), "Vector index out of bounds.");
            let i = i * self.strides().0;
            self.data.get_unchecked_linear(i)
        }
    }
}

impl<T, D: Dim, S: RawStorageMut<T, D>> Vector<T, D, S> {
    /// Gets a mutable reference to the i-th element of this column vector without bound checking.
    /// # Safety
    /// `i` must be less than `D`.
    #[inline]
    #[must_use]
    pub unsafe fn vget_unchecked_mut(&mut self, i: usize) -> &mut T {
        unsafe {
            debug_assert!(i < self.nrows(), "Vector index out of bounds.");
            let i = i * self.strides().0;
            self.data.get_unchecked_linear_mut(i)
        }
    }
}

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C> + IsContiguous> Matrix<T, R, C, S> {
    /// Returns a slice containing all matrix elements in column-major order.
    ///
    /// This method provides safe, direct access to the underlying matrix data as a slice.
    /// Elements are arranged in column-major order: all elements from the first column,
    /// followed by all elements from the second column, and so on.
    ///
    /// This method is only available for matrices with contiguous storage (indicated by the
    /// `IsContiguous` trait bound). Most standard matrix types like `Matrix`, `DMatrix`,
    /// and `Vector` satisfy this requirement, but some matrix views may not.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(1, 2, 3,
    ///                        4, 5, 6);
    ///
    /// // Elements are in column-major order: column 1, then column 2, then column 3
    /// let slice = m.as_slice();
    /// assert_eq!(slice, &[1, 4, 2, 5, 3, 6]);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(10, 20, 30);
    /// assert_eq!(v.as_slice(), &[10, 20, 30]);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`as_mut_slice`](#method.as_mut_slice) - Returns a mutable slice
    /// - [`as_ptr`](#method.as_ptr) - Returns a raw pointer for unsafe access
    /// - [`iter`](#method.iter) - Iterate over elements
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // Safety: this is OK thanks to the IsContiguous trait.
        unsafe { self.data.as_slice_unchecked() }
    }
}

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C> + IsContiguous> AsRef<[T]> for Matrix<T, R, C, S> {
    /// Extracts a slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, R: Dim, C: Dim, S: RawStorageMut<T, R, C> + IsContiguous> Matrix<T, R, C, S> {
    /// Returns a mutable slice containing all matrix elements in column-major order.
    ///
    /// This method provides safe, direct mutable access to the underlying matrix data as a slice.
    /// Elements are arranged in column-major order: all elements from the first column,
    /// followed by all elements from the second column, and so on.
    ///
    /// This method is only available for matrices with contiguous storage (indicated by the
    /// `IsContiguous` trait bound). Most standard matrix types like `Matrix`, `DMatrix`,
    /// and `Vector` satisfy this requirement, but some matrix views may not.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    ///
    /// // Modify elements through the slice (in column-major order)
    /// let slice = m.as_mut_slice();
    /// slice[0] = 10;  // Modifies m[(0, 0)]
    /// slice[1] = 40;  // Modifies m[(1, 0)]
    ///
    /// assert_eq!(m[(0, 0)], 10);
    /// assert_eq!(m[(1, 0)], 40);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::new(1, 2, 3);
    /// v.as_mut_slice().iter_mut().for_each(|x| *x *= 2);
    /// assert_eq!(v, Vector3::new(2, 4, 6));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`as_slice`](#method.as_slice) - Returns an immutable slice
    /// - [`as_mut_ptr`](#method.as_mut_ptr) - Returns a raw mutable pointer for unsafe access
    /// - [`iter_mut`](#method.iter_mut) - Iterate mutably over elements
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safety: this is OK thanks to the IsContiguous trait.
        unsafe { self.data.as_mut_slice_unchecked() }
    }
}

impl<T: Scalar, D: Dim, S: RawStorageMut<T, D, D>> Matrix<T, D, D, S> {
    /// Transposes a square matrix in-place.
    ///
    /// This method swaps rows and columns of a square matrix without allocating new memory.
    /// It's more efficient than `transpose()` when you don't need to keep the original matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(1, 2, 3,
    ///                          4, 5, 6,
    ///                          7, 8, 9);
    ///
    /// m.transpose_mut();
    /// assert_eq!(m, Matrix3::new(1, 4, 7,
    ///                            2, 5, 8,
    ///                            3, 6, 9));
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut m = Matrix2::new(1.0, 2.0,
    ///                          3.0, 4.0);
    /// m.transpose_mut();
    /// assert_eq!(m[(0, 1)], 3.0);
    /// assert_eq!(m[(1, 0)], 2.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transpose`](#method.transpose) - Returns a new transposed matrix
    /// - [`adjoint_mut`](#method.adjoint_mut) - In-place conjugate transpose
    pub fn transpose_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 1..dim {
            for j in 0..i {
                unsafe { self.swap_unchecked((i, j), (j, i)) }
            }
        }
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Takes the adjoint (aka. conjugate-transpose) of `self` and store the result into `out`.
    #[inline]
    fn adjoint_to_uninit<Status, R2, C2, SB>(
        &self,
        _status: Status,
        out: &mut Matrix<Status::Value, R2, C2, SB>,
    ) where
        Status: InitStatus<T>,
        R2: Dim,
        C2: Dim,
        SB: RawStorageMut<Status::Value, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transpose-copy."
        );

        // TODO: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                // Safety: all indices are in range.
                unsafe {
                    Status::init(
                        out.get_unchecked_mut((j, i)),
                        self.get_unchecked((i, j)).clone().simd_conjugate(),
                    );
                }
            }
        }
    }

    /// Computes the adjoint (conjugate transpose) and stores it into a pre-allocated matrix.
    ///
    /// This method is useful when you want to avoid allocating a new matrix for the adjoint
    /// operation. The output matrix must have dimensions matching the adjoint (i.e., if `self`
    /// is `m × n`, then `out` must be `n × m`).
    ///
    /// # Panics
    ///
    /// Panics if the output matrix dimensions don't match the expected adjoint dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let m = Matrix2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    /// );
    ///
    /// let mut out = Matrix2::zeros();
    /// m.adjoint_to(&mut out);
    ///
    /// // Adjoint: transposed and conjugated
    /// assert_eq!(out[(0, 0)], Complex::new(1.0, -2.0));
    /// assert_eq!(out[(1, 0)], Complex::new(3.0, -4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`adjoint`](#method.adjoint) - Allocates and returns a new adjoint matrix
    /// - [`transpose_to`](#method.transpose_to) - Transpose without conjugation
    #[inline]
    pub fn adjoint_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: RawStorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        self.adjoint_to_uninit(Init, out)
    }

    /// Returns the adjoint (conjugate transpose) of this matrix.
    ///
    /// The adjoint of a matrix is obtained by taking the transpose and then taking the
    /// complex conjugate of each element. For real-valued matrices, this is equivalent to
    /// the transpose. For complex matrices, each element is conjugated (the sign of the
    /// imaginary part is flipped).
    ///
    /// If the original matrix has dimensions `m × n`, the adjoint will have dimensions `n × m`.
    ///
    /// This operation is commonly used in quantum mechanics, signal processing, and when
    /// working with Hermitian matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// // Real matrix: adjoint is the same as transpose
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// let adj = m.adjoint();
    /// assert_eq!(adj[(0, 0)], 1.0);
    /// assert_eq!(adj[(1, 0)], 2.0);
    /// assert_eq!(adj[(0, 1)], 3.0);
    /// ```
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// // Complex matrix: transpose and conjugate each element
    /// let m = Matrix2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    /// );
    /// let adj = m.adjoint();
    /// assert_eq!(adj[(0, 0)], Complex::new(1.0, -2.0));  // Conjugated
    /// assert_eq!(adj[(1, 0)], Complex::new(3.0, -4.0));  // Conjugated and transposed
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transpose`](#method.transpose) - Transpose without conjugation
    /// - [`conjugate`](#method.conjugate) - Conjugate without transposition
    /// - [`adjoint_mut`](#method.adjoint_mut) - In-place adjoint for square matrices
    #[inline]
    #[must_use = "Did you mean to use adjoint_mut()?"]
    pub fn adjoint(&self) -> OMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<C, R>,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut res = Matrix::uninit(ncols, nrows);
        self.adjoint_to_uninit(Uninit, &mut res);

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Takes the conjugate and transposes `self` and store the result into `out`.
    #[deprecated(note = "Renamed `self.adjoint_to(out)`.")]
    #[inline]
    pub fn conjugate_transpose_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: RawStorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        self.adjoint_to(out)
    }

    /// The conjugate transposition of `self`.
    #[deprecated(note = "Renamed `self.adjoint()`.")]
    #[inline]
    pub fn conjugate_transpose(&self) -> OMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<C, R>,
    {
        self.adjoint()
    }

    /// Returns the complex conjugate of this matrix.
    ///
    /// The conjugate of a matrix is obtained by taking the complex conjugate of each element.
    /// For complex numbers, this flips the sign of the imaginary part. For real numbers,
    /// the conjugate is the same as the original number.
    ///
    /// The matrix dimensions remain unchanged (unlike `transpose` or `adjoint`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// // Real matrix: conjugate returns a copy
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// let conj = m.conjugate();
    /// assert_eq!(conj, m);
    /// ```
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// // Complex vector: flip the sign of imaginary parts
    /// let v = Vector3::new(
    ///     Complex::new(1.0, 2.0),
    ///     Complex::new(3.0, -4.0),
    ///     Complex::new(5.0, 0.0)
    /// );
    /// let conj = v.conjugate();
    /// assert_eq!(conj[0], Complex::new(1.0, -2.0));
    /// assert_eq!(conj[1], Complex::new(3.0, 4.0));
    /// assert_eq!(conj[2], Complex::new(5.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`adjoint`](#method.adjoint) - Conjugate transpose
    /// - [`conjugate_mut`](#method.conjugate_mut) - In-place conjugation
    /// - [`transpose`](#method.transpose) - Transpose without conjugation
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        self.map(|e| e.simd_conjugate())
    }

    /// Divides each element of the matrix by a real scalar.
    ///
    /// This method is specifically for complex matrices, where each complex element
    /// is divided by the given real number. For real matrices, this is equivalent to
    /// regular scalar division.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let m = Matrix2::new(
    ///     Complex::new(2.0, 4.0), Complex::new(6.0, 8.0),
    ///     Complex::new(10.0, 12.0), Complex::new(14.0, 16.0)
    /// );
    ///
    /// let result = m.unscale(2.0);
    /// assert_eq!(result[(0, 0)], Complex::new(1.0, 2.0));
    /// assert_eq!(result[(1, 1)], Complex::new(7.0, 8.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`scale`](#method.scale) - Multiply by a scalar
    /// - [`unscale_mut`](#method.unscale_mut) - In-place version
    #[inline]
    #[must_use = "Did you mean to use unscale_mut()?"]
    pub fn unscale(&self, real: T::SimdRealField) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        self.map(|e| e.simd_unscale(real.clone()))
    }

    /// Multiplies each element of the matrix by a real scalar.
    ///
    /// This method is specifically for complex matrices, where each complex element
    /// is multiplied by the given real number. For real matrices, this is equivalent to
    /// regular scalar multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let m = Matrix2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    /// );
    ///
    /// let result = m.scale(2.0);
    /// assert_eq!(result[(0, 0)], Complex::new(2.0, 4.0));
    /// assert_eq!(result[(1, 1)], Complex::new(14.0, 16.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unscale`](#method.unscale) - Divide by a scalar
    /// - [`scale_mut`](#method.scale_mut) - In-place version
    #[inline]
    #[must_use = "Did you mean to use scale_mut()?"]
    pub fn scale(&self, real: T::SimdRealField) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        self.map(|e| e.simd_scale(real.clone()))
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Conjugates the matrix in-place.
    ///
    /// For complex matrices, this flips the sign of the imaginary part of each element.
    /// For real matrices, this has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let mut v = Vector3::new(
    ///     Complex::new(1.0, 2.0),
    ///     Complex::new(3.0, -4.0),
    ///     Complex::new(5.0, 0.0)
    /// );
    ///
    /// v.conjugate_mut();
    /// assert_eq!(v[0], Complex::new(1.0, -2.0));
    /// assert_eq!(v[1], Complex::new(3.0, 4.0));
    /// assert_eq!(v[2], Complex::new(5.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`conjugate`](#method.conjugate) - Non-mutating version
    /// - [`adjoint_mut`](#method.adjoint_mut) - In-place conjugate transpose
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.apply(|e| *e = e.clone().simd_conjugate())
    }

    /// Divides each element by a real scalar in-place.
    ///
    /// This is the in-place version of [`unscale`](#method.unscale).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let mut m = Matrix2::new(
    ///     Complex::new(2.0, 4.0), Complex::new(6.0, 8.0),
    ///     Complex::new(10.0, 12.0), Complex::new(14.0, 16.0)
    /// );
    ///
    /// m.unscale_mut(2.0);
    /// assert_eq!(m[(0, 0)], Complex::new(1.0, 2.0));
    /// assert_eq!(m[(1, 1)], Complex::new(7.0, 8.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unscale`](#method.unscale) - Non-mutating version
    /// - [`scale_mut`](#method.scale_mut) - Multiply by a scalar
    #[inline]
    pub fn unscale_mut(&mut self, real: T::SimdRealField) {
        self.apply(|e| *e = e.clone().simd_unscale(real.clone()))
    }

    /// Multiplies each element by a real scalar in-place.
    ///
    /// This is the in-place version of [`scale`](#method.scale).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let mut m = Matrix2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    /// );
    ///
    /// m.scale_mut(2.0);
    /// assert_eq!(m[(0, 0)], Complex::new(2.0, 4.0));
    /// assert_eq!(m[(1, 1)], Complex::new(14.0, 16.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`scale`](#method.scale) - Non-mutating version
    /// - [`unscale_mut`](#method.unscale_mut) - Divide by a scalar
    #[inline]
    pub fn scale_mut(&mut self, real: T::SimdRealField) {
        self.apply(|e| *e = e.clone().simd_scale(real.clone()))
    }
}

impl<T: SimdComplexField, D: Dim, S: RawStorageMut<T, D, D>> Matrix<T, D, D, S> {
    /// Sets `self` to its adjoint.
    #[deprecated(note = "Renamed to `self.adjoint_mut()`.")]
    pub fn conjugate_transform_mut(&mut self) {
        self.adjoint_mut()
    }

    /// Computes the adjoint (conjugate transpose) of a square matrix in-place.
    ///
    /// This method transposes the matrix and conjugates each element, without allocating
    /// new memory. It combines the effects of [`transpose_mut`](#method.transpose_mut)
    /// and [`conjugate_mut`](#method.conjugate_mut).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let mut m = Matrix2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    /// );
    ///
    /// m.adjoint_mut();
    ///
    /// // Transposed and conjugated
    /// assert_eq!(m[(0, 0)], Complex::new(1.0, -2.0));
    /// assert_eq!(m[(0, 1)], Complex::new(5.0, -6.0));
    /// assert_eq!(m[(1, 0)], Complex::new(3.0, -4.0));
    /// assert_eq!(m[(1, 1)], Complex::new(7.0, -8.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`adjoint`](#method.adjoint) - Returns a new adjoint matrix
    /// - [`transpose_mut`](#method.transpose_mut) - In-place transpose without conjugation
    /// - [`conjugate_mut`](#method.conjugate_mut) - In-place conjugation without transpose
    pub fn adjoint_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 0..dim {
            for j in 0..i {
                unsafe {
                    let ref_ij = self.get_unchecked((i, j)).clone();
                    let ref_ji = self.get_unchecked((j, i)).clone();
                    let conj_ij = ref_ij.simd_conjugate();
                    let conj_ji = ref_ji.simd_conjugate();
                    *self.get_unchecked_mut((i, j)) = conj_ji;
                    *self.get_unchecked_mut((j, i)) = conj_ij;
                }
            }

            {
                let diag = unsafe { self.get_unchecked_mut((i, i)) };
                *diag = diag.clone().simd_conjugate();
            }
        }
    }
}

impl<T: Scalar, D: Dim, S: RawStorage<T, D, D>> SquareMatrix<T, D, S> {
    /// Extracts the diagonal of the matrix as a column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let m = Matrix3::new(1, 2, 3,
    ///                      4, 5, 6,
    ///                      7, 8, 9);
    ///
    /// let diag = m.diagonal();
    /// assert_eq!(diag, Vector3::new(1, 5, 9));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map_diagonal`](#method.map_diagonal) - Transform diagonal elements
    /// - [`trace`](#method.trace) - Sum of diagonal elements
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        self.map_diagonal(|e| e)
    }

    /// Applies a function to each diagonal element and returns the results as a vector.
    ///
    /// This is more efficient than `self.diagonal().map(f)` because it allocates only once.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let m = Matrix3::new(1, 2, 3,
    ///                      4, 5, 6,
    ///                      7, 8, 9);
    ///
    /// let diag_squared = m.map_diagonal(|x| x * x);
    /// assert_eq!(diag_squared, Vector3::new(1, 25, 81));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`diagonal`](#method.diagonal) - Extract diagonal without transformation
    /// - [`trace`](#method.trace) - Sum of diagonal elements
    #[must_use]
    pub fn map_diagonal<T2: Scalar>(&self, mut f: impl FnMut(T) -> T2) -> OVector<T2, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        assert!(
            self.is_square(),
            "Unable to get the diagonal of a non-square matrix."
        );

        let dim = self.shape_generic().0;
        let mut res = Matrix::uninit(dim, Const::<1>);

        for i in 0..dim.value() {
            // Safety: all indices are in range.
            unsafe {
                *res.vget_unchecked_mut(i) =
                    MaybeUninit::new(f(self.get_unchecked((i, i)).clone()));
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Computes the trace of a square matrix (sum of diagonal elements).
    ///
    /// The trace is defined as the sum of all elements on the main diagonal.
    /// This is an important quantity in linear algebra and appears in many
    /// formulas and theorems.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::new(1, 2, 3,
    ///                      4, 5, 6,
    ///                      7, 8, 9);
    ///
    /// assert_eq!(m.trace(), 1 + 5 + 9);
    /// ```
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let identity = Matrix2::<f64>::identity();
    /// assert_eq!(identity.trace(), 2.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`diagonal`](#method.diagonal) - Extract the diagonal as a vector
    #[inline]
    #[must_use]
    pub fn trace(&self) -> T
    where
        T: Scalar + Zero + ClosedAddAssign,
    {
        assert!(
            self.is_square(),
            "Cannot compute the trace of non-square matrix."
        );

        let dim = self.shape_generic().0;
        let mut res = T::zero();

        for i in 0..dim.value() {
            res += unsafe { self.get_unchecked((i, i)).clone() };
        }

        res
    }
}

impl<T: SimdComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Computes the symmetric part of the matrix: `0.5 * (self + self.transpose())`.
    ///
    /// The symmetric part is the average of the matrix and its transpose. The result
    /// is always a symmetric matrix. This is useful in mechanics and physics where
    /// symmetric tensors represent physical quantities.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    ///
    /// let sym = m.symmetric_part();
    /// // Result: (m + m^T) / 2
    /// assert_eq!(sym, Matrix2::new(1.0, 2.5,
    ///                              2.5, 4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`hermitian_part`](#method.hermitian_part) - For complex matrices
    /// - [`transpose`](#method.transpose) - Matrix transpose
    #[inline]
    #[must_use]
    pub fn symmetric_part(&self) -> OMatrix<T, D, D>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        assert!(
            self.is_square(),
            "Cannot compute the symmetric part of a non-square matrix."
        );
        let mut tr = self.transpose();
        tr += self;
        tr *= crate::convert::<_, T>(0.5);
        tr
    }

    /// Computes the Hermitian part of the matrix: `0.5 * (self + self.adjoint())`.
    ///
    /// The Hermitian part is the average of the matrix and its adjoint (conjugate transpose).
    /// The result is always a Hermitian matrix. For real matrices, this is equivalent to
    /// [`symmetric_part`](#method.symmetric_part).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let m = Matrix2::new(
    ///     Complex::new(1.0, 0.0), Complex::new(2.0, 3.0),
    ///     Complex::new(4.0, 5.0), Complex::new(6.0, 0.0)
    /// );
    ///
    /// let herm = m.hermitian_part();
    /// // Result is Hermitian: h[i,j] = conj(h[j,i])
    /// assert_eq!(herm[(0, 0)].im, 0.0); // Diagonal is real
    /// assert_eq!(herm[(1, 1)].im, 0.0); // Diagonal is real
    /// ```
    ///
    /// # See Also
    ///
    /// - [`symmetric_part`](#method.symmetric_part) - For real matrices
    /// - [`adjoint`](#method.adjoint) - Conjugate transpose
    #[inline]
    #[must_use]
    pub fn hermitian_part(&self) -> OMatrix<T, D, D>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        assert!(
            self.is_square(),
            "Cannot compute the hermitian part of a non-square matrix."
        );

        let mut tr = self.adjoint();
        tr += self;
        tr *= crate::convert::<_, T>(0.5);
        tr
    }
}

impl<T: Scalar + Zero + One, D: DimAdd<U1> + IsNotStaticOne, S: RawStorage<T, D, D>>
    Matrix<T, D, D, S>
{
    /// Yields the homogeneous matrix for this matrix, i.e., appending an additional dimension and
    /// and setting the diagonal element to `1`.
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimSum<D, U1>, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<DimSum<D, U1>, DimSum<D, U1>>,
    {
        assert!(
            self.is_square(),
            "Only square matrices can currently be transformed to homogeneous coordinates."
        );
        let dim = DimSum::<D, U1>::from_usize(self.nrows() + 1);
        let mut res = OMatrix::identity_generic(dim, dim);
        res.generic_view_mut::<D, D>((0, 0), self.shape_generic())
            .copy_from(self);
        res
    }
}

impl<T: Scalar + Zero, D: DimAdd<U1>, S: RawStorage<T, D>> Vector<T, D, S> {
    /// Computes the coordinates in projective space of this vector, i.e., appends a `0` to its
    /// coordinates.
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OVector<T, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<DimSum<D, U1>>,
    {
        self.push(T::zero())
    }

    /// Constructs a vector from coordinates in projective space, i.e., removes a `0` at the end of
    /// `self`. Returns `None` if this last component is not zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: Vector<T, DimSum<D, U1>, SB>) -> Option<OVector<T, D>>
    where
        SB: RawStorage<T, DimSum<D, U1>>,
        DefaultAllocator: Allocator<D>,
    {
        if v[v.len() - 1].is_zero() {
            let nrows = D::from_usize(v.len() - 1);
            Some(v.generic_view((0, 0), (nrows, Const::<1>)).into_owned())
        } else {
            None
        }
    }
}

impl<T: Scalar, D: DimAdd<U1>, S: RawStorage<T, D>> Vector<T, D, S> {
    /// Constructs a new vector of higher dimension by appending `element` to the end of `self`.
    #[inline]
    #[must_use]
    pub fn push(&self, element: T) -> OVector<T, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<DimSum<D, U1>>,
    {
        let len = self.len();
        let hnrows = DimSum::<D, U1>::from_usize(len + 1);
        let mut res = Matrix::uninit(hnrows, Const::<1>);
        // This is basically a copy_from except that we warp the copied
        // values into MaybeUninit.
        res.generic_view_mut((0, 0), self.shape_generic())
            .zip_apply(self, |out, e| *out = MaybeUninit::new(e));
        res[(len, 0)] = MaybeUninit::new(element);

        // Safety: res has been fully initialized.
        unsafe { res.assume_init() }
    }
}

impl<T, R: Dim, C: Dim, S> AbsDiffEq for Matrix<T, R, C, S>
where
    T: Scalar + AbsDiffEq,
    S: RawStorage<T, R, C>,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon.clone()))
    }
}

impl<T, R: Dim, C: Dim, S> RelativeEq for Matrix<T, R, C, S>
where
    T: Scalar + RelativeEq,
    S: Storage<T, R, C>,
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.relative_eq(other, epsilon, max_relative)
    }
}

impl<T, R: Dim, C: Dim, S> UlpsEq for Matrix<T, R, C, S>
where
    T: Scalar + UlpsEq,
    S: RawStorage<T, R, C>,
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon.clone(), max_ulps))
    }
}

impl<T, R: Dim, C: Dim, S> PartialOrd for Matrix<T, R, C, S>
where
    T: Scalar + PartialOrd,
    S: RawStorage<T, R, C>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.shape() != other.shape() {
            return None;
        }

        if self.nrows() == 0 || self.ncols() == 0 {
            return Some(Ordering::Equal);
        }

        let mut first_ord = unsafe {
            self.data
                .get_unchecked_linear(0)
                .partial_cmp(other.data.get_unchecked_linear(0))
        };

        if let Some(first_ord) = first_ord.as_mut() {
            let mut it = self.iter().zip(other.iter());
            let _ = it.next(); // Drop the first elements (we already tested it).

            for (left, right) in it {
                if let Some(ord) = left.partial_cmp(right) {
                    match ord {
                        Ordering::Equal => { /* Does not change anything. */ }
                        Ordering::Less => {
                            if *first_ord == Ordering::Greater {
                                return None;
                            }
                            *first_ord = ord
                        }
                        Ordering::Greater => {
                            if *first_ord == Ordering::Less {
                                return None;
                            }
                            *first_ord = ord
                        }
                    }
                } else {
                    return None;
                }
            }
        }

        first_ord
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        assert_eq!(
            self.shape(),
            right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.lt(b))
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        assert_eq!(
            self.shape(),
            right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.le(b))
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        assert_eq!(
            self.shape(),
            right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.gt(b))
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        assert_eq!(
            self.shape(),
            right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.ge(b))
    }
}

impl<T, R: Dim, C: Dim, S> Eq for Matrix<T, R, C, S>
where
    T: Eq,
    S: RawStorage<T, R, C>,
{
}

impl<T, R, R2, C, C2, S, S2> PartialEq<Matrix<T, R2, C2, S2>> for Matrix<T, R, C, S>
where
    T: PartialEq,
    C: Dim,
    C2: Dim,
    R: Dim,
    R2: Dim,
    S: RawStorage<T, R, C>,
    S2: RawStorage<T, R2, C2>,
{
    #[inline]
    fn eq(&self, right: &Matrix<T, R2, C2, S2>) -> bool {
        self.shape() == right.shape() && self.iter().zip(right.iter()).all(|(l, r)| l == r)
    }
}

macro_rules! impl_fmt {
    ($trait: path, $fmt_str_without_precision: expr_2021, $fmt_str_with_precision: expr_2021) => {
        impl<T, R: Dim, C: Dim, S> $trait for Matrix<T, R, C, S>
        where
            T: Scalar + $trait,
            S: RawStorage<T, R, C>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(feature = "std")]
                fn val_width<T: Scalar + $trait>(val: &T, f: &mut fmt::Formatter<'_>) -> usize {
                    match f.precision() {
                        Some(precision) => format!($fmt_str_with_precision, val, precision)
                            .chars()
                            .count(),
                        None => format!($fmt_str_without_precision, val).chars().count(),
                    }
                }

                #[cfg(not(feature = "std"))]
                fn val_width<T: Scalar + $trait>(_: &T, _: &mut fmt::Formatter<'_>) -> usize {
                    4
                }

                let (nrows, ncols) = self.shape();

                if nrows == 0 || ncols == 0 {
                    return write!(f, "[ ]");
                }

                let mut max_length = 0;

                for i in 0..nrows {
                    for j in 0..ncols {
                        max_length = crate::max(max_length, val_width(&self[(i, j)], f));
                    }
                }

                let max_length_with_space = max_length + 1;

                writeln!(f)?;
                writeln!(
                    f,
                    "  ┌ {:>width$} ┐",
                    "",
                    width = max_length_with_space * ncols - 1
                )?;

                for i in 0..nrows {
                    write!(f, "  │")?;
                    for j in 0..ncols {
                        let number_length = val_width(&self[(i, j)], f) + 1;
                        let pad = max_length_with_space - number_length;
                        write!(f, " {:>thepad$}", "", thepad = pad)?;
                        match f.precision() {
                            Some(precision) => {
                                write!(f, $fmt_str_with_precision, (*self)[(i, j)], precision)?
                            }
                            None => write!(f, $fmt_str_without_precision, (*self)[(i, j)])?,
                        }
                    }
                    writeln!(f, " │")?;
                }

                writeln!(
                    f,
                    "  └ {:>width$} ┘",
                    "",
                    width = max_length_with_space * ncols - 1
                )?;
                writeln!(f)
            }
        }
    };
}
impl_fmt!(fmt::Display, "{}", "{:.1$}");
impl_fmt!(fmt::LowerExp, "{:e}", "{:.1$e}");
impl_fmt!(fmt::UpperExp, "{:E}", "{:.1$E}");
impl_fmt!(fmt::Octal, "{:o}", "{:1$o}");
impl_fmt!(fmt::LowerHex, "{:x}", "{:1$x}");
impl_fmt!(fmt::UpperHex, "{:X}", "{:1$X}");
impl_fmt!(fmt::Binary, "{:b}", "{:.1$b}");
impl_fmt!(fmt::Pointer, "{:p}", "{:.1$p}");

#[cfg(test)]
mod tests {
    #[test]
    fn empty_display() {
        let vec: Vec<f64> = Vec::new();
        let dvector = crate::DVector::from_vec(vec);
        assert_eq!(format!("{}", dvector), "[ ]")
    }

    #[test]
    fn lower_exp() {
        let test = crate::Matrix2::new(1e6, 2e5, 2e-5, 1.);
        assert_eq!(
            format!("{:e}", test),
            r"
  ┌           ┐
  │  1e6  2e5 │
  │ 2e-5  1e0 │
  └           ┘

"
        )
    }
}

/// # Cross product
impl<
    T: Scalar + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
> Matrix<T, R, C, S>
{
    /// The perpendicular product between two 2D column vectors, i.e. `a.x * b.y - a.y * b.x`.
    #[inline]
    #[must_use]
    pub fn perp<R2, C2, SB>(&self, b: &Matrix<T, R2, C2, SB>) -> T
    where
        R2: Dim,
        C2: Dim,
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, U2>
            + SameNumberOfColumns<C, U1>
            + SameNumberOfRows<R2, U2>
            + SameNumberOfColumns<C2, U1>,
    {
        let shape = self.shape();
        assert_eq!(
            shape,
            b.shape(),
            "2D vector perpendicular product dimension mismatch."
        );
        assert_eq!(
            shape,
            (2, 1),
            "2D perpendicular product requires (2, 1) vectors {shape:?}",
        );

        // SAFETY: assertion above ensures correct shape
        let ax = unsafe { self.get_unchecked((0, 0)).clone() };
        let ay = unsafe { self.get_unchecked((1, 0)).clone() };
        let bx = unsafe { b.get_unchecked((0, 0)).clone() };
        let by = unsafe { b.get_unchecked((1, 0)).clone() };

        ax * by - ay * bx
    }

    // TODO: use specialization instead of an assertion.
    /// The 3D cross product between two vectors.
    ///
    /// Panics if the shape is not 3D vector. In the future, this will be implemented only for
    /// dynamically-sized matrices and statically-sized 3D matrices.
    #[inline]
    #[must_use]
    pub fn cross<R2, C2, SB>(&self, b: &Matrix<T, R2, C2, SB>) -> MatrixCross<T, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        SB: RawStorage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let shape = self.shape();
        assert_eq!(shape, b.shape(), "Vector cross product dimension mismatch.");
        assert!(
            shape == (3, 1) || shape == (1, 3),
            "Vector cross product dimension mismatch: must be (3, 1) or (1, 3) but found {shape:?}.",
        );

        if shape.0 == 3 {
            unsafe {
                let mut res = Matrix::uninit(Dim::from_usize(3), Dim::from_usize(1));

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((1, 0));
                let az = self.get_unchecked((2, 0));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((1, 0));
                let bz = b.get_unchecked((2, 0));

                *res.get_unchecked_mut((0, 0)) =
                    MaybeUninit::new(ay.clone() * bz.clone() - az.clone() * by.clone());
                *res.get_unchecked_mut((1, 0)) =
                    MaybeUninit::new(az.clone() * bx.clone() - ax.clone() * bz.clone());
                *res.get_unchecked_mut((2, 0)) =
                    MaybeUninit::new(ax.clone() * by.clone() - ay.clone() * bx.clone());

                // Safety: res is now fully initialized.
                res.assume_init()
            }
        } else {
            unsafe {
                let mut res = Matrix::uninit(Dim::from_usize(1), Dim::from_usize(3));

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((0, 1));
                let az = self.get_unchecked((0, 2));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((0, 1));
                let bz = b.get_unchecked((0, 2));

                *res.get_unchecked_mut((0, 0)) =
                    MaybeUninit::new(ay.clone() * bz.clone() - az.clone() * by.clone());
                *res.get_unchecked_mut((0, 1)) =
                    MaybeUninit::new(az.clone() * bx.clone() - ax.clone() * bz.clone());
                *res.get_unchecked_mut((0, 2)) =
                    MaybeUninit::new(ax.clone() * by.clone() - ay.clone() * bx.clone());

                // Safety: res is now fully initialized.
                res.assume_init()
            }
        }
    }
}

impl<T: Scalar + Field, S: RawStorage<T, U3>> Vector<T, U3, S> {
    /// Computes the matrix `M` such that for all vector `v` we have `M * v == self.cross(&v)`.
    #[inline]
    #[must_use]
    pub fn cross_matrix(&self) -> OMatrix<T, U3, U3> {
        OMatrix::<T, U3, U3>::new(
            T::zero(),
            -self[2].clone(),
            self[1].clone(),
            self[2].clone(),
            T::zero(),
            -self[0].clone(),
            -self[1].clone(),
            self[0].clone(),
            T::zero(),
        )
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// The smallest angle between two vectors.
    #[inline]
    #[must_use]
    pub fn angle<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<T, R2, C2, SB>) -> T::SimdRealField
    where
        SB: Storage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        let prod = self.dotc(other);
        let n1 = self.norm();
        let n2 = other.norm();

        if n1.is_zero() || n2.is_zero() {
            T::SimdRealField::zero()
        } else {
            let cang = prod.simd_real() / (n1 * n2);
            cang.simd_clamp(-T::SimdRealField::one(), T::SimdRealField::one())
                .simd_acos()
        }
    }
}

impl<T, R: Dim, C: Dim, S> AbsDiffEq for Unit<Matrix<T, R, C, S>>
where
    T: Scalar + AbsDiffEq,
    S: RawStorage<T, R, C>,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<T, R: Dim, C: Dim, S> RelativeEq for Unit<Matrix<T, R, C, S>>
where
    T: Scalar + RelativeEq,
    S: Storage<T, R, C>,
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<T, R: Dim, C: Dim, S> UlpsEq for Unit<Matrix<T, R, C, S>>
where
    T: Scalar + UlpsEq,
    S: RawStorage<T, R, C>,
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

impl<T, R, C, S> Hash for Matrix<T, R, C, S>
where
    T: Scalar + Hash,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (nrows, ncols) = self.shape();
        (nrows, ncols).hash(state);

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    self.get_unchecked((i, j)).hash(state);
                }
            }
        }
    }
}

impl<T, D, S> Unit<Vector<T, D, S>>
where
    T: Scalar,
    D: Dim,
    S: RawStorage<T, D, U1>,
{
    /// Cast the components of `self` to another type.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::<f64>::y_axis();
    /// let v2 = v.cast::<f32>();
    /// assert_eq!(v2, Vector3::<f32>::y_axis());
    /// ```
    pub fn cast<T2: Scalar>(self) -> Unit<OVector<T2, D>>
    where
        T: Scalar,
        OVector<T2, D>: SupersetOf<Vector<T, D, S>>,
        DefaultAllocator: Allocator<D, U1>,
    {
        Unit::new_unchecked(crate::convert_ref(self.as_ref()))
    }
}

impl<T, S> Matrix<T, U1, U1, S>
where
    S: RawStorage<T, U1, U1>,
{
    /// Returns a reference to the single element in this matrix.
    ///
    /// As opposed to indexing, using this provides type-safety
    /// when flattening dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let inner_product: f32 = *(v.transpose() * v).as_scalar();
    /// ```
    ///
    ///```compile_fail
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let inner_product = (v * v.transpose()).item(); // Typo, does not compile.
    ///```
    pub fn as_scalar(&self) -> &T {
        &self[(0, 0)]
    }
    /// Get a mutable reference to the single element in this matrix
    ///
    /// As opposed to indexing, using this provides type-safety
    /// when flattening dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let mut inner_product = (v.transpose() * v);
    /// *inner_product.as_scalar_mut() = 3.;
    /// ```
    ///
    ///```compile_fail
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let mut inner_product = (v * v.transpose());
    /// *inner_product.as_scalar_mut() = 3.;
    ///```
    pub fn as_scalar_mut(&mut self) -> &mut T
    where
        S: RawStorageMut<T, U1>,
    {
        &mut self[(0, 0)]
    }
    /// Convert this 1x1 matrix by reference into a scalar.
    ///
    /// As opposed to indexing, using this provides type-safety
    /// when flattening dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let mut inner_product: f32 = (v.transpose() * v).to_scalar();
    /// ```
    ///
    ///```compile_fail
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let mut inner_product: f32 = (v * v.transpose()).to_scalar();
    ///```
    pub fn to_scalar(&self) -> T
    where
        T: Clone,
    {
        self.as_scalar().clone()
    }
}

impl<T> super::alias::Matrix1<T> {
    /// Convert this 1x1 matrix into a scalar.
    ///
    /// As opposed to indexing, using this provides type-safety
    /// when flattening dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Vector3, Matrix2, U1};
    /// let v = Vector3::new(0., 0., 1.);
    /// let inner_product: f32 = (v.transpose() * v).into_scalar();
    /// assert_eq!(inner_product, 1.);
    /// ```
    ///
    ///```compile_fail
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(0., 0., 1.);
    /// let mut inner_product: f32 = (v * v.transpose()).into_scalar();
    ///```
    pub fn into_scalar(self) -> T {
        let [[scalar]] = self.data.0;
        scalar
    }
}
