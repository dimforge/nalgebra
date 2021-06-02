use num::{One, Zero};
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub, Field, SupersetOf};
use simba::simd::SimdPartialOrd;

use crate::base::allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR};
use crate::base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, DimAdd, DimSum, IsNotStaticOne, U1, U2, U3};
use crate::base::iter::{
    ColumnIter, ColumnIterMut, MatrixIter, MatrixIterMut, RowIter, RowIterMut,
};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, Owned, SameShapeStorage, Storage, StorageMut,
};
use crate::base::{Const, DefaultAllocator, OMatrix, OVector, Scalar, Unit};
use crate::{ArrayStorage, SMatrix, SimdComplexField};

#[cfg(any(feature = "std", feature = "alloc"))]
use crate::{DMatrix, DVector, Dynamic, VecStorage};

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
///
/// #### Statistics
/// - [Common operations <span style="float:right;">`row_sum`, `column_mean`, `variance`…</span>](#common-statistics-operations)
/// - [Find the min and max components <span style="float:right;">`min`, `max`, `amin`, `amax`, `camin`, `cmax`…</span>](#find-the-min-and-max-components)
/// - [Find the min and max components (vector-specific methods) <span style="float:right;">`argmin`, `argmax`, `icamin`, `icamax`…</span>](#find-the-min-and-max-components-vector-specific-methods)
///
/// #### Iteration, map, and fold
/// - [Iteration on components, rows, and columns <span style="float:right;">`iter`, `column_iter`…</span>](#iteration-on-components-rows-and-columns)
/// - [Elementwise mapping and folding <span style="float:right;">`map`, `fold`, `zip_map`…</span>](#elementwise-mapping-and-folding)
/// - [Folding or columns and rows <span style="float:right;">`compress_rows`, `compress_columns`…</span>](#folding-on-columns-and-rows)
///
/// #### Vector and matrix slicing
/// - [Creating matrix slices from `&[T]` <span style="float:right;">`from_slice`, `from_slice_with_strides`…</span>](#creating-matrix-slices-from-t)
/// - [Creating mutable matrix slices from `&mut [T]` <span style="float:right;">`from_slice_mut`, `from_slice_with_strides_mut`…</span>](#creating-mutable-matrix-slices-from-mut-t)
/// - [Slicing based on index and length <span style="float:right;">`row`, `columns`, `slice`…</span>](#slicing-based-on-index-and-length)
/// - [Mutable slicing based on index and length <span style="float:right;">`row_mut`, `columns_mut`, `slice_mut`…</span>](#mutable-slicing-based-on-index-and-length)
/// - [Slicing based on ranges <span style="float:right;">`rows_range`, `columns_range`…</span>](#slicing-based-on-ranges)
/// - [Mutable slicing based on ranges <span style="float:right;">`rows_range_mut`, `columns_range_mut`…</span>](#mutable-slicing-based-on-ranges)
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
/// components.
///
/// The matrix dimensions parameters `R` and `C` can either be:
/// - type-level unsigned integer constants (e.g. `U1`, `U124`) from the `nalgebra::` root module.
/// All numbers from 0 to 127 are defined that way.
/// - type-level unsigned integer constants (e.g. `U1024`, `U10000`) from the `typenum::` crate.
/// Using those, you will not get error messages as nice as for numbers smaller than 128 defined on
/// the `nalgebra::` module.
/// - the special value `Dynamic` from the `nalgebra::` root module. This indicates that the
/// specified dimension is not known at compile-time. Note that this will generally imply that the
/// matrix data storage `S` performs a dynamic allocation and contains extra metadata for the
/// matrix shape.
///
/// Note that mixing `Dynamic` with type-level unsigned integers is allowed. Actually, a
/// dynamically-sized column vector should be represented as a `Matrix<T, Dynamic, U1, S>` (given
/// some concrete types for `T` and a compatible data storage type `S`).
#[repr(C)]
#[derive(Clone, Copy)]
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
    //       of the `Storage` trait. However, because we don't have
    //       specialization, this is not bossible because these `T, R, C`
    //       allows us to desambiguate a lot of configurations.
    _phantoms: PhantomData<(T, R, C)>,
}

impl<T, R: Dim, C: Dim, S: fmt::Debug> fmt::Debug for Matrix<T, R, C, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Matrix")
            .field("data", &self.data)
            .finish()
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

#[cfg(feature = "abomonation-serialize")]
impl<T: Scalar, R: Dim, C: Dim, S: Abomonation> Abomonation for Matrix<T, R, C, S> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

#[cfg(feature = "compare")]
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> matrixcompare_core::Matrix<T>
    for Matrix<T, R, C, S>
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> matrixcompare_core::Access<T> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "compare")]
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> matrixcompare_core::DenseAccess<T>
    for Matrix<T, R, C, S>
{
    fn fetch_single(&self, row: usize, col: usize) -> T {
        self.index((row, col)).clone()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> bytemuck::Zeroable
    for Matrix<T, R, C, S>
where
    S: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> bytemuck::Pod for Matrix<T, R, C, S>
where
    S: bytemuck::Pod,
    Self: Copy,
{
}

#[cfg(feature = "rkyv-serialize-no-std")]
mod rkyv_impl {
    use super::Matrix;
    use core::marker::PhantomData;
    use rkyv::{offset_of, project_struct, Archive, Deserialize, Fallible, Serialize};

    impl<T: Archive, R: Archive, C: Archive, S: Archive> Archive for Matrix<T, R, C, S> {
        type Archived = Matrix<T::Archived, R::Archived, C::Archived, S::Archived>;
        type Resolver = S::Resolver;

        fn resolve(
            &self,
            pos: usize,
            resolver: Self::Resolver,
            out: &mut core::mem::MaybeUninit<Self::Archived>,
        ) {
            self.data.resolve(
                pos + offset_of!(Self::Archived, data),
                resolver,
                project_struct!(out: Self::Archived => data),
            );
        }
    }

    impl<T: Archive, R: Archive, C: Archive, S: Serialize<_S>, _S: Fallible + ?Sized> Serialize<_S>
        for Matrix<T, R, C, S>
    {
        fn serialize(&self, serializer: &mut _S) -> Result<Self::Resolver, _S::Error> {
            Ok(self.data.serialize(serializer)?)
        }
    }

    impl<T: Archive, R: Archive, C: Archive, S: Archive, D: Fallible + ?Sized>
        Deserialize<Matrix<T, R, C, S>, D>
        for Matrix<T::Archived, R::Archived, C::Archived, S::Archived>
    where
        S::Archived: Deserialize<S, D>,
    {
        fn deserialize(&self, deserializer: &mut D) -> Result<Matrix<T, R, C, S>, D::Error> {
            Ok(Matrix {
                data: self.data.deserialize(deserializer)?,
                _phantoms: PhantomData,
            })
        }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S> {
    /// Creates a new matrix with the given data without statically checking that the matrix
    /// dimension matches the storage dimension.
    #[inline(always)]
    pub const unsafe fn from_data_statically_unchecked(data: S) -> Matrix<T, R, C, S> {
        Matrix {
            data,
            _phantoms: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> SMatrix<T, R, C> {
    /// Creates a new statically-allocated matrix from the given [ArrayStorage].
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
    /// Creates a new heap-allocated matrix from the given [VecStorage].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    pub const fn from_vec_storage(storage: VecStorage<T, Dynamic, Dynamic>) -> Self {
        // This is sound because the dimensions of the matrix and the storage are guaranteed
        // to be the same
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

// TODO: Consider removing/deprecating `from_vec_storage` once we are able to make
// `from_data` const fn compatible
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> DVector<T> {
    /// Creates a new heap-allocated matrix from the given [VecStorage].
    ///
    /// This method exists primarily as a workaround for the fact that `from_data` can not
    /// work in `const fn` contexts.
    pub const fn from_vec_storage(storage: VecStorage<T, Dynamic, U1>) -> Self {
        // This is sound because the dimensions of the matrix and the storage are guaranteed
        // to be the same
        unsafe { Self::from_data_statically_unchecked(storage) }
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Creates a new matrix with the given data.
    #[inline(always)]
    pub fn from_data(data: S) -> Self {
        unsafe { Self::from_data_statically_unchecked(data) }
    }

    /// Creates a new uninitialized matrix with the given uninitialized data
    pub unsafe fn from_uninitialized_data(data: mem::MaybeUninit<S>) -> mem::MaybeUninit<Self> {
        let res: Matrix<T, R, C, mem::MaybeUninit<S>> = Matrix {
            data,
            _phantoms: PhantomData,
        };
        let res: mem::MaybeUninit<Matrix<T, R, C, mem::MaybeUninit<S>>> =
            mem::MaybeUninit::new(res);
        // safety: since we wrap the inner MaybeUninit in an outer MaybeUninit above, the fact that the `data` field is partially-uninitialized is still opaque.
        // with s/transmute_copy/transmute/, rustc claims that `MaybeUninit<Matrix<T, R, C, MaybeUninit<S>>>` may be of a different size from `MaybeUninit<Matrix<T, R, C, S>>`
        // but MaybeUninit's documentation says "MaybeUninit<T> is guaranteed to have the same size, alignment, and ABI as T", which implies those types should be the same size
        let res: mem::MaybeUninit<Matrix<T, R, C, S>> = mem::transmute_copy(&res);
        res
    }

    /// The shape of this matrix returned as the tuple (number of rows, number of columns).
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.shape(), (3, 4));
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    /// The number of rows of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.nrows(), 3);
    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    /// The number of columns of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.ncols(), 4);
    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    /// The strides (row stride, column stride) of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mat = DMatrix::<f32>::zeros(10, 10);
    /// let slice = mat.slice_with_steps((0, 0), (5, 3), (1, 2));
    /// // The column strides is the number of steps (here 2) multiplied by the corresponding dimension.
    /// assert_eq!(mat.strides(), (1, 10));
    #[inline]
    pub fn strides(&self) -> (usize, usize) {
        let (srows, scols) = self.data.strides();
        (srows.value(), scols.value())
    }

    /// Computes the row and column coordinates of the i-th element of this matrix seen as a
    /// vector.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let i = m.vector_to_matrix_index(3);
    /// assert_eq!(i, (1, 1));
    /// assert_eq!(m[i], m[3]);
    /// ```
    #[inline]
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

    /// Returns a pointer to the start of the matrix.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be aligned
    /// and non-null.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(1, 2,
    ///                      3, 4);
    /// let ptr = m.as_ptr();
    /// assert_eq!(unsafe { *ptr }, m[0]);
    /// ```
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.ptr()
    }

    /// Tests whether `self` and `rhs` are equal up to a given epsilon.
    ///
    /// See `relative_eq` from the `RelativeEq` trait for more details.
    #[inline]
    pub fn relative_eq<R2, C2, SB>(
        &self,
        other: &Matrix<T, R2, C2, SB>,
        eps: T::Epsilon,
        max_relative: T::Epsilon,
    ) -> bool
    where
        T: RelativeEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        T::Epsilon: Copy,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.relative_eq(b, eps, max_relative))
    }

    /// Tests whether `self` and `rhs` are exactly equal.
    #[inline]
    pub fn eq<R2, C2, SB>(&self, other: &Matrix<T, R2, C2, SB>) -> bool
    where
        T: PartialEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| *a == *b)
    }

    /// Moves this matrix into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        Matrix::from_data(self.data.into_owned())
    }

    // TODO: this could probably benefit from specialization.
    // XXX: bad name.
    /// Moves this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn into_owned_sum<R2, C2>(self) -> MatrixSum<T, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<T, R, C, R2, C2>,
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

    /// Clones this matrix to one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        Matrix::from_data(self.data.clone_owned())
    }

    /// Clones this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn clone_owned_sum<R2, C2>(&self) -> MatrixSum<T, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<T, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();
        let nrows: SameShapeR<R, R2> = Dim::from_usize(nrows);
        let ncols: SameShapeC<C, C2> = Dim::from_usize(ncols);

        let mut res: MatrixSum<T, R, C, R2, C2> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(nrows, ncols) };

        // TODO: use copy_from
        for j in 0..res.ncols() {
            for i in 0..res.nrows() {
                unsafe {
                    *res.get_unchecked_mut((i, j)) = self.get_unchecked((i, j)).inlined_clone();
                }
            }
        }

        res
    }

    /// Transposes `self` and store the result into `out`.
    #[inline]
    pub fn transpose_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<T, R2, C2>,
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
                unsafe {
                    *out.get_unchecked_mut((j, i)) = self.get_unchecked((i, j)).inlined_clone();
                }
            }
        }
    }

    /// Transposes `self`.
    #[inline]
    #[must_use = "Did you mean to use transpose_mut()?"]
    pub fn transpose(&self) -> OMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res = crate::unimplemented_or_uninitialized_generic!(ncols, nrows);
            self.transpose_to(&mut res);

            res
        }
    }
}

/// # Elementwise mapping and folding
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a matrix containing the result of `f` applied to each of its entries.
    #[inline]
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, mut f: F) -> OMatrix<T2, R, C>
    where
        DefaultAllocator: Allocator<T2, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res: OMatrix<T2, R, C> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a)
                }
            }
        }

        res
    }

    /// Cast the components of `self` to another type.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// let q = Vector3::new(1.0f64, 2.0, 3.0);
    /// let q2 = q.cast::<f32>();
    /// assert_eq!(q2, Vector3::new(1.0f32, 2.0, 3.0));
    /// ```
    pub fn cast<T2: Scalar>(self) -> OMatrix<T2, R, C>
    where
        OMatrix<T2, R, C>: SupersetOf<Self>,
        DefaultAllocator: Allocator<T2, R, C>,
    {
        crate::convert(self)
    }

    /// Similar to `self.iter().fold(init, f)` except that `init` is replaced by a closure.
    ///
    /// The initialization closure is given the first component of this matrix:
    /// - If the matrix has no component (0 rows or 0 columns) then `init_f` is called with `None`
    /// and its return value is the value returned by this method.
    /// - If the matrix has has least one component, then `init_f` is called with the first component
    /// to compute the initial value. Folding then continues on all the remaining components of the matrix.
    #[inline]
    pub fn fold_with<T2>(
        &self,
        init_f: impl FnOnce(Option<&T>) -> T2,
        f: impl FnMut(T2, &T) -> T2,
    ) -> T2 {
        let mut it = self.iter();
        let init = init_f(it.next());
        it.fold(init, f)
    }

    /// Returns a matrix containing the result of `f` applied to each of its entries. Unlike `map`,
    /// `f` also gets passed the row and column index, i.e. `f(row, col, value)`.
    #[inline]
    pub fn map_with_location<T2: Scalar, F: FnMut(usize, usize, T) -> T2>(
        &self,
        mut f: F,
    ) -> OMatrix<T2, R, C>
    where
        DefaultAllocator: Allocator<T2, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res: OMatrix<T2, R, C> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(i, j, a)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `rhs`.
    #[inline]
    pub fn zip_map<T2, N3, S2, F>(&self, rhs: &Matrix<T2, R, C, S2>, mut f: F) -> OMatrix<N3, R, C>
    where
        T2: Scalar,
        N3: Scalar,
        S2: Storage<T2, R, C>,
        F: FnMut(T, T2) -> N3,
        DefaultAllocator: Allocator<N3, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res: OMatrix<N3, R, C> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(nrows, ncols) };

        assert_eq!(
            (nrows.value(), ncols.value()),
            rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = rhs.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a, b)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `b`, and `c`.
    #[inline]
    pub fn zip_zip_map<T2, N3, N4, S2, S3, F>(
        &self,
        b: &Matrix<T2, R, C, S2>,
        c: &Matrix<N3, R, C, S3>,
        mut f: F,
    ) -> OMatrix<N4, R, C>
    where
        T2: Scalar,
        N3: Scalar,
        N4: Scalar,
        S2: Storage<T2, R, C>,
        S3: Storage<N3, R, C>,
        F: FnMut(T, T2, N3) -> N4,
        DefaultAllocator: Allocator<N4, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res: OMatrix<N4, R, C> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(nrows, ncols) };

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
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = b.data.get_unchecked(i, j).inlined_clone();
                    let c = c.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a, b, c)
                }
            }
        }

        res
    }

    /// Folds a function `f` on each entry of `self`.
    #[inline]
    pub fn fold<Acc>(&self, init: Acc, mut f: impl FnMut(Acc, T) -> Acc) -> Acc {
        let (nrows, ncols) = self.data.shape();

        let mut res = init;

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    res = f(res, a)
                }
            }
        }

        res
    }

    /// Folds a function `f` on each pairs of entries from `self` and `rhs`.
    #[inline]
    pub fn zip_fold<T2, R2, C2, S2, Acc>(
        &self,
        rhs: &Matrix<T2, R2, C2, S2>,
        init: Acc,
        mut f: impl FnMut(Acc, T, T2) -> Acc,
    ) -> Acc
    where
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: Storage<T2, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = init;

        assert_eq!(
            (nrows.value(), ncols.value()),
            rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = rhs.data.get_unchecked(i, j).inlined_clone();
                    res = f(res, a, b)
                }
            }
        }

        res
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on it.
    #[inline]
    pub fn apply<F: FnMut(T) -> T>(&mut self, mut f: F)
    where
        S: StorageMut<T, R, C>,
    {
        let (nrows, ncols) = self.shape();

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    *e = f(e.inlined_clone())
                }
            }
        }
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on its components
    /// joined with the components from `rhs`.
    #[inline]
    pub fn zip_apply<T2, R2, C2, S2>(
        &mut self,
        rhs: &Matrix<T2, R2, C2, S2>,
        mut f: impl FnMut(T, T2) -> T,
    ) where
        S: StorageMut<T, R, C>,
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: Storage<T2, R2, C2>,
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
                    let rhs = rhs.get_unchecked((i, j)).inlined_clone();
                    *e = f(e.inlined_clone(), rhs)
                }
            }
        }
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on its components
    /// joined with the components from `b` and `c`.
    #[inline]
    pub fn zip_zip_apply<T2, R2, C2, S2, N3, R3, C3, S3>(
        &mut self,
        b: &Matrix<T2, R2, C2, S2>,
        c: &Matrix<N3, R3, C3, S3>,
        mut f: impl FnMut(T, T2, N3) -> T,
    ) where
        S: StorageMut<T, R, C>,
        T2: Scalar,
        R2: Dim,
        C2: Dim,
        S2: Storage<T2, R2, C2>,
        N3: Scalar,
        R3: Dim,
        C3: Dim,
        S3: Storage<N3, R3, C3>,
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
                    let b = b.get_unchecked((i, j)).inlined_clone();
                    let c = c.get_unchecked((i, j)).inlined_clone();
                    *e = f(e.inlined_clone(), b, c)
                }
            }
        }
    }
}

/// # Iteration on components, rows, and columns
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Iterates through this matrix coordinates in column-major order.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(11, 12, 13,
    ///                          21, 22, 23);
    /// let mut it = mat.iter();
    /// assert_eq!(*it.next().unwrap(), 11);
    /// assert_eq!(*it.next().unwrap(), 21);
    /// assert_eq!(*it.next().unwrap(), 12);
    /// assert_eq!(*it.next().unwrap(), 22);
    /// assert_eq!(*it.next().unwrap(), 13);
    /// assert_eq!(*it.next().unwrap(), 23);
    /// assert!(it.next().is_none());
    #[inline]
    pub fn iter(&self) -> MatrixIter<T, R, C, S> {
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
    pub fn row_iter(&self) -> RowIter<T, R, C, S> {
        RowIter::new(self)
    }

    /// Iterate through the columns of this matrix.
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, column) in a.column_iter().enumerate() {
    ///     assert_eq!(column, a.column(i))
    /// }
    /// ```
    #[inline]
    pub fn column_iter(&self) -> ColumnIter<T, R, C, S> {
        ColumnIter::new(self)
    }

    /// Mutably iterates through this matrix coordinates.
    #[inline]
    pub fn iter_mut(&mut self) -> MatrixIterMut<T, R, C, S>
    where
        S: StorageMut<T, R, C>,
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
    pub fn row_iter_mut(&mut self) -> RowIterMut<T, R, C, S>
    where
        S: StorageMut<T, R, C>,
    {
        RowIterMut::new(self)
    }

    /// Mutably iterates through this matrix columns.
    ///
    /// # Example
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
    #[inline]
    pub fn column_iter_mut(&mut self) -> ColumnIterMut<T, R, C, S>
    where
        S: StorageMut<T, R, C>,
    {
        ColumnIterMut::new(self)
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a mutable pointer to the start of the matrix.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be aligned
    /// and non-null.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.ptr_mut()
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        debug_assert!(row_cols1.0 < self.nrows() && row_cols1.1 < self.ncols());
        debug_assert!(row_cols2.0 < self.nrows() && row_cols2.1 < self.ncols());
        self.data.swap_unchecked(row_cols1, row_cols2)
    }

    /// Swaps two entries.
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

    /// Fills this matrix with the content of a slice. Both must hold the same number of elements.
    ///
    /// The components of the slice are assumed to be ordered in column-major order.
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[T]) {
        let (nrows, ncols) = self.shape();

        assert!(
            nrows * ncols == slice.len(),
            "The slice must contain the same number of elements as the matrix."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut((i, j)) =
                        slice.get_unchecked(i + j * nrows).inlined_clone();
                }
            }
        }
    }

    /// Fills this matrix with the content of another one. Both must have the same shape.
    #[inline]
    pub fn copy_from<R2, C2, SB>(&mut self, other: &Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(
            self.shape() == other.shape(),
            "Unable to copy from a matrix with a different shape."
        );

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((i, j)).inlined_clone();
                }
            }
        }
    }

    /// Fills this matrix with the content of the transpose another one.
    #[inline]
    pub fn tr_copy_from<R2, C2, SB>(&mut self, other: &Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
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
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((j, i)).inlined_clone();
                }
            }
        }
    }

    // TODO: rename `apply` to `apply_mut` and `apply_into` to `apply`?
    /// Returns `self` with each of its components replaced by the result of a closure `f` applied on it.
    #[inline]
    pub fn apply_into<F: FnMut(T) -> T>(mut self, f: F) -> Self {
        self.apply(f);
        self
    }
}

impl<T: Scalar, D: Dim, S: Storage<T, D>> Vector<T, D, S> {
    /// Gets a reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked(&self, i: usize) -> &T {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear(i)
    }
}

impl<T: Scalar, D: Dim, S: StorageMut<T, D>> Vector<T, D, S> {
    /// Gets a mutable reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked_mut(&mut self, i: usize) -> &mut T {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear_mut(i)
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: ContiguousStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Extracts a slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: ContiguousStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Extracts a mutable slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

impl<T: Scalar, D: Dim, S: StorageMut<T, D, D>> Matrix<T, D, D, S> {
    /// Transposes the square matrix `self` in-place.
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

impl<T: SimdComplexField, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Takes the adjoint (aka. conjugate-transpose) of `self` and store the result into `out`.
    #[inline]
    pub fn adjoint_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<T, R2, C2>,
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
                unsafe {
                    *out.get_unchecked_mut((j, i)) = self.get_unchecked((i, j)).simd_conjugate();
                }
            }
        }
    }

    /// The adjoint (aka. conjugate-transpose) of `self`.
    #[inline]
    #[must_use = "Did you mean to use adjoint_mut()?"]
    pub fn adjoint(&self) -> OMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res: OMatrix<_, C, R> =
                crate::unimplemented_or_uninitialized_generic!(ncols, nrows);
            self.adjoint_to(&mut res);

            res
        }
    }

    /// Takes the conjugate and transposes `self` and store the result into `out`.
    #[deprecated(note = "Renamed `self.adjoint_to(out)`.")]
    #[inline]
    pub fn conjugate_transpose_to<R2, C2, SB>(&self, out: &mut Matrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        self.adjoint_to(out)
    }

    /// The conjugate transposition of `self`.
    #[deprecated(note = "Renamed `self.adjoint()`.")]
    #[inline]
    pub fn conjugate_transpose(&self) -> OMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        self.adjoint()
    }

    /// The conjugate of `self`.
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        self.map(|e| e.simd_conjugate())
    }

    /// Divides each component of the complex matrix `self` by the given real.
    #[inline]
    #[must_use = "Did you mean to use unscale_mut()?"]
    pub fn unscale(&self, real: T::SimdRealField) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        self.map(|e| e.simd_unscale(real))
    }

    /// Multiplies each component of the complex matrix `self` by the given real.
    #[inline]
    #[must_use = "Did you mean to use scale_mut()?"]
    pub fn scale(&self, real: T::SimdRealField) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        self.map(|e| e.simd_scale(real))
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// The conjugate of the complex matrix `self` computed in-place.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.apply(|e| e.simd_conjugate())
    }

    /// Divides each component of the complex matrix `self` by the given real.
    #[inline]
    pub fn unscale_mut(&mut self, real: T::SimdRealField) {
        self.apply(|e| e.simd_unscale(real))
    }

    /// Multiplies each component of the complex matrix `self` by the given real.
    #[inline]
    pub fn scale_mut(&mut self, real: T::SimdRealField) {
        self.apply(|e| e.simd_scale(real))
    }
}

impl<T: SimdComplexField, D: Dim, S: StorageMut<T, D, D>> Matrix<T, D, D, S> {
    /// Sets `self` to its adjoint.
    #[deprecated(note = "Renamed to `self.adjoint_mut()`.")]
    pub fn conjugate_transform_mut(&mut self) {
        self.adjoint_mut()
    }

    /// Sets `self` to its adjoint (aka. conjugate-transpose).
    pub fn adjoint_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 0..dim {
            for j in 0..i {
                unsafe {
                    let ref_ij = self.get_unchecked_mut((i, j)) as *mut T;
                    let ref_ji = self.get_unchecked_mut((j, i)) as *mut T;
                    let conj_ij = (*ref_ij).simd_conjugate();
                    let conj_ji = (*ref_ji).simd_conjugate();
                    *ref_ij = conj_ji;
                    *ref_ji = conj_ij;
                }
            }

            {
                let diag = unsafe { self.get_unchecked_mut((i, i)) };
                *diag = diag.simd_conjugate();
            }
        }
    }
}

impl<T: Scalar, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// The diagonal of this matrix.
    #[inline]
    pub fn diagonal(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<T, D>,
    {
        self.map_diagonal(|e| e)
    }

    /// Apply the given function to this matrix's diagonal and returns it.
    ///
    /// This is a more efficient version of `self.diagonal().map(f)` since this
    /// allocates only once.
    pub fn map_diagonal<T2: Scalar>(&self, mut f: impl FnMut(T) -> T2) -> OVector<T2, D>
    where
        DefaultAllocator: Allocator<T2, D>,
    {
        assert!(
            self.is_square(),
            "Unable to get the diagonal of a non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res: OVector<T2, D> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(dim, Const::<1>) };

        for i in 0..dim.value() {
            unsafe {
                *res.vget_unchecked_mut(i) = f(self.get_unchecked((i, i)).inlined_clone());
            }
        }

        res
    }

    /// Computes a trace of a square matrix, i.e., the sum of its diagonal elements.
    #[inline]
    pub fn trace(&self) -> T
    where
        T: Scalar + Zero + ClosedAdd,
    {
        assert!(
            self.is_square(),
            "Cannot compute the trace of non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res = T::zero();

        for i in 0..dim.value() {
            res += unsafe { self.get_unchecked((i, i)).inlined_clone() };
        }

        res
    }
}

impl<T: SimdComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// The symmetric part of `self`, i.e., `0.5 * (self + self.transpose())`.
    #[inline]
    pub fn symmetric_part(&self) -> OMatrix<T, D, D>
    where
        DefaultAllocator: Allocator<T, D, D>,
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

    /// The hermitian part of `self`, i.e., `0.5 * (self + self.adjoint())`.
    #[inline]
    pub fn hermitian_part(&self) -> OMatrix<T, D, D>
    where
        DefaultAllocator: Allocator<T, D, D>,
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

impl<T: Scalar + Zero + One, D: DimAdd<U1> + IsNotStaticOne, S: Storage<T, D, D>>
    Matrix<T, D, D, S>
{
    /// Yields the homogeneous matrix for this matrix, i.e., appending an additional dimension and
    /// and setting the diagonal element to `1`.
    #[inline]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimSum<D, U1>, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<T, DimSum<D, U1>, DimSum<D, U1>>,
    {
        assert!(
            self.is_square(),
            "Only square matrices can currently be transformed to homogeneous coordinates."
        );
        let dim = DimSum::<D, U1>::from_usize(self.nrows() + 1);
        let mut res = OMatrix::identity_generic(dim, dim);
        res.generic_slice_mut::<D, D>((0, 0), self.data.shape())
            .copy_from(&self);
        res
    }
}

impl<T: Scalar + Zero, D: DimAdd<U1>, S: Storage<T, D>> Vector<T, D, S> {
    /// Computes the coordinates in projective space of this vector, i.e., appends a `0` to its
    /// coordinates.
    #[inline]
    pub fn to_homogeneous(&self) -> OVector<T, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<T, DimSum<D, U1>>,
    {
        self.push(T::zero())
    }

    /// Constructs a vector from coordinates in projective space, i.e., removes a `0` at the end of
    /// `self`. Returns `None` if this last component is not zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: Vector<T, DimSum<D, U1>, SB>) -> Option<OVector<T, D>>
    where
        SB: Storage<T, DimSum<D, U1>>,
        DefaultAllocator: Allocator<T, D>,
    {
        if v[v.len() - 1].is_zero() {
            let nrows = D::from_usize(v.len() - 1);
            Some(v.generic_slice((0, 0), (nrows, Const::<1>)).into_owned())
        } else {
            None
        }
    }
}

impl<T: Scalar + Zero, D: DimAdd<U1>, S: Storage<T, D>> Vector<T, D, S> {
    /// Constructs a new vector of higher dimension by appending `element` to the end of `self`.
    #[inline]
    pub fn push(&self, element: T) -> OVector<T, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<T, DimSum<D, U1>>,
    {
        let len = self.len();
        let hnrows = DimSum::<D, U1>::from_usize(len + 1);
        let mut res: OVector<T, _> =
            unsafe { crate::unimplemented_or_uninitialized_generic!(hnrows, Const::<1>) };
        res.generic_slice_mut((0, 0), self.data.shape())
            .copy_from(self);
        res[(len, 0)] = element;

        res
    }
}

impl<T, R: Dim, C: Dim, S> AbsDiffEq for Matrix<T, R, C, S>
where
    T: Scalar + AbsDiffEq,
    S: Storage<T, R, C>,
    T::Epsilon: Copy,
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
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<T, R: Dim, C: Dim, S> RelativeEq for Matrix<T, R, C, S>
where
    T: Scalar + RelativeEq,
    S: Storage<T, R, C>,
    T::Epsilon: Copy,
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
    S: Storage<T, R, C>,
    T::Epsilon: Copy,
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
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl<T, R: Dim, C: Dim, S> PartialOrd for Matrix<T, R, C, S>
where
    T: Scalar + PartialOrd,
    S: Storage<T, R, C>,
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
    T: Scalar + Eq,
    S: Storage<T, R, C>,
{
}

impl<T, R, R2, C, C2, S, S2> PartialEq<Matrix<T, R2, C2, S2>> for Matrix<T, R, C, S>
where
    T: Scalar + PartialEq,
    C: Dim,
    C2: Dim,
    R: Dim,
    R2: Dim,
    S: Storage<T, R, C>,
    S2: Storage<T, R2, C2>,
{
    #[inline]
    fn eq(&self, right: &Matrix<T, R2, C2, S2>) -> bool {
        self.shape() == right.shape() && self.iter().zip(right.iter()).all(|(l, r)| l == r)
    }
}

macro_rules! impl_fmt {
    ($trait: path, $fmt_str_without_precision: expr, $fmt_str_with_precision: expr) => {
        impl<T, R: Dim, C: Dim, S> $trait for Matrix<T, R, C, S>
        where
            T: Scalar + $trait,
            S: Storage<T, R, C>,
            DefaultAllocator: Allocator<usize, R, C>,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                #[cfg(feature = "std")]
                fn val_width<T: Scalar + $trait>(val: &T, f: &mut fmt::Formatter) -> usize {
                    match f.precision() {
                        Some(precision) => format!($fmt_str_with_precision, val, precision)
                            .chars()
                            .count(),
                        None => format!($fmt_str_without_precision, val).chars().count(),
                    }
                }

                #[cfg(not(feature = "std"))]
                fn val_width<T: Scalar + $trait>(_: &T, _: &mut fmt::Formatter) -> usize {
                    4
                }

                let (nrows, ncols) = self.data.shape();

                if nrows.value() == 0 || ncols.value() == 0 {
                    return write!(f, "[ ]");
                }

                let mut max_length = 0;
                let mut lengths: OMatrix<usize, R, C> = Matrix::zeros_generic(nrows, ncols);
                let (nrows, ncols) = self.shape();

                for i in 0..nrows {
                    for j in 0..ncols {
                        lengths[(i, j)] = val_width(&self[(i, j)], f);
                        max_length = crate::max(max_length, lengths[(i, j)]);
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
                        let number_length = lengths[(i, j)] + 1;
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

/// # Cross product
impl<T: Scalar + ClosedAdd + ClosedSub + ClosedMul, R: Dim, C: Dim, S: Storage<T, R, C>>
    Matrix<T, R, C, S>
{
    /// The perpendicular product between two 2D column vectors, i.e. `a.x * b.y - a.y * b.x`.
    #[inline]
    pub fn perp<R2, C2, SB>(&self, b: &Matrix<T, R2, C2, SB>) -> T
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, U2>
            + SameNumberOfColumns<C, U1>
            + SameNumberOfRows<R2, U2>
            + SameNumberOfColumns<C2, U1>,
    {
        assert!(
            self.shape() == (2, 1),
            "2D perpendicular product requires (2, 1) vector but found {:?}",
            self.shape()
        );

        unsafe {
            self.get_unchecked((0, 0)).inlined_clone() * b.get_unchecked((1, 0)).inlined_clone()
                - self.get_unchecked((1, 0)).inlined_clone()
                    * b.get_unchecked((0, 0)).inlined_clone()
        }
    }

    // TODO: use specialization instead of an assertion.
    /// The 3D cross product between two vectors.
    ///
    /// Panics if the shape is not 3D vector. In the future, this will be implemented only for
    /// dynamically-sized matrices and statically-sized 3D matrices.
    #[inline]
    pub fn cross<R2, C2, SB>(&self, b: &Matrix<T, R2, C2, SB>) -> MatrixCross<T, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<T, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let shape = self.shape();
        assert_eq!(shape, b.shape(), "Vector cross product dimension mismatch.");
        assert!(
            (shape.0 == 3 && shape.1 == 1) || (shape.0 == 1 && shape.1 == 3),
            "Vector cross product dimension mismatch: must be (3, 1) or (1, 3) but found {:?}.",
            shape
        );

        if shape.0 == 3 {
            unsafe {
                // TODO: soooo ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(3);
                let ncols = SameShapeC::<C, C2>::from_usize(1);
                let mut res: MatrixCross<T, R, C, R2, C2> =
                    crate::unimplemented_or_uninitialized_generic!(nrows, ncols);

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((1, 0));
                let az = self.get_unchecked((2, 0));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((1, 0));
                let bz = b.get_unchecked((2, 0));

                *res.get_unchecked_mut((0, 0)) = ay.inlined_clone() * bz.inlined_clone()
                    - az.inlined_clone() * by.inlined_clone();
                *res.get_unchecked_mut((1, 0)) = az.inlined_clone() * bx.inlined_clone()
                    - ax.inlined_clone() * bz.inlined_clone();
                *res.get_unchecked_mut((2, 0)) = ax.inlined_clone() * by.inlined_clone()
                    - ay.inlined_clone() * bx.inlined_clone();

                res
            }
        } else {
            unsafe {
                // TODO: ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(1);
                let ncols = SameShapeC::<C, C2>::from_usize(3);
                let mut res: MatrixCross<T, R, C, R2, C2> =
                    crate::unimplemented_or_uninitialized_generic!(nrows, ncols);

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((0, 1));
                let az = self.get_unchecked((0, 2));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((0, 1));
                let bz = b.get_unchecked((0, 2));

                *res.get_unchecked_mut((0, 0)) = ay.inlined_clone() * bz.inlined_clone()
                    - az.inlined_clone() * by.inlined_clone();
                *res.get_unchecked_mut((0, 1)) = az.inlined_clone() * bx.inlined_clone()
                    - ax.inlined_clone() * bz.inlined_clone();
                *res.get_unchecked_mut((0, 2)) = ax.inlined_clone() * by.inlined_clone()
                    - ay.inlined_clone() * bx.inlined_clone();

                res
            }
        }
    }
}

impl<T: Scalar + Field, S: Storage<T, U3>> Vector<T, U3, S> {
    /// Computes the matrix `M` such that for all vector `v` we have `M * v == self.cross(&v)`.
    #[inline]
    pub fn cross_matrix(&self) -> OMatrix<T, U3, U3> {
        OMatrix::<T, U3, U3>::new(
            T::zero(),
            -self[2].inlined_clone(),
            self[1].inlined_clone(),
            self[2].inlined_clone(),
            T::zero(),
            -self[0].inlined_clone(),
            -self[1].inlined_clone(),
            self[0].inlined_clone(),
            T::zero(),
        )
    }
}

impl<T: SimdComplexField, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// The smallest angle between two vectors.
    #[inline]
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
    S: Storage<T, R, C>,
    T::Epsilon: Copy,
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
    T::Epsilon: Copy,
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
    S: Storage<T, R, C>,
    T::Epsilon: Copy,
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
    S: Storage<T, R, C>,
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
