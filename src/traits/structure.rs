//! Traits giving structural informations on linear algebra objects or the space they live in.

use std::{f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, isize, usize};
use std::slice::{Iter, IterMut};
use std::ops::{Add, Sub, Mul, Div, Rem,
               AddAssign, SubAssign, MulAssign, DivAssign, RemAssign,
               Index, IndexMut, Neg};
use num::{Float, Zero, One};
use traits::operations::{Axpy, Transpose, Inverse, Absolute, ApproxEq};
use traits::geometry::{Dot, Norm, Origin};

/// Basic integral numeric trait.
pub trait BaseNum: Copy + Zero + One +
                   Add<Self, Output = Self> + Sub<Self, Output = Self> +
                   Mul<Self, Output = Self> + Div<Self, Output = Self> +
                   Rem<Self, Output = Self> +
                   AddAssign<Self> + SubAssign<Self> +
                   MulAssign<Self> + DivAssign<Self> +
                   RemAssign<Self> +
                   PartialEq + Absolute<Self> + Axpy<Self> {
}

/// Basic floating-point number numeric trait.
pub trait BaseFloat: Float + Cast<f64> + BaseNum + ApproxEq<Self> + Neg {
    /// Archimedes' constant.
    fn pi() -> Self;
    /// 2.0 * pi.
    fn two_pi() -> Self;
    /// pi / 2.0.
    fn frac_pi_2() -> Self;
    /// pi / 3.0.
    fn frac_pi_3() -> Self;
    /// pi / 4.0.
    fn frac_pi_4() -> Self;
    /// pi / 6.0.
    fn frac_pi_6() -> Self;
    /// pi / 8.0.
    fn frac_pi_8() -> Self;
    /// 1.0 / pi.
    fn frac_1_pi() -> Self;
    /// 2.0 / pi.
    fn frac_2_pi() -> Self;
    /// 2.0 / sqrt(pi).
    fn frac_2_sqrt_pi() -> Self;

    /// Euler's number.
    fn e() -> Self;
    /// log2(e).
    fn log2_e() -> Self;
    /// log10(e).
    fn log10_e() -> Self;
    /// ln(2.0).
    fn ln_2() -> Self;
    /// ln(10.0).
    fn ln_10() -> Self;
}

/// Traits of objects which can be created from an object of type `T`.
pub trait Cast<T> {
    /// Converts an element of type `T` to an element of type `Self`.
    fn from(t: T) -> Self;
}

/// Trait of matrices.
///
/// A matrix has rows and columns and are able to multiply them.
pub trait Matrix<N, R, C: Mul<Self, Output = R>>: Sized +
                                               Row<R> + Column<C> + Mul<R, Output = C> +
                                               Index<(usize, usize), Output = N>
{ }

impl<N, M, R, C> Matrix<N, R, C> for M
    where M: Row<R> + Column<C> + Mul<R, Output = C> + Index<(usize, usize), Output = N>,
          C: Mul<M, Output = R>,
{ }

/// Trait implemented by square matrices.
pub trait SquareMatrix<N, V: Mul<Self, Output = V>>: Matrix<N, V, V> +
                                                  Mul<Self, Output = Self> +
                                                  Eye + Transpose + Diagonal<V> + Inverse + Dimension + One {
}

impl<N, V, M> SquareMatrix<N, V> for M
    where M: Matrix<N, V, V> + Mul<M, Output = M> + Eye + Transpose + Diagonal<V> + Inverse + Dimension + One,
          V: Mul<M, Output = V> {
}

/// Trait for constructing the identity matrix
pub trait Eye {
    /// Return the identity matrix of specified dimension
    fn new_identity(dimension: usize) -> Self;
}

/// Trait for constructiong an object repeating a value.
pub trait Repeat<N> {
    /// Returns a value with filled by `val`.
    fn repeat(val: N) -> Self;
}

/// Types that have maximum and minimum value.
pub trait Bounded {
    /// The minimum value.
    #[inline]
    fn min_value() -> Self;
    /// The maximum value.
    #[inline]
    fn max_value() -> Self;
}

// FIXME: return an iterator instead
/// Traits of objects which can form a basis (typically vectors).
pub trait Basis: Sized {
    /// Iterates through the canonical basis of the space in which this object lives.
    fn canonical_basis<F: FnMut(Self) -> bool>(F);

    /// Iterates through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis<F: FnMut(Self) -> bool>(&Self, F);

    /// Gets the ith element of the canonical basis.
    fn canonical_basis_element(i: usize) -> Option<Self>;
}

/// Trait to access rows of a matrix or a vector.
pub trait Row<R> {
    /// The number of column of `self`.
    fn nrows(&self) -> usize;
    /// Reads the `i`-th row of `self`.
    fn row(&self, i: usize) -> R;
    /// Writes the `i`-th row of `self`.
    fn set_row(&mut self, i: usize, R);

    // FIXME: add iterators on rows: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access columns of a matrix or vector.
pub trait Column<C> {
    /// The number of column of this matrix or vector.
    fn ncols(&self) -> usize;

    /// Reads the `i`-th column of `self`.
    fn column(&self, i: usize) -> C;

    /// Writes the `i`-th column of `self`.
    fn set_column(&mut self, i: usize, C);

    // FIXME: add iterators on columns: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access part of a column of a matrix
pub trait ColumnSlice<C> {
    /// Returns a view to a slice of a column of a matrix.
    fn column_slice(&self, column_id: usize, row_start: usize, row_end: usize) -> C;
}

/// Trait to access part of a row of a matrix
pub trait RowSlice<R> {
    /// Returns a view to a slice of a row of a matrix.
    fn row_slice(&self, row_id: usize, column_start: usize, column_end: usize) -> R;
}

/// Trait of objects having a spacial dimension known at compile time.
pub trait Dimension: Sized {
    /// The dimension of the object.
    fn dimension(_unused: Option<Self>) -> usize;
}

/// Trait to get the diagonal of square matrices.
pub trait Diagonal<V> {
    /// Creates a new matrix with the given diagonal.
    fn from_diagonal(diagonal: &V) -> Self;

    /// The diagonal of this matrix.
    fn diagonal(&self) -> V;
}

/// Trait to set the diagonal of square matrices.
pub trait DiagonalMut<V>: Diagonal<V> {
    /// Sets the diagonal of this matrix.
    fn set_diagonal(&mut self, diagonal: &V);
}

/// The shape of an indexable object.
pub trait Shape<I>: Index<I> {
    /// Returns the shape of an indexable object.
    fn shape(&self) -> I;
}

/// This is a workaround of current Rust limitations.
///
/// It exists because the `I` trait cannot be used to express write access.
/// Thus, this is the same as the `I` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<I, N>: Shape<I> + IndexMut<I, Output = N> {
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: I, j: I);

    /// Reads the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_at(&self, i: I) -> N;
    /// Writes to the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_set(&mut self, i: I, N);
}

/// This is a workaround of current Rust limitations.
///
/// Traits of objects which can be iterated through like a vector.
pub trait Iterable<N> {
    /// Gets a vector-like read-only iterator.
    fn iter(&self) -> Iter<N>;
}

/// This is a workaround of current Rust limitations.
///
/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut<N> {
    /// Gets a vector-like read-write iterator.
    fn iter_mut(&mut self) -> IterMut<N>;
}

/*
 * Vector related traits.
 */
/// Trait grouping most common operations on vectors.
pub trait NumVector<N>: Add<Self, Output = Self> + Sub<Self, Output = Self> +
                        // Mul<Self, Output = Self> + Div<Self, Output = Self> +

                        // Add<N, Output = Self> + Sub<N, Output = Self> +
                        Mul<N, Output = Self> + Div<N, Output = Self> +

                        AddAssign<Self> + SubAssign<Self> +
                        // MulAssign<Self> + DivAssign<Self> +

                        // AddAssign<N> + SubAssign<N> +
                        MulAssign<N> + DivAssign<N> +

                        Dimension + Index<usize, Output = N> +
                        Zero + PartialEq + Dot<N> + Axpy<N> {
}

/// Trait of vector with components implementing the `BaseFloat` trait.
pub trait FloatVector<N: BaseFloat>: NumVector<N> + Norm<NormType = N> + Neg<Output = Self> + Basis + ApproxEq<N> {
}

/*
 * Point related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
pub trait PointAsVector {
    /// The vector type of the vector space associated to this point's affine space.
    type Vector;

    /// Converts this point to its associated vector.
    fn to_vector(self) -> Self::Vector;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_vector(&self) -> &Self::Vector;

    // NOTE: this is used in some places to overcome some limitations untill the trait reform is
    // done on rustc.
    /// Sets the coordinates of this point to match those of a given vector.
    fn set_coords(&mut self, coords: Self::Vector);
}

/// Trait grouping most common operations on points.
// XXX: the vector space element `V` should be an associated type. Though this would prevent V from
// having bounds (they are not supported yet). So, for now, we will just use a type parameter.
pub trait NumPoint<N>:
          Copy +
          PointAsVector +
          Dimension +
          Origin +
          PartialEq +
          Axpy<N> +
          Sub<Self, Output = <Self as PointAsVector>::Vector> +

          Mul<N, Output = Self> + Div<N, Output = Self> +
          Add<<Self as PointAsVector>::Vector, Output = Self> +

          MulAssign<N> + DivAssign<N> +
          AddAssign<<Self as PointAsVector>::Vector> +

          Index<usize, Output = N> { // FIXME: + Sub<V, Self>
}

/// Trait of points with components implementing the `BaseFloat` trait.
pub trait FloatPoint<N: BaseFloat>: NumPoint<N> + Sized
    where <Self as PointAsVector>::Vector: Norm<NormType = N> {
    /// Computes the square distance between two points.
    #[inline]
    fn distance_squared(&self, other: &Self) -> N {
        (*self - *other).norm_squared()
    }

    /// Computes the distance between two points.
    #[inline]
    fn distance(&self, other: &Self) -> N {
        (*self - *other).norm()
    }
}

/*
 *
 *
 * Some implementations for builtin types.
 *
 *
 */
// Bounded
macro_rules! impl_bounded(
    ($n: ty, $min: expr, $max: expr) => {
        impl Bounded for $n {
            #[inline]
            fn min_value() -> $n {
                $min
            }

            #[inline]
            fn max_value() -> $n {
                $max
            }
        }
    }
);

impl_bounded!(f32, f32::MIN, f32::MAX);
impl_bounded!(f64, f64::MIN, f64::MAX);
impl_bounded!(i8, i8::MIN, i8::MAX);
impl_bounded!(i16, i16::MIN, i16::MAX);
impl_bounded!(i32, i32::MIN, i32::MAX);
impl_bounded!(i64, i64::MIN, i64::MAX);
impl_bounded!(isize, isize::MIN, isize::MAX);
impl_bounded!(u8, u8::MIN, u8::MAX);
impl_bounded!(u16, u16::MIN, u16::MAX);
impl_bounded!(u32, u32::MIN, u32::MAX);
impl_bounded!(u64, u64::MIN, u64::MAX);
impl_bounded!(usize, usize::MIN, usize::MAX);


// BaseFloat
macro_rules! impl_base_float(
    ($n: ident) => {
        impl BaseFloat for $n {
            /// Archimedes' constant.
            fn pi() -> $n {
                $n::consts::PI
            }

            /// 2.0 * pi.
            fn two_pi() -> $n {
                2.0 * $n::consts::PI
            }

            /// pi / 2.0.
            fn frac_pi_2() -> $n {
                $n::consts::FRAC_PI_2
            }

            /// pi / 3.0.
            fn frac_pi_3() -> $n {
                $n::consts::FRAC_PI_3
            }

            /// pi / 4.0.
            fn frac_pi_4() -> $n {
                $n::consts::FRAC_PI_4
            }

            /// pi / 6.0.
            fn frac_pi_6() -> $n {
                $n::consts::FRAC_PI_6
            }

            /// pi / 8.0.
            fn frac_pi_8() -> $n {
                $n::consts::FRAC_PI_8
            }

            /// 1.0 / pi.
            fn frac_1_pi() -> $n {
                $n::consts::FRAC_1_PI
            }

            /// 2.0 / pi.
            fn frac_2_pi() -> $n {
                $n::consts::FRAC_2_PI
            }

            /// 2.0 / sqrt(pi).
            fn frac_2_sqrt_pi() -> $n {
                $n::consts::FRAC_2_SQRT_PI
            }


            /// Euler's number.
            fn e() -> $n {
                $n::consts::E
            }

            /// log2(e).
            fn log2_e() -> $n {
                $n::consts::LOG2_E
            }

            /// log10(e).
            fn log10_e() -> $n {
                $n::consts::LOG10_E
            }

            /// ln(2.0).
            fn ln_2() -> $n {
                $n::consts::LN_2
            }

            /// ln(10.0).
            fn ln_10() -> $n {
                $n::consts::LN_10
            }
        }
    }
);

impl<T: Copy + Zero + One +
   Add<T, Output = T> + Sub<T, Output = T> +
   Mul<T, Output = T> + Div<T, Output = T> +
   Rem<T, Output = T> +
   AddAssign<T> + SubAssign<T> +
   MulAssign<T> + DivAssign<T> +
   RemAssign<T> +
   PartialEq + Absolute<T> + Axpy<T>> BaseNum for T {}

impl_base_float!(f32);
impl_base_float!(f64);
