//! Traits giving structural informations on linear algebra objects or the space they live in.

use std::f32;
use std::f64;
use std::num::{Int, Float, FloatMath};
use std::slice::{Items, MutItems};
use traits::operations::{RMul, LMul, Axpy, Transpose, Inv, Absolute};
use traits::geometry::{Dot, Norm, Orig};

/// Basic integral numeric trait.
pub trait BaseNum: Copy + Zero + One + Add<Self, Self> + Sub<Self, Self> + Mul<Self, Self> +
                   Div<Self, Self> + Rem<Self, Self> + Neg<Self> + PartialEq + Absolute<Self> +
                   Axpy<Self> {
}

/// Basic floating-point number numeric trait.
pub trait BaseFloat: FloatMath + BaseNum {
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
    fn frac_2_sqrtpi() -> Self;

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
pub trait Mat<N, R, C>: Row<R> + Col<C> + RMul<R> + LMul<C> + Index<(uint, uint), N> { }

impl<N, M, R, C> Mat<N, R, C> for M
    where M: Row<R> + Col<C> + RMul<R> + LMul<C> + Index<(uint, uint), N> {
}

/// Trait implemented by square matrices.
pub trait SquareMat<N, V>: Mat<N, V, V> + Mul<Self, Self> + Eye + Transpose + Diag<V> + Inv + Dim +
                           One {
}

impl<N, V, M> SquareMat<N, V> for M
    where M: Mat<N, V, V> + Mul<M, M> + Eye + Transpose + Diag<V> + Inv + Dim + One {
}

/// Trait for constructing the identity matrix
pub trait Eye {
    /// Return the identity matrix of specified dimension
    fn new_identity(dim: uint) -> Self;
}

/// Additive identity.
pub trait Zero {
    /// Returns the additive identity.
    fn zero() -> Self;
    /// Tests if `self` is exactly zero.
    fn is_zero(&self) -> bool;
}

/// Multiplicative identity.
pub trait One {
    /// Returns the multiplicative identity.
    fn one() -> Self;
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
pub trait Basis {
    /// Iterates through the canonical basis of the space in which this object lives.
    fn canonical_basis(|Self| -> bool);

    /// Iterates through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis(&Self, |Self| -> bool);

    /// Gets the ith element of the canonical basis.
    fn canonical_basis_element(i: uint) -> Option<Self>;
}

/// Trait to access rows of a matrix or a vector.
pub trait Row<R> {
    /// The number of column of `self`.
    fn nrows(&self) -> uint;
    /// Reads the `i`-th row of `self`.
    fn row(&self, i: uint) -> R;
    /// Writes the `i`-th row of `self`.
    fn set_row(&mut self, i: uint, R);

    // FIXME: add iterators on rows: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access columns of a matrix or vector.
pub trait Col<C> {
    /// The number of column of this matrix or vector.
    fn ncols(&self) -> uint;

    /// Reads the `i`-th column of `self`.
    fn col(&self, i: uint) -> C;

    /// Writes the `i`-th column of `self`.
    fn set_col(&mut self, i: uint, C);

    // FIXME: add iterators on columns: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access part of a column of a matrix
pub trait ColSlice<C> {
    /// Returns a view to a slice of a column of a matrix.
    fn col_slice(&self, col_id: uint, row_start: uint, row_end: uint) -> C;
}

/// Trait to access part of a row of a matrix
pub trait RowSlice<R> {
    /// Returns a view to a slice of a row of a matrix.
    fn row_slice(&self, row_id: uint, col_start: uint, col_end: uint) -> R;
}

/// Trait of objects having a spacial dimension known at compile time.
pub trait Dim {
    /// The dimension of the object.
    fn dim(unused_self: Option<Self>) -> uint;
}

/// Trait to get the diagonal of square matrices.
pub trait Diag<V> {
    /// Creates a new matrix with the given diagonal.
    fn from_diag(diag: &V) -> Self;

    /// Sets the diagonal of this matrix.
    fn set_diag(&mut self, diag: &V);

    /// The diagonal of this matrix.
    fn diag(&self) -> V;
}

/// The shape of an indexable object.
pub trait Shape<I, Res>: Index<I, Res> {
    /// Returns the shape of an indexable object.
    fn shape(&self) -> I;
}

/// This is a workaround of current Rust limitations.
///
/// It exists because the `I` trait cannot be used to express write access.
/// Thus, this is the same as the `I` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<I, Res>: Shape<I, Res> + IndexMut<I, Res> {
    #[deprecated = "use the Index `[]` overloaded operator instead"]
    /// Reads the `i`-th element of `self`.
    fn at(&self, i: I) -> Res;
    #[deprecated = "use the IndexMut `[]` overloaded operator instead"]
    /// Writes to the `i`-th element of `self`.
    fn set(&mut self, i: I, Res);
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: I, j: I);

    /// Reads the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_at(&self, i: I) -> Res;
    /// Writes to the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_set(&mut self, i: I, Res);
}

/// This is a workaround of current Rust limitations.
///
/// Traits of objects which can be iterated through like a vector.
pub trait Iterable<N> {
    /// Gets a vector-like read-only iterator.
    fn iter<'l>(&'l self) -> Items<'l, N>;
}

/// This is a workaround of current Rust limitations.
///
/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut<N> {
    /// Gets a vector-like read-write iterator.
    fn iter_mut<'l>(&'l mut self) -> MutItems<'l, N>;
}

/*
 * Vec related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
#[deprecated = "This will be removed in the future. Use point + vector operations instead."]
pub trait VecAsPnt<P> {
    /// Converts this point to its associated vector.
    fn to_pnt(self) -> P;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_pnt<'a>(&'a self) -> &'a P;
}

/// Trait grouping most common operations on vectors.
pub trait NumVec<N>: Dim + Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero + PartialEq +
                     Mul<N, Self> + Div<N, Self> + Dot<N> + Axpy<N> + Index<uint, N> {
}

/// Trait of vector with components implementing the `BaseFloat` trait.
pub trait FloatVec<N: BaseFloat>: NumVec<N> + Norm<N> + Basis {
}

/*
 * Pnt related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
pub trait PntAsVec<V> {
    /// Converts this point to its associated vector.
    fn to_vec(self) -> V;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_vec<'a>(&'a self) -> &'a V;

    // NOTE: this is used in some places to overcome some limitations untill the trait reform is
    // done on rustc.
    /// Sets the coordinates of this point to match those of a given vector.
    fn set_coords(&mut self, coords: V);
}

/// Trait grouping most common operations on points.
// XXX: the vector space element `V` should be an associated type. Though this would prevent V from
// having bounds (they are not supported yet). So, for now, we will just use a type parameter.
pub trait NumPnt<N, V>:
          Copy + PntAsVec<V> + Dim + Sub<Self, V> + Orig + Neg<Self> + PartialEq + Mul<N, Self> +
          Div<N, Self> + Add<V, Self> + Axpy<N> + Index<uint, N> { // FIXME: + Sub<V, Self>
}

/// Trait of points with components implementing the `BaseFloat` trait.
pub trait FloatPnt<N: BaseFloat, V: Norm<N>>: NumPnt<N, V> {
    /// Computes the square distance between two points.
    #[inline]
    fn sqdist(&self, other: &Self) -> N {
        (*self - *other).sqnorm()
    }

    /// Computes the distance between two points.
    #[inline]
    fn dist(&self, other: &Self) -> N {
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



// Zero and One
macro_rules! impl_zero_one(
    ($n: ty, $zero: expr, $one: expr) => {
        impl Zero for $n {
            #[inline]
            fn zero() -> $n {
                $zero
            }

            #[inline]
            fn is_zero(&self) -> bool {
                *self == $zero
            }
        }

        impl One for $n {
            fn one() -> $n {
                $one
            }
        }
    }
);

impl_zero_one!(f32, 0.0, 1.0);
impl_zero_one!(f64, 0.0, 1.0);
impl_zero_one!(i8, 0, 1);
impl_zero_one!(i16, 0, 1);
impl_zero_one!(i32, 0, 1);
impl_zero_one!(i64, 0, 1);
impl_zero_one!(int, 0, 1);
impl_zero_one!(u8, 0, 1);
impl_zero_one!(u16, 0, 1);
impl_zero_one!(u32, 0, 1);
impl_zero_one!(u64, 0, 1);
impl_zero_one!(uint, 0, 1);


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

impl_bounded!(f32, Float::min_value(), Float::max_value());
impl_bounded!(f64, Float::min_value(), Float::max_value());
impl_bounded!(i8, Int::min_value(), Int::max_value());
impl_bounded!(i16, Int::min_value(), Int::max_value());
impl_bounded!(i32, Int::min_value(), Int::max_value());
impl_bounded!(i64, Int::min_value(), Int::max_value());
impl_bounded!(int, Int::min_value(), Int::max_value());
impl_bounded!(u8, Int::min_value(), Int::max_value());
impl_bounded!(u16, Int::min_value(), Int::max_value());
impl_bounded!(u32, Int::min_value(), Int::max_value());
impl_bounded!(u64, Int::min_value(), Int::max_value());
impl_bounded!(uint, Int::min_value(), Int::max_value());


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
                $n::consts::PI_2
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
            fn frac_2_sqrtpi() -> $n {
                $n::consts::FRAC_2_SQRTPI
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

impl BaseNum for f32 { }
impl BaseNum for f64 { }

impl_base_float!(f32);
impl_base_float!(f64);
