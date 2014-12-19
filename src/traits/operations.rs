//! Low level operations on vectors and matrices.

use std::num::{Float, SignedInt};
use traits::structure::SquareMat;

/// Result of a partial ordering.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Show, Copy)]
pub enum POrdering {
    /// Result of a strict comparison.
    PartialLess,
    /// Equality relationship.
    PartialEqual,
    /// Result of a strict comparison.
    PartialGreater,
    /// Result of a comparison between two objects that are not comparable.
    NotComparable
}

impl POrdering {
    /// Returns `true` if `self` is equal to `Equal`.
    pub fn is_eq(&self) -> bool {
        *self == POrdering::PartialEqual
    }

    /// Returns `true` if `self` is equal to `Less`.
    pub fn is_lt(&self) -> bool {
        *self == POrdering::PartialLess
    }

    /// Returns `true` if `self` is equal to `Less` or `Equal`.
    pub fn is_le(&self) -> bool {
        *self == POrdering::PartialLess || *self == POrdering::PartialEqual
    }

    /// Returns `true` if `self` is equal to `Greater`.
    pub fn is_gt(&self) -> bool {
        *self == POrdering::PartialGreater
    }

    /// Returns `true` if `self` is equal to `Greater` or `Equal`.
    pub fn is_ge(&self) -> bool {
        *self == POrdering::PartialGreater || *self == POrdering::PartialEqual
    }

    /// Returns `true` if `self` is equal to `NotComparable`.
    pub fn is_not_comparable(&self) -> bool {
        *self == POrdering::NotComparable
    }

    /// Creates a `POrdering` from an `Ordering`.
    pub fn from_ordering(ord: Ordering) -> POrdering {
        match ord {
            Less    => POrdering::PartialLess,
            Equal   => POrdering::PartialEqual,
            Greater => POrdering::PartialGreater
        }
    }

    /// Converts this `POrdering` to an `Ordering`.
    ///
    /// Returns `None` if `self` is `NotComparable`.
    pub fn to_ordering(self) -> Option<Ordering> {
        match self {
            POrdering::PartialLess    => Some(Less),
            POrdering::PartialEqual   => Some(Equal),
            POrdering::PartialGreater => Some(Greater),
            POrdering::NotComparable  => None
        }
    }
}

/// Pointwise ordering operations.
pub trait POrd {
    /// Returns the infimum of this value and another
    fn inf(&self, other: &Self) -> Self;

    /// Returns the supremum of this value and another
    fn sup(&self, other: &Self) -> Self;

    /// Compare `self` and `other` using a partial ordering relation.
    fn partial_cmp(&self, other: &Self) -> POrdering;

    /// Returns `true` iff `self` and `other` are comparable and `self <= other`.
    #[inline]
    fn partial_le(&self, other: &Self) -> bool {
        POrd::partial_cmp(self, other).is_le()
    }

    /// Returns `true` iff `self` and `other` are comparable and `self < other`.
    #[inline]
    fn partial_lt(&self, other: &Self) -> bool {
        POrd::partial_cmp(self, other).is_lt()
    }

    /// Returns `true` iff `self` and `other` are comparable and `self >= other`.
    #[inline]
    fn partial_ge(&self, other: &Self) -> bool {
        POrd::partial_cmp(self, other).is_ge()
    }

    /// Returns `true` iff `self` and `other` are comparable and `self > other`.
    #[inline]
    fn partial_gt(&self, other: &Self) -> bool {
        POrd::partial_cmp(self, other).is_gt()
    }

    /// Return the minimum of `self` and `other` if they are comparable.
    #[inline]
    fn partial_min<'a>(&'a self, other: &'a Self) -> Option<&'a Self> {
        match POrd::partial_cmp(self, other) {
            POrdering::PartialLess | POrdering::PartialEqual => Some(self),
            POrdering::PartialGreater             => Some(other),
            POrdering::NotComparable              => None
        }
    }

    /// Return the maximum of `self` and `other` if they are comparable.
    #[inline]
    fn partial_max<'a>(&'a self, other: &'a Self) -> Option<&'a Self> {
        match POrd::partial_cmp(self, other) {
            POrdering::PartialGreater | POrdering::PartialEqual => Some(self),
            POrdering::PartialLess   => Some(other),
            POrdering::NotComparable => None
        }
    }

    /// Clamp `value` between `min` and `max`. Returns `None` if `value` is not comparable to
    /// `min` or `max`.
    #[inline]
    fn partial_clamp<'a>(&'a self, min: &'a Self, max: &'a Self) -> Option<&'a Self> {
        let v_min = self.partial_cmp(min);
        let v_max = self.partial_cmp(max);

        if v_min.is_not_comparable() || v_max.is_not_comparable() {
            None
        }
        else {
            if v_min.is_lt() {
                Some(min)
            }
            else if v_max.is_gt() {
                Some(max)
            }
            else {
                Some(self)
            }
        }
    }
}

/// Trait for testing approximate equality
pub trait ApproxEq<Eps> {
    /// Default epsilon for approximation.
    fn approx_epsilon(unused_self: Option<Self>) -> Eps;

    /// Tests approximate equality using a custom epsilon.
    fn approx_eq_eps(&self, other: &Self, epsilon: &Eps) -> bool;

    /// Tests approximate equality.
    #[inline]
    fn approx_eq(&self, other: &Self) -> bool {
        self.approx_eq_eps(other, &ApproxEq::approx_epsilon(None::<Self>))
    }
}

impl ApproxEq<f32> for f32 {
    #[inline]
    fn approx_epsilon(_: Option<f32>) -> f32 {
        1.0e-6
    }

    #[inline]
    fn approx_eq_eps(&self, other: &f32, epsilon: &f32) -> bool {
        ::abs(&(*self - *other)) < *epsilon
    }
}

impl ApproxEq<f64> for f64 {
    #[inline]
    fn approx_epsilon(_: Option<f64>) -> f64 {
        1.0e-6
    }

    #[inline]
    fn approx_eq_eps(&self, other: &f64, approx_epsilon: &f64) -> bool {
        ::abs(&(*self - *other)) < *approx_epsilon
    }
}

/// Trait of objects having an absolute value.
/// This is useful if the object does not have the same type as its absolute value.
pub trait Absolute<A> {
    /// Computes some absolute value of this object.
    /// Typically, this will make all component of a matrix or vector positive.
    fn abs(&Self) -> A;
}

/// Trait of objects having an inverse. Typically used to implement matrix inverse.
pub trait Inv {
    /// Returns the inverse of `m`.
    fn inv_cpy(&self) -> Option<Self>;

    /// In-place version of `inverse`.
    fn inv(&mut self) -> bool;
}

/// Trait of objects having a determinant. Typically used by square matrices.
pub trait Det<N> {
    /// Returns the determinant of `m`.
    fn det(&self) -> N;
}

/// Trait of objects which can be transposed.
pub trait Transpose {
    /// Computes the transpose of a matrix.
    fn transpose_cpy(&self) -> Self;

    /// In-place version of `transposed`.
    fn transpose(&mut self);
}

/// Traits of objects having an outer product.
pub trait Outer<M> {
    /// Computes the outer product: `a * b`
    fn outer(&self, other: &Self) -> M;
}

/// Trait for computing the covariance of a set of data.
pub trait Cov<M> {
    /// Computes the covariance of the obsevations stored by `m`:
    ///
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn cov(&self) -> M;

    /// Computes the covariance of the obsevations stored by `m`:
    ///
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn cov_to(&self, out: &mut M) {
        *out = self.cov()
    }
}

/// Trait for computing the covariance of a set of data.
pub trait Mean<N> {
    /// Computes the mean of the observations stored by `v`.
    /// 
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn mean(&self) -> N;
}

/// Trait for computing the eigenvector and eigenvalues of a square matrix usin the QR algorithm.
pub trait EigenQR<N, V>: SquareMat<N, V> {
    /// Computes the eigenvectors and eigenvalues of this matrix.
    fn eigen_qr(&self, eps: &N, niter: uint) -> (Self, V);
}

// XXX: those two traits should not exist since there is generalized operator overloading of Add
// and Sub.
// However, using the same trait multiple time as a trait bound (ex: impl<T: Add<N, V> + Add<V, V>)
// does not work properly, mainly because the way we are doing generalized operator overloading is
// verry hacky.
//
// Hopefully, this will be fixed on a future version of the language!
/// Trait of objects having a right multiplication with another element.
pub trait RMul<V> {
    /// Computes `self * v`
    fn rmul(&self, v: &V) -> V;
}

impl<M: Copy + Mul<T, T>, T: Copy> RMul<T> for M {
    fn rmul(&self, v: &T) -> T {
        *self * *v
    }
}

/// Trait of objects having a left multiplication with another element.
pub trait LMul<V> {
    /// Computes `v * self`
    fn lmul(&self, &V) -> V;
}

impl<T: Copy + Mul<M, T>, M: Copy> LMul<T> for M {
    fn lmul(&self, v: &T) -> T {
        *v * *self
    }
}

// XXX: those traits should not exist since there is generalized operator overloading of Add and
// Sub.
// However, using the same trait multiple time as a trait bound (ex: impl<T: Add<N, V> + Add<V, V>)
// does not work properly, mainly because the way we are doing generalized operator overloading is
// verry hacky.
//
// Hopefully, this will be fixed on a future version of the language!
/// Trait of objects having an addition with a scalar.
pub trait ScalarAdd<N> {
    /// Gets the result of `self + n`.
    fn add_s(&self, n: &N) -> Self;
}

/// Trait of objects having a subtraction with a scalar.
pub trait ScalarSub<N> {
    /// Gets the result of `self - n`.
    fn sub_s(&self, n: &N) -> Self;
}

/// Trait of objects having a multiplication with a scalar.
pub trait ScalarMul<N> {
    /// Gets the result of `self * n`.
    fn mul_s(&self, n: &N) -> Self;
}

/// Trait of objects having a division by a scalar.
pub trait ScalarDiv<N> {
    /// Gets the result of `self / n`.
    fn div_s(&self, n: &N) -> Self;
}

/// Trait of objects implementing the `y = ax + y` operation.
pub trait Axpy<N> {
    /// Adds $$a * x$$ to `self`.
    fn axpy(&mut self, a: &N, x: &Self);
}

/*
 *
 *
 * Some implementations for scalar types.
 *
 *
 */
// FIXME: move this to another module ?
macro_rules! impl_absolute(
    ($n: ty) => {
        impl Absolute<$n> for $n {
            #[inline]
            fn abs(n: &$n) -> $n {
                n.abs()
            }
        }
    }
);
macro_rules! impl_absolute_id(
    ($n: ty) => {
        impl Absolute<$n> for $n {
            #[inline]
            fn abs(n: &$n) -> $n {
                *n
            }
        }
    }
);

impl_absolute!(f32);
impl_absolute!(f64);
impl_absolute!(i8);
impl_absolute!(i16);
impl_absolute!(i32);
impl_absolute!(i64);
impl_absolute!(int);
impl_absolute_id!(u8);
impl_absolute_id!(u16);
impl_absolute_id!(u32);
impl_absolute_id!(u64);
impl_absolute_id!(uint);

macro_rules! impl_axpy(
    ($n: ty) => {
        impl Axpy<$n> for $n {
            #[inline]
            fn axpy(&mut self, a: &$n, x: &$n) {
                *self = *self + *a * *x
            }
        }
    }
);

impl_axpy!(f32);
impl_axpy!(f64);
impl_axpy!(i8);
impl_axpy!(i16);
impl_axpy!(i32);
impl_axpy!(i64);
impl_axpy!(int);
impl_axpy!(u8);
impl_axpy!(u16);
impl_axpy!(u32);
impl_axpy!(u64);
impl_axpy!(uint);
