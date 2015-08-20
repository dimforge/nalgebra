//! Low level operations on vectors and matrices.

use num::{Float, Signed};
use std::ops::Mul;
use std::cmp::Ordering;
use traits::structure::SquareMat;

/// Result of a partial ordering.
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
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
            Ordering::Less    => POrdering::PartialLess,
            Ordering::Equal   => POrdering::PartialEqual,
            Ordering::Greater => POrdering::PartialGreater
        }
    }

    /// Converts this `POrdering` to an `Ordering`.
    ///
    /// Returns `None` if `self` is `NotComparable`.
    pub fn to_ordering(self) -> Option<Ordering> {
        match self {
            POrdering::PartialLess    => Some(Ordering::Less),
            POrdering::PartialEqual   => Some(Ordering::Equal),
            POrdering::PartialGreater => Some(Ordering::Greater),
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
pub trait ApproxEq<Eps>: Sized {
    /// Default epsilon for approximation.
    fn approx_epsilon(unused_mut: Option<Self>) -> Eps;

    /// Tests approximate equality using a custom epsilon.
    fn approx_eq_eps(&self, other: &Self, epsilon: &Eps) -> bool;

    /// Default ULPs for approximation.
    fn approx_ulps(unused_mut: Option<Self>) -> u32;

    /// Tests approximate equality using units in the last place (ULPs)
    fn approx_eq_ulps(&self, other: &Self, ulps: u32) -> bool;

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

    fn approx_ulps(_: Option<f32>) -> u32 {
        8
    }

    fn approx_eq_ulps(&self, other: &f32, ulps: u32) -> bool {
        // Handle -0 == +0
        if *self == *other { return true; }

        // Otherwise, differing signs should be not-equal, even if within ulps
        if self.signum() != other.signum() { return false; }

        // IEEE754 floats are in the same order as 2s complement isizes
        // so this trick (subtracting the isizes) works.
        let iself: i32 = unsafe { ::std::mem::transmute(*self) };
        let iother: i32 = unsafe { ::std::mem::transmute(*other) };

        (iself - iother).abs() < ulps as i32
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

    fn approx_ulps(_: Option<f64>) -> u32 {
        8
    }

    fn approx_eq_ulps(&self, other: &f64, ulps: u32) -> bool {
        // Handle -0 == +0
        if *self == *other { return true; }

        // Otherwise, differing signs should be not-equal, even if within ulps
        if self.signum() != other.signum() { return false; }

        let iself: i64 = unsafe { ::std::mem::transmute(*self) };
        let iother: i64 = unsafe { ::std::mem::transmute(*other) };

        (iself - iother).abs() < ulps as i64
    }
}

impl<'a, N, T: ApproxEq<N>> ApproxEq<N> for &'a T {
    fn approx_epsilon(_: Option<&'a T>) -> N {
        ApproxEq::approx_epsilon(None::<T>)
    }

    fn approx_eq_eps(&self, other: &&'a T, approx_epsilon: &N) -> bool {
        ApproxEq::approx_eq_eps(*self, *other, approx_epsilon)
    }

    fn approx_ulps(_: Option<&'a T>) -> u32 {
        ApproxEq::approx_ulps(None::<T>)
    }

    fn approx_eq_ulps(&self, other: &&'a T, ulps: u32) -> bool {
        ApproxEq::approx_eq_ulps(*self, *other, ulps)
    }
}

impl<'a, N, T: ApproxEq<N>> ApproxEq<N> for &'a mut T {
    fn approx_epsilon(_: Option<&'a mut T>) -> N {
        ApproxEq::approx_epsilon(None::<T>)
    }

    fn approx_eq_eps(&self, other: &&'a mut T, approx_epsilon: &N) -> bool {
        ApproxEq::approx_eq_eps(*self, *other, approx_epsilon)
    }

    fn approx_ulps(_: Option<&'a mut T>) -> u32 {
        ApproxEq::approx_ulps(None::<T>)
    }

    fn approx_eq_ulps(&self, other: &&'a mut T, ulps: u32) -> bool {
        ApproxEq::approx_eq_ulps(*self, *other, ulps)
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
pub trait Inv: Sized {
    /// Returns the inverse of `m`.
    fn inv(&self) -> Option<Self>;

    /// In-place version of `inverse`.
    fn inv_mut(&mut self) -> bool;
}

/// Trait of objects having a determinant. Typically used by square matrices.
pub trait Det<N> {
    /// Returns the determinant of `m`.
    fn det(&self) -> N;
}

/// Trait of objects which can be transposed.
pub trait Transpose {
    /// Computes the transpose of a matrix.
    fn transpose(&self) -> Self;

    /// In-place version of `transposed`.
    fn transpose_mut(&mut self);
}

/// Traits of objects having an outer product.
pub trait Outer {
    /// Result type of the outer product.
    type OuterProductType;

    /// Computes the outer product: `a * b`
    fn outer(&self, other: &Self) -> Self::OuterProductType;
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
pub trait EigenQR<N, V: Mul<Self, Output = V>>: SquareMat<N, V> {
    /// Computes the eigenvectors and eigenvalues of this matrix.
    fn eigen_qr(&self, eps: &N, niter: usize) -> (Self, V);
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
impl_absolute!(isize);
impl_absolute_id!(u8);
impl_absolute_id!(u16);
impl_absolute_id!(u32);
impl_absolute_id!(u64);
impl_absolute_id!(usize);

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
impl_axpy!(isize);
impl_axpy!(u8);
impl_axpy!(u16);
impl_axpy!(u32);
impl_axpy!(u64);
impl_axpy!(usize);
