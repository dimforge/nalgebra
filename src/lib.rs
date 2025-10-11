/*!
# nalgebra

**nalgebra** is a linear algebra library written for Rust targeting:

* General-purpose linear algebra (still lacks a lot of features…)
* Real-time computer graphics.
* Real-time computer physics.

## Using **nalgebra**
You will need the last stable build of the [rust compiler](https://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
// TODO: replace the * by the latest version.
nalgebra = "*"
```


Most useful functionalities of **nalgebra** are grouped in the root module `nalgebra::`.

However, the recommended way to use **nalgebra** is to import types and traits
explicitly, and call free-functions using the `na::` prefix:

```
#[macro_use]
extern crate approx; // For the macro assert_relative_eq!
extern crate nalgebra as na;
use na::{Vector3, Rotation3};

fn main() {
    let axis  = Vector3::x_axis();
    let angle = 1.57;
    let b     = Rotation3::from_axis_angle(&axis, angle);

    assert_relative_eq!(b.axis().unwrap(), axis);
    assert_relative_eq!(b.angle(), angle);
}
```


## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* A single parametrizable type [`Matrix`] for vectors, (square or rectangular) matrices, and
  slices with dimensions known either at compile-time (using type-level integers) or at runtime.
* Matrices and vectors with compile-time sizes are statically allocated while dynamic ones are
  allocated on the heap.
* Convenient aliases for low-dimensional matrices and vectors: [`Vector1`] to
  [`Vector6`] and [`Matrix1x1`](Matrix1) to [`Matrix6x6`](Matrix6), including rectangular
  matrices like [`Matrix2x5`].
* Points sizes known at compile time, and convenience aliases: [`Point1`] to
  [`Point6`].
* Translation (seen as a transformation that composes by multiplication):
  [`Translation2`], [`Translation3`].
* Rotation matrices: [`Rotation2`], [`Rotation3`].
* Quaternions: [`Quaternion`], [`UnitQuaternion`] (for 3D rotation).
* Unit complex numbers can be used for 2D rotation: [`UnitComplex`].
* Algebraic entities with a norm equal to one: [`Unit<T>`](Unit), e.g., `Unit<Vector3<f32>>`.
* Isometries (translation ⨯ rotation): [`Isometry2`], [`Isometry3`]
* Similarity transformations (translation ⨯ rotation ⨯ uniform scale):
  [`Similarity2`], [`Similarity3`].
* Affine transformations stored as a homogeneous matrix:
  [`Affine2`], [`Affine3`].
* Projective (i.e. invertible) transformations stored as a homogeneous matrix:
  [`Projective2`], [`Projective3`].
* General transformations that does not have to be invertible, stored as a homogeneous matrix:
  [`Transform2`], [`Transform3`].
* 3D projections for computer graphics: [`Perspective3`],
  [`Orthographic3`].
* Matrix factorizations: [`Cholesky`], [`QR`], [`LU`], [`FullPivLU`],
  [`SVD`], [`Schur`], [`Hessenberg`], [`SymmetricEigen`].
* Insertion and removal of rows of columns of a matrix.
*/

#![deny(
    missing_docs,
    nonstandard_style,
    unused_variables,
    unused_mut,
    unused_parens,
    rust_2018_idioms,
    rust_2024_compatibility,
    future_incompatible,
    missing_copy_implementations
)]
#![cfg_attr(not(feature = "rkyv-serialize-no-std"), deny(unused_results))] // TODO: deny this globally once bytecheck stops generating unused results.
#![doc(
    html_favicon_url = "https://nalgebra.rs/img/favicon.ico",
    html_root_url = "https://docs.rs/nalgebra/0.25.0"
)]
#![cfg_attr(not(feature = "std"), no_std)]
// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Generates an appropriate deprecation note with a suggestion for replacement.
///
/// Used for deprecating slice types in various locations throughout the library.
/// See #1076 for more information.
macro_rules! slice_deprecation_note {
    ($replacement:ident) => {
        concat!("Use ", stringify!($replacement),
            r###" instead. See [issue #1076](https://github.com/dimforge/nalgebra/issues/1076) for more information."###)
    }
}

pub(crate) use slice_deprecation_note;

#[cfg(feature = "rand-no-std")]
extern crate rand_package as rand;

#[cfg(feature = "serde-serialize-no-std")]
#[macro_use]
extern crate serde;

#[macro_use]
extern crate approx;
extern crate num_traits as num;

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[cfg_attr(test, macro_use)]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[macro_use]
#[cfg(feature = "io")]
extern crate pest_derive;

pub mod base;
#[cfg(feature = "debug")]
pub mod debug;
pub mod geometry;
#[cfg(feature = "io")]
pub mod io;
pub mod linalg;
#[cfg(feature = "proptest-support")]
pub mod proptest;
#[cfg(feature = "sparse")]
pub mod sparse;
mod third_party;

pub use crate::base::*;
pub use crate::geometry::*;
pub use crate::linalg::*;
#[cfg(feature = "sparse")]
pub use crate::sparse::*;
#[cfg(feature = "std")]
#[deprecated(
    note = "The 'core' module is being renamed to 'base' to avoid conflicts with the 'core' crate."
)]
pub use base as core;

#[cfg(feature = "macros")]
pub use nalgebra_macros::{dmatrix, dvector, matrix, point, stack, vector};

use simba::scalar::SupersetOf;
use std::cmp::{self, Ordering, PartialOrd};

use num::{One, Signed, Zero};

use base::allocator::Allocator;
pub use num_complex::Complex;
pub use simba::scalar::{
    ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField, Field,
    RealField,
};
pub use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};

/// Gets the multiplicative identity element for any type that has one.
///
/// This is a generic function that returns the value `1` for the given type `T`.
/// For scalars (like `f32`, `f64`, `i32`), this returns the numeric value `1`.
/// For matrices, this returns the identity matrix.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Get the multiplicative identity for f64
/// let identity_f64 = na::one::<f64>();
/// assert_eq!(identity_f64, 1.0);
///
/// // Get the multiplicative identity for i32
/// let identity_i32 = na::one::<i32>();
/// assert_eq!(identity_i32, 1);
/// ```
///
/// # See also:
///
/// * [`origin()`](crate::OPoint::origin) - For getting the origin point
/// * [`zero()`] - For getting the additive identity element
#[inline]
pub fn one<T: One>() -> T {
    T::one()
}

/// Gets the additive identity element for any type that has one.
///
/// This is a generic function that returns the value `0` for the given type `T`.
/// For scalars (like `f32`, `f64`, `i32`), this returns the numeric value `0`.
/// For vectors and matrices, this returns a zero-filled vector or matrix.
///
/// # Examples
///
/// ```
/// use nalgebra::{self as na, Vector3, Matrix2};
///
/// // Get zero for f64
/// let zero_f64 = na::zero::<f64>();
/// assert_eq!(zero_f64, 0.0);
///
/// // Get zero vector
/// let zero_vec: Vector3<f32> = na::zero();
/// assert_eq!(zero_vec, Vector3::new(0.0, 0.0, 0.0));
///
/// // Get zero matrix
/// let zero_mat: Matrix2<f64> = na::zero();
/// assert_eq!(zero_mat, Matrix2::zeros());
/// ```
///
/// # See also:
///
/// * [`one()`] - For getting the multiplicative identity element
/// * [`origin()`](crate::OPoint::origin) - For getting the origin point
#[inline]
pub fn zero<T: Zero>() -> T {
    T::zero()
}

/*
 *
 * Ordering
 *
 */
// XXX: this is very naive and could probably be optimized for specific types.
// XXX: also, we might just want to use divisions, but assuming `val` is usually not far from `min`
// or `max`, would it still be more efficient?
/// Wraps a value into the range `[min, max]` using modular arithmetic.
///
/// This function takes a value and "wraps" it into the specified range by repeatedly
/// adding or subtracting the range width until the value falls within bounds.
/// This is useful for periodic values like angles.
///
/// The range must not be empty (i.e., `min` must be less than `max`).
///
/// # Panics
///
/// Panics if `min >= max`.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Wrap a value that's too large
/// let wrapped = na::wrap(15.0, 0.0, 10.0);
/// assert_eq!(wrapped, 5.0);
///
/// // Wrap a value that's too small
/// let wrapped = na::wrap(-5.0, 0.0, 10.0);
/// assert_eq!(wrapped, 5.0);
///
/// // A value already in range stays the same
/// let wrapped = na::wrap(7.0, 0.0, 10.0);
/// assert_eq!(wrapped, 7.0);
///
/// // Useful for wrapping angles
/// let angle = na::wrap(370.0, 0.0, 360.0);
/// assert_eq!(angle, 10.0);
/// ```
#[must_use]
#[inline]
pub fn wrap<T>(mut val: T, min: T, max: T) -> T
where
    T: Copy + PartialOrd + ClosedAddAssign + ClosedSubAssign,
{
    assert!(min < max, "Invalid wrapping bounds.");
    let width = max - min;

    if val < min {
        val += width;

        while val < min {
            val += width
        }
    } else if val > max {
        val -= width;

        while val > max {
            val -= width
        }
    }

    val
}

/// Clamps a value to the interval `[min, max]`.
///
/// This function constrains a value to be within a specified range:
/// - If the value is less than `min`, it returns `min`
/// - If the value is greater than `max`, it returns `max`
/// - Otherwise, it returns the original value
///
/// This is useful for ensuring values stay within valid bounds, such as
/// keeping coordinates within screen bounds or limiting user input.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Clamp a value that's too large
/// let clamped = na::clamp(15.0, 0.0, 10.0);
/// assert_eq!(clamped, 10.0);
///
/// // Clamp a value that's too small
/// let clamped = na::clamp(-5.0, 0.0, 10.0);
/// assert_eq!(clamped, 0.0);
///
/// // A value within range stays the same
/// let clamped = na::clamp(7.0, 0.0, 10.0);
/// assert_eq!(clamped, 7.0);
///
/// // Useful for limiting values
/// let health = na::clamp(150, 0, 100);
/// assert_eq!(health, 100);
/// ```
#[must_use]
#[inline]
pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val > min {
        if val < max { val } else { max }
    } else {
        min
    }
}

/// Returns the maximum of two values.
///
/// This function compares two values and returns the larger one.
/// It requires that the values implement total ordering (the `Ord` trait).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// let result = na::max(5, 10);
/// assert_eq!(result, 10);
///
/// let result = na::max(-3, -1);
/// assert_eq!(result, -1);
///
/// let result = na::max(42, 42);
/// assert_eq!(result, 42);
/// ```
///
/// # See also
///
/// * [`min()`] - For getting the minimum of two values
/// * [`partial_max()`] - For types with partial ordering (like `f32` with NaN)
#[inline]
pub fn max<T: Ord>(a: T, b: T) -> T {
    cmp::max(a, b)
}

/// Returns the minimum of two values.
///
/// This function compares two values and returns the smaller one.
/// It requires that the values implement total ordering (the `Ord` trait).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// let result = na::min(5, 10);
/// assert_eq!(result, 5);
///
/// let result = na::min(-3, -1);
/// assert_eq!(result, -3);
///
/// let result = na::min(42, 42);
/// assert_eq!(result, 42);
/// ```
///
/// # See also
///
/// * [`max()`] - For getting the maximum of two values
/// * [`partial_min()`] - For types with partial ordering (like `f32` with NaN)
#[inline]
pub fn min<T: Ord>(a: T, b: T) -> T {
    cmp::min(a, b)
}

/// The absolute value of `a`.
///
/// Deprecated: Use [`Matrix::abs()`] or [`ComplexField::abs()`] instead.
#[deprecated(note = "use the inherent method `Matrix::abs` or `ComplexField::abs` instead")]
#[inline]
pub fn abs<T: Signed>(a: &T) -> T {
    a.abs()
}

/// Returns the infimum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::inf` instead")]
#[inline]
pub fn inf<T, R: Dim, C: Dim>(a: &OMatrix<T, R, C>, b: &OMatrix<T, R, C>) -> OMatrix<T, R, C>
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<R, C>,
{
    a.inf(b)
}

/// Returns the supremum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::sup` instead")]
#[inline]
pub fn sup<T, R: Dim, C: Dim>(a: &OMatrix<T, R, C>, b: &OMatrix<T, R, C>) -> OMatrix<T, R, C>
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<R, C>,
{
    a.sup(b)
}

/// Returns simultaneously the infimum and supremum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::inf_sup` instead")]
#[inline]
pub fn inf_sup<T, R: Dim, C: Dim>(
    a: &OMatrix<T, R, C>,
    b: &OMatrix<T, R, C>,
) -> (OMatrix<T, R, C>, OMatrix<T, R, C>)
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<R, C>,
{
    a.inf_sup(b)
}

/// Compares two values using partial ordering.
///
/// This function attempts to compare two values and returns `Some(ordering)`
/// if they are comparable, or `None` if they cannot be compared (e.g., when
/// comparing `NaN` with a number).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
/// use std::cmp::Ordering;
///
/// // Comparable values
/// let result = na::partial_cmp(&5.0, &10.0);
/// assert_eq!(result, Some(Ordering::Less));
///
/// // NaN is not comparable
/// let result = na::partial_cmp(&f64::NAN, &10.0);
/// assert_eq!(result, None);
/// ```
#[inline]
pub fn partial_cmp<T: PartialOrd>(a: &T, b: &T) -> Option<Ordering> {
    a.partial_cmp(b)
}

/// Returns `true` if `a` is less than `b` (and they are comparable).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// assert!(na::partial_lt(&5.0, &10.0));
/// assert!(!na::partial_lt(&10.0, &5.0));
/// assert!(!na::partial_lt(&5.0, &5.0));
/// ```
#[inline]
pub fn partial_lt<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.lt(b)
}

/// Returns `true` if `a` is less than or equal to `b` (and they are comparable).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// assert!(na::partial_le(&5.0, &10.0));
/// assert!(na::partial_le(&5.0, &5.0));
/// assert!(!na::partial_le(&10.0, &5.0));
/// ```
#[inline]
pub fn partial_le<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.le(b)
}

/// Returns `true` if `a` is greater than `b` (and they are comparable).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// assert!(na::partial_gt(&10.0, &5.0));
/// assert!(!na::partial_gt(&5.0, &10.0));
/// assert!(!na::partial_gt(&5.0, &5.0));
/// ```
#[inline]
pub fn partial_gt<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.gt(b)
}

/// Returns `true` if `a` is greater than or equal to `b` (and they are comparable).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// assert!(na::partial_ge(&10.0, &5.0));
/// assert!(na::partial_ge(&5.0, &5.0));
/// assert!(!na::partial_ge(&5.0, &10.0));
/// ```
#[inline]
pub fn partial_ge<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.ge(b)
}

/// Returns the minimum of two values if they are comparable.
///
/// This function is like [`min()`] but works with types that have partial ordering,
/// such as floating-point numbers. It returns `Some(&min)` if the values are comparable,
/// or `None` if they cannot be compared (e.g., when one is `NaN`).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Comparable values
/// let result = na::partial_min(&5.0, &10.0);
/// assert_eq!(result, Some(&5.0));
///
/// // Equal values
/// let result = na::partial_min(&5.0, &5.0);
/// assert_eq!(result, Some(&5.0));
///
/// // NaN is not comparable
/// let result = na::partial_min(&f64::NAN, &10.0);
/// assert_eq!(result, None);
/// ```
///
/// # See also
///
/// * [`partial_max()`] - For getting the maximum of two partially ordered values
/// * [`min()`] - For types with total ordering
#[inline]
pub fn partial_min<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Greater => Some(b),
            _ => Some(a),
        }
    } else {
        None
    }
}

/// Returns the maximum of two values if they are comparable.
///
/// This function is like [`max()`] but works with types that have partial ordering,
/// such as floating-point numbers. It returns `Some(&max)` if the values are comparable,
/// or `None` if they cannot be compared (e.g., when one is `NaN`).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Comparable values
/// let result = na::partial_max(&5.0, &10.0);
/// assert_eq!(result, Some(&10.0));
///
/// // Equal values
/// let result = na::partial_max(&5.0, &5.0);
/// assert_eq!(result, Some(&5.0));
///
/// // NaN is not comparable
/// let result = na::partial_max(&f64::NAN, &10.0);
/// assert_eq!(result, None);
/// ```
///
/// # See also
///
/// * [`partial_min()`] - For getting the minimum of two partially ordered values
/// * [`max()`] - For types with total ordering
#[inline]
pub fn partial_max<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Less => Some(b),
            _ => Some(a),
        }
    } else {
        None
    }
}

/// Clamps a value between `min` and `max` for partially ordered types.
///
/// This function is like [`clamp()`] but works with types that have partial ordering,
/// such as floating-point numbers. It returns `Some(&clamped_value)` if the value is
/// comparable to both `min` and `max`, or `None` if any comparison fails (e.g., when
/// dealing with `NaN`).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Clamp a value that's too large
/// let result = na::partial_clamp(&15.0, &0.0, &10.0);
/// assert_eq!(result, Some(&10.0));
///
/// // Clamp a value that's too small
/// let result = na::partial_clamp(&-5.0, &0.0, &10.0);
/// assert_eq!(result, Some(&0.0));
///
/// // Value within range
/// let result = na::partial_clamp(&5.0, &0.0, &10.0);
/// assert_eq!(result, Some(&5.0));
///
/// // NaN is not comparable
/// let result = na::partial_clamp(&f64::NAN, &0.0, &10.0);
/// assert_eq!(result, None);
/// ```
///
/// # See also
///
/// * [`clamp()`] - For types with total ordering
#[inline]
pub fn partial_clamp<'a, T: PartialOrd>(value: &'a T, min: &'a T, max: &'a T) -> Option<&'a T> {
    if let (Some(cmp_min), Some(cmp_max)) = (value.partial_cmp(min), value.partial_cmp(max)) {
        if cmp_min == Ordering::Less {
            Some(min)
        } else if cmp_max == Ordering::Greater {
            Some(max)
        } else {
            Some(value)
        }
    } else {
        None
    }
}

/// Sorts two values in increasing order using partial ordering.
///
/// This function returns a tuple `(smaller, larger)` if the values are comparable,
/// or `None` if they cannot be compared (e.g., when one is `NaN`).
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Sort two comparable values
/// let result = na::partial_sort2(&10.0, &5.0);
/// assert_eq!(result, Some((&5.0, &10.0)));
///
/// // Already sorted values
/// let result = na::partial_sort2(&5.0, &10.0);
/// assert_eq!(result, Some((&5.0, &10.0)));
///
/// // Equal values
/// let result = na::partial_sort2(&5.0, &5.0);
/// assert_eq!(result, Some((&5.0, &5.0)));
///
/// // NaN is not comparable
/// let result = na::partial_sort2(&f64::NAN, &10.0);
/// assert_eq!(result, None);
/// ```
#[inline]
pub fn partial_sort2<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<(&'a T, &'a T)> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Less => Some((a, b)),
            _ => Some((b, a)),
        }
    } else {
        None
    }
}

/*
 *
 * Point operations.
 *
 */
/// Computes the center point (midpoint) between two points.
///
/// This function calculates the point that is exactly halfway between two given points.
/// The result is equivalent to `(p1 + p2) / 2`.
///
/// # Examples
///
/// ```
/// use nalgebra::{Point2, Point3};
/// use nalgebra as na;
///
/// // 2D midpoint
/// let p1 = Point2::new(0.0, 0.0);
/// let p2 = Point2::new(10.0, 10.0);
/// let mid = na::center(&p1, &p2);
/// assert_eq!(mid, Point2::new(5.0, 5.0));
///
/// // 3D midpoint
/// let p1 = Point3::new(1.0, 2.0, 3.0);
/// let p2 = Point3::new(5.0, 6.0, 7.0);
/// let mid = na::center(&p1, &p2);
/// assert_eq!(mid, Point3::new(3.0, 4.0, 5.0));
/// ```
///
/// # See also:
///
/// * [`distance()`] - To compute the distance between two points
/// * [`distance_squared()`] - To compute the squared distance between two points
#[inline]
pub fn center<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> Point<T, D> {
    ((&p1.coords + &p2.coords) * convert::<_, T>(0.5)).into()
}

/// Computes the Euclidean distance between two points.
///
/// This function calculates the straight-line distance between two points in space.
/// For 2D points, this is like measuring with a ruler. For 3D points, it's the
/// straight-line distance through 3D space.
///
/// The distance is computed as the norm (length) of the vector from `p1` to `p2`.
///
/// # Examples
///
/// ```
/// use nalgebra::{Point2, Point3};
/// use nalgebra as na;
///
/// // 2D distance
/// let p1 = Point2::new(0.0, 0.0);
/// let p2 = Point2::new(3.0, 4.0);
/// let dist = na::distance(&p1, &p2);
/// assert_eq!(dist, 5.0); // 3-4-5 triangle
///
/// // 3D distance
/// let p1 = Point3::new(0.0, 0.0, 0.0);
/// let p2 = Point3::new(1.0, 0.0, 0.0);
/// let dist = na::distance(&p1, &p2);
/// assert_eq!(dist, 1.0);
/// ```
///
/// # See also:
///
/// * [`distance_squared()`] - Faster if you only need squared distance
/// * [`center()`] - To compute the midpoint between two points
#[inline]
pub fn distance<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> T::SimdRealField {
    (&p2.coords - &p1.coords).norm()
}

/// Computes the squared Euclidean distance between two points.
///
/// This function is similar to [`distance()`], but returns the squared distance
/// instead of taking the square root. This is more efficient when you only need
/// to compare distances or when the actual distance value isn't required.
///
/// For example, to find the closest point, comparing squared distances is sufficient
/// and avoids the expensive square root computation.
///
/// # Examples
///
/// ```
/// use nalgebra::{Point2, Point3};
/// use nalgebra as na;
///
/// // 2D squared distance
/// let p1 = Point2::new(0.0, 0.0);
/// let p2 = Point2::new(3.0, 4.0);
/// let dist_sq = na::distance_squared(&p1, &p2);
/// assert_eq!(dist_sq, 25.0); // 3² + 4² = 25
///
/// // Use squared distance for comparisons (more efficient)
/// let p1 = Point3::new(0.0, 0.0, 0.0);
/// let p2 = Point3::new(1.0, 1.0, 1.0);
/// let p3 = Point3::new(2.0, 2.0, 2.0);
///
/// // p2 is closer to p1 than p3 is
/// assert!(na::distance_squared(&p1, &p2) < na::distance_squared(&p1, &p3));
/// ```
///
/// # See also:
///
/// * [`distance()`] - When you need the actual distance
/// * [`center()`] - To compute the midpoint between two points
#[inline]
pub fn distance_squared<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> T::SimdRealField {
    (&p2.coords - &p1.coords).norm_squared()
}

/*
 * Cast
 */
/// Converts an object from one type to an equivalent or more general one.
///
/// This function performs a conversion that is guaranteed to succeed because the
/// target type `To` is a superset of the source type `From`. For example, converting
/// from `f32` to `f64`, or from a 2D vector to a 3D vector (by adding a zero component).
///
/// # Type Parameters
///
/// * `From` - The source type to convert from
/// * `To` - The target type to convert to (must be a superset of `From`)
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Convert between numeric types
/// let x: f32 = 3.14;
/// let y: f64 = na::convert(x);
/// assert_eq!(y, 3.14f64);
///
/// // Convert integer to float
/// let a: i32 = 42;
/// let b: f64 = na::convert(a);
/// assert_eq!(b, 42.0);
/// ```
///
/// # See also:
///
/// * [`try_convert()`] - For conversions that might fail (to more specific types)
/// * [`convert_ref()`] - Same as `convert` but takes a reference
/// * [`is_convertible()`] - Check if a conversion is possible
#[inline]
pub fn convert<From, To: SupersetOf<From>>(t: From) -> To {
    To::from_subset(&t)
}

/// Attempts to convert an object to a more specific type.
///
/// This function tries to convert from a more general type to a more specific one.
/// Unlike [`convert()`], this conversion might fail, so it returns an `Option`.
/// For example, converting from `f64` to `f32` might fail if the value is out of range.
///
/// # Returns
///
/// * `Some(value)` if the conversion succeeds
/// * `None` if the conversion fails (e.g., value out of range, precision loss, etc.)
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Successful conversion
/// let x: f64 = 3.14;
/// let y: Option<f32> = na::try_convert(x);
/// assert!(y.is_some());
///
/// // Conversion that might fail with very large values
/// let big: f64 = 1e308;
/// let small: Option<f32> = na::try_convert(big);
/// // May be None if value doesn't fit in f32
///
/// // Float to integer (rounds towards zero)
/// let x: f64 = 42.7;
/// let y: Option<i32> = na::try_convert(x);
/// assert_eq!(y, Some(42));
/// ```
///
/// # See also:
///
/// * [`convert()`] - For conversions guaranteed to succeed (to more general types)
/// * [`try_convert_ref()`] - Same but takes a reference
/// * [`is_convertible()`] - Check if conversion will succeed before converting
#[inline]
pub fn try_convert<From: SupersetOf<To>, To>(t: From) -> Option<To> {
    t.to_subset()
}

/// Checks if a value can be converted to a more specific type without actually converting it.
///
/// This function is useful when you want to check if a conversion is possible before
/// attempting it, avoiding the need to handle the `Option` from [`try_convert()`].
///
/// # Returns
///
/// * `true` if [`try_convert()`] would return `Some(value)`
/// * `false` if [`try_convert()`] would return `None`
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Check if f64 value fits in f32
/// let x: f64 = 3.14;
/// assert!(na::is_convertible::<_, f32>(&x));
///
/// // Check before converting
/// let value: f64 = 100.0;
/// if na::is_convertible::<_, i32>(&value) {
///     let result: i32 = na::try_convert(value).unwrap();
///     assert_eq!(result, 100);
/// }
/// ```
///
/// # See also:
///
/// * [`try_convert()`] - To actually perform the conversion
/// * [`convert()`] - For conversions that always succeed
#[inline]
pub fn is_convertible<From: SupersetOf<To>, To>(t: &From) -> bool {
    t.is_in_subset()
}

/// Converts to a more specific type without checking if the conversion is valid.
///
/// # Safety
///
/// This function performs the same conversion as [`try_convert()`] but skips all
/// validation checks. This can be faster but you must ensure the conversion is valid,
/// otherwise the behavior is unspecified (though not undefined in the Rust sense).
///
/// Use [`is_convertible()`] first if you're not certain the conversion will succeed.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// // Only use when you're certain the conversion is valid
/// let x: f64 = 3.14;
/// if na::is_convertible::<_, f32>(&x) {
///     let y: f32 = na::convert_unchecked(x);
///     assert!((y - 3.14f32).abs() < 1e-6);
/// }
/// ```
///
/// # See also:
///
/// * [`try_convert()`] - Safe version that returns `Option`
/// * [`is_convertible()`] - Check validity before converting
/// * [`convert()`] - For conversions that always succeed
#[inline]
pub fn convert_unchecked<From: SupersetOf<To>, To>(t: From) -> To {
    t.to_subset_unchecked()
}

/// Converts a reference to an object from one type to an equivalent or more general one.
///
/// This is the reference version of [`convert()`]. It takes a reference and produces
/// an owned value of the target type. The conversion is guaranteed to succeed because
/// the target type is a superset of the source type.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// let x: f32 = 3.14;
/// let y: f64 = na::convert_ref(&x);
/// assert_eq!(y, 3.14f64);
///
/// // The original value is still accessible
/// assert_eq!(x, 3.14f32);
/// ```
///
/// # See also:
///
/// * [`convert()`] - Takes ownership instead of a reference
/// * [`try_convert_ref()`] - For conversions that might fail
#[inline]
pub fn convert_ref<From, To: SupersetOf<From>>(t: &From) -> To {
    To::from_subset(t)
}

/// Attempts to convert a reference to an object to a more specific type.
///
/// This is the reference version of [`try_convert()`]. It takes a reference and
/// attempts to produce an owned value of the target type. Returns `None` if the
/// conversion fails.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// let x: f64 = 3.14;
/// let y: Option<f32> = na::try_convert_ref(&x);
/// assert!(y.is_some());
///
/// // The original value is still accessible
/// assert_eq!(x, 3.14f64);
/// ```
///
/// # See also:
///
/// * [`try_convert()`] - Takes ownership instead of a reference
/// * [`convert_ref()`] - For conversions guaranteed to succeed
/// * [`is_convertible()`] - Check if conversion will succeed
#[inline]
pub fn try_convert_ref<From: SupersetOf<To>, To>(t: &From) -> Option<To> {
    t.to_subset()
}

/// Converts a reference to a more specific type without checking if the conversion is valid.
///
/// This is the reference version of [`convert_unchecked()`]. Use with caution!
///
/// # Safety
///
/// You must ensure the conversion is valid. Use [`is_convertible()`] first if you're
/// not certain the conversion will succeed.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
///
/// let x: f64 = 3.14;
/// if na::is_convertible::<_, f32>(&x) {
///     let y: f32 = na::convert_ref_unchecked(&x);
///     assert!((y - 3.14f32).abs() < 1e-6);
/// }
/// ```
///
/// # See also:
///
/// * [`convert_unchecked()`] - Takes ownership instead of a reference
/// * [`try_convert_ref()`] - Safe version that returns `Option`
/// * [`is_convertible()`] - Check validity before converting
#[inline]
pub fn convert_ref_unchecked<From: SupersetOf<To>, To>(t: &From) -> To {
    t.to_subset_unchecked()
}
