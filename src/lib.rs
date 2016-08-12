/*!
# nalgebra

**nalgebra** is a low-dimensional linear algebra library written for Rust targeting:

* low-dimensional general-purpose linear algebra (still lacks a lot of features…).
* real time computer graphics.
* real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org/doc/nalgebra).

## Using **nalgebra**
All the functionality of **nalgebra** is grouped in one place: the root module `nalgebra::`.  This
module re-exports everything and includes free functions for all traits methods performing
out-of-place operations.

Thus, you can import the whole prelude using:

```.ignore
use nalgebra::*;
```

However, the recommended way to use **nalgebra** is to import types and traits explicitly, and call
free-functions using the `na::` prefix:

```.rust
extern crate nalgebra as na;
use na::{Vector3, Rotation3, Rotation};

fn main() {
    let     a = Vector3::new(1.0f64, 1.0, 1.0);
    let mut b = Rotation3::new(na::zero());

    b.append_rotation_mut(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```

## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* Vectors with predefined static sizes: `Vector1`, `Vector2`, `Vector3`, `Vector4`, `Vector5`, `Vector6`.
* Vector with a user-defined static size: `VectorN` (available only with the `generic_sizes` feature).
* Points with static sizes: `Point1`, `Point2`, `Point3`, `Point4`, `Point5`, `Point6`.
* Square matrices with static sizes: `Matrix1`, `Matrix2`, `Matrix3`, `Matrix4`, `Matrix5`, `Matrix6 `.
* Rotation matrices: `Rotation2`, `Rotation3`
* Quaternions: `Quaternion`, `UnitQuaternion`.
* Isometries (translation ⨯ rotation): `Isometry2`, `Isometry3`
* Similarity transformations (translation ⨯ rotation ⨯ uniform scale): `Similarity2`, `Similarity3`.
* 3D projections for computer graphics: `Perspective3`, `PerspectiveMatrix3`, `Orthographic3`, `OrthographicMatrix3`.
* Dynamically sized heap-allocated vector: `DVector`.
* Dynamically sized stack-allocated vectors with a maximum size: `DVector1` to `DVector6`.
* Dynamically sized heap-allocated (square or rectangular) matrix: `DMatrix`.
* Linear algebra and data analysis operators: `Covariance`, `Mean`, `qr`, `cholesky`.
* Almost one trait per functionality: useful for generic programming.


## **nalgebra** in use
Here are some projects using **nalgebra**.
Feel free to add your project to this list if you happen to use **nalgebra**!

* [nphysics](https://github.com/sebcrozet/nphysics): a real-time physics engine.
* [ncollide](https://github.com/sebcrozet/ncollide): a collision detection library.
* [kiss3d](https://github.com/sebcrozet/kiss3d): a minimalistic graphics engine.
* [nrays](https://github.com/sebcrozet/nrays): a ray tracer.
*/

#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![warn(missing_docs)]
#![doc(html_root_url = "http://nalgebra.org/doc")]

extern crate rustc_serialize;
extern crate rand;
extern crate num;

#[cfg(feature="generic_sizes")]
extern crate generic_array;

#[cfg(feature="arbitrary")]
extern crate quickcheck;

use std::cmp;
use std::ops::{Neg, Mul};
use num::{Zero, One};
pub use traits::{
    Absolute,
    AbsoluteRotate,
    ApproxEq,
    Axpy,
    Basis,
    BaseFloat,
    BaseNum,
    Bounded,
    Cast,
    Column,
    ColumnSlice, RowSlice,
    Covariance,
    Cross,
    CrossMatrix,
    Determinant,
    Diagonal,
    Dimension,
    Dot,
    EigenQR,
    Eye,
    FloatPoint,
    FloatVector,
    FromHomogeneous,
    Indexable,
    Inverse,
    Iterable,
    IterableMut,
    Matrix,
    Mean,
    Norm,
    NumPoint,
    NumVector,
    Origin,
    Outer,
    PartialOrder,
    PartialOrdering,
    PointAsVector,
    Repeat,
    Rotate, Rotation, RotationMatrix, RotationWithTranslation, RotationTo,
    Row,
    Shape,
    SquareMatrix,
    ToHomogeneous,
    Transform, Transformation,
    Translate, Translation,
    Transpose,
    UniformSphereSample
};

#[cfg(feature="generic_sizes")]
pub use structs::VectorN;

pub use structs::{
    Identity,
    DMatrix, DMatrix1, DMatrix2,  DMatrix3,  DMatrix4,  DMatrix5,  DMatrix6,
    DVector, DVector1, DVector2,  DVector3,  DVector4,  DVector5,  DVector6,
    Isometry2, Isometry3,
    Similarity2, Similarity3,
    Matrix1, Matrix2, Matrix3, Matrix4,
    Matrix5, Matrix6,
    Rotation2, Rotation3,
    Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
    Point1, Point2, Point3, Point4, Point5, Point6,
    Perspective3, PerspectiveMatrix3,
    Orthographic3, OrthographicMatrix3,
    Quaternion, UnitQuaternion,
    Unit
};

pub use linalg::{
    qr,
    householder_matrix,
    cholesky,
    hessenberg
};

mod structs;
mod traits;
mod linalg;
mod macros;

// mod lower_triangular;
// mod chol;

/// Change the input value to ensure it is on the range `[min, max]`.
#[inline]
pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val > min {
        if val < max {
            val
        }
        else {
            max
        }
    }
    else {
        min
    }
}

/// Same as `cmp::max`.
#[inline]
pub fn max<T: Ord>(a: T, b: T) -> T {
    cmp::max(a, b)
}

/// Same as `cmp::min`.
#[inline]
pub fn min<T: Ord>(a: T, b: T) -> T {
    cmp::min(a, b)
}

/// Returns the infimum of `a` and `b`.
#[inline]
pub fn inf<T: PartialOrder>(a: &T, b: &T) -> T {
    PartialOrder::inf(a, b)
}

/// Returns the supremum of `a` and `b`.
#[inline]
pub fn sup<T: PartialOrder>(a: &T, b: &T) -> T {
    PartialOrder::sup(a, b)
}

/// Compare `a` and `b` using a partial ordering relation.
#[inline]
pub fn partial_cmp<T: PartialOrder>(a: &T, b: &T) -> PartialOrdering {
    PartialOrder::partial_cmp(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a < b`.
#[inline]
pub fn partial_lt<T: PartialOrder>(a: &T, b: &T) -> bool {
    PartialOrder::partial_lt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a <= b`.
#[inline]
pub fn partial_le<T: PartialOrder>(a: &T, b: &T) -> bool {
    PartialOrder::partial_le(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a > b`.
#[inline]
pub fn partial_gt<T: PartialOrder>(a: &T, b: &T) -> bool {
    PartialOrder::partial_gt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a >= b`.
#[inline]
pub fn partial_ge<T: PartialOrder>(a: &T, b: &T) -> bool {
    PartialOrder::partial_ge(a, b)
}

/// Return the minimum of `a` and `b` if they are comparable.
#[inline]
pub fn partial_min<'a, T: PartialOrder>(a: &'a T, b: &'a T) -> Option<&'a T> {
    PartialOrder::partial_min(a, b)
}

/// Return the maximum of `a` and `b` if they are comparable.
#[inline]
pub fn partial_max<'a, T: PartialOrder>(a: &'a T, b: &'a T) -> Option<&'a T> {
    PartialOrder::partial_max(a, b)
}

/// Clamp `value` between `min` and `max`. Returns `None` if `value` is not comparable to
/// `min` or `max`.
#[inline]
pub fn partial_clamp<'a, T: PartialOrder>(value: &'a T, min: &'a T, max: &'a T) -> Option<&'a T> {
    PartialOrder::partial_clamp(value, min, max)
}

//
//
// Constructors
//
//

/// Create a special identity object.
///
/// Same as `Identity::new()`.
#[inline]
pub fn identity() -> Identity {
    Identity::new()
}

/// Create a zero-valued value.
///
/// This is the same as `std::num::zero()`.
#[inline]
pub fn zero<T: Zero>() -> T {
    Zero::zero()
}

/// Tests is a value is iqual to zero.
#[inline]
pub fn is_zero<T: Zero>(val: &T) -> bool {
    val.is_zero()
}

/// Create a one-valued value.
///
/// This is the same as `std::num::one()`.
#[inline]
pub fn one<T: One>() -> T {
    One::one()
}

//
//
// Geometry
//
//

/// Returns the trivial origin of an affine space.
#[inline]
pub fn origin<P: Origin>() -> P {
    Origin::origin()
}

/// Returns the center of two points.
#[inline]
pub fn center<N: BaseFloat, P: FloatPoint<N>>(a: &P, b: &P) -> P
        where <P as PointAsVector>::Vector: Norm<NormType = N> {
    (*a + b.to_vector()) / ::cast(2.0)
}

/*
 * FloatPoint
 */
/// Returns the distance between two points.
#[inline]
pub fn distance<N: BaseFloat, P: FloatPoint<N>>(a: &P, b: &P) -> N where <P as PointAsVector>::Vector: Norm<NormType = N> {
    a.distance(b)
}

/// Returns the squared distance between two points.
#[inline]
pub fn distance_squared<N: BaseFloat, P: FloatPoint<N>>(a: &P, b: &P) -> N 
        where <P as PointAsVector>::Vector: Norm<NormType = N> {
    a.distance_squared(b)
}

/*
 * Translation<V>
 */

/// Gets the translation applicable by `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Isometry3};
///
/// fn main() {
///     let t     = Isometry3::new(Vector3::new(1.0f64, 1.0, 1.0), na::zero());
///     let trans = na::translation(&t);
///
///     assert!(trans == Vector3::new(1.0, 1.0, 1.0));
/// }
/// ```
#[inline]
pub fn translation<V, M: Translation<V>>(m: &M) -> V {
    m.translation()
}

/// Gets the inverse translation applicable by `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Isometry3};
///
/// fn main() {
///     let t      = Isometry3::new(Vector3::new(1.0f64, 1.0, 1.0), na::zero());
///     let itrans = na::inverse_translation(&t);
///
///     assert!(itrans == Vector3::new(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline]
pub fn inverse_translation<V, M: Translation<V>>(m: &M) -> V {
    m.inverse_translation()
}

/// Applies the translation `v` to a copy of `m`.
#[inline]
pub fn append_translation<V, M: Translation<V>>(m: &M, v: &V) -> M {
    Translation::append_translation(m, v)
}

/*
 * Translate<P>
 */

/// Applies a translation to a point.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Point3, Vector3, Isometry3};
///
/// fn main() {
///     let t  = Isometry3::new(Vector3::new(1.0f64, 1.0, 1.0), na::zero());
///     let p  = Point3::new(2.0, 2.0, 2.0);
///
///     let tp = na::translate(&t, &p);
///
///     assert!(tp == Point3::new(3.0, 3.0, 3.0))
/// }
/// ```
#[inline]
pub fn translate<P, M: Translate<P>>(m: &M, p: &P) -> P {
    m.translate(p)
}

/// Applies an inverse translation to a point.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Point3, Vector3, Isometry3};
///
/// fn main() {
///     let t  = Isometry3::new(Vector3::new(1.0f64, 1.0, 1.0), na::zero());
///     let p  = Point3::new(2.0, 2.0, 2.0);
///
///     let tp = na::inverse_translate(&t, &p);
///
///     assert!(na::approx_eq(&tp, &Point3::new(1.0, 1.0, 1.0)))
/// }
#[inline]
pub fn inverse_translate<P, M: Translate<P>>(m: &M, p: &P) -> P {
    m.inverse_translate(p)
}

/*
 * Rotation<V>
 */

/// Gets the rotation applicable by `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Rotation3};
///
/// fn main() {
///     let t = Rotation3::new(Vector3::new(1.0f64, 1.0, 1.0));
///
///     assert!(na::approx_eq(&na::rotation(&t), &Vector3::new(1.0, 1.0, 1.0)));
/// }
/// ```
#[inline]
pub fn rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.rotation()
}


/// Gets the inverse rotation applicable by `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Rotation3};
///
/// fn main() {
///     let t = Rotation3::new(Vector3::new(1.0f64, 1.0, 1.0));
///
///     assert!(na::approx_eq(&na::inverse_rotation(&t), &Vector3::new(-1.0, -1.0, -1.0)));
/// }
/// ```
#[inline]
pub fn inverse_rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.inverse_rotation()
}

// FIXME: this example is a bit shity
/// Applies the rotation `v` to a copy of `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Rotation3};
///
/// fn main() {
///     let t  = Rotation3::new(Vector3::new(0.0f64, 0.0, 0.0));
///     let v  = Vector3::new(1.0, 1.0, 1.0);
///     let rt = na::append_rotation(&t, &v);
///
///     assert!(na::approx_eq(&na::rotation(&rt), &Vector3::new(1.0, 1.0, 1.0)))
/// }
/// ```
#[inline]
pub fn append_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    Rotation::append_rotation(m, v)
}

// FIXME: this example is a bit shity
/// Pre-applies the rotation `v` to a copy of `m`.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{Vector3, Rotation3};
///
/// fn main() {
///     let t  = Rotation3::new(Vector3::new(0.0f64, 0.0, 0.0));
///     let v  = Vector3::new(1.0, 1.0, 1.0);
///     let rt = na::prepend_rotation(&t, &v);
///
///     assert!(na::approx_eq(&na::rotation(&rt), &Vector3::new(1.0, 1.0, 1.0)))
/// }
/// ```
#[inline]
pub fn prepend_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    Rotation::prepend_rotation(m, v)
}

/*
 * Rotate<V>
 */

/// Applies a rotation to a vector.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{BaseFloat, Rotation3, Vector3};
///
/// fn main() {
///     let t  = Rotation3::new(Vector3::new(0.0f64, 0.0, 0.5 * <f64 as BaseFloat>::pi()));
///     let v  = Vector3::new(1.0, 0.0, 0.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(na::approx_eq(&tv, &Vector3::new(0.0, 1.0, 0.0)))
/// }
/// ```
#[inline]
pub fn rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.rotate(v)
}


/// Applies an inverse rotation to a vector.
///
/// ```rust
/// extern crate nalgebra as na;
/// use na::{BaseFloat, Rotation3, Vector3};
///
/// fn main() {
///     let t  = Rotation3::new(Vector3::new(0.0f64, 0.0, 0.5 * <f64 as BaseFloat>::pi()));
///     let v  = Vector3::new(1.0, 0.0, 0.0);
///
///     let tv = na::inverse_rotate(&t, &v);
///
///     assert!(na::approx_eq(&tv, &Vector3::new(0.0, -1.0, 0.0)))
/// }
/// ```
#[inline]
pub fn inverse_rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.inverse_rotate(v)
}

/*
 * RotationWithTranslation<LV, AV>
 */

/// Rotates a copy of `m` by `amount` using `center` as the pivot point.
#[inline]
pub fn append_rotation_wrt_point<LV: Neg<Output = LV> + Copy,
                                 AV,
                                 M: RotationWithTranslation<LV, AV>>(
                                 m:      &M,
                                 amount: &AV,
                                 center: &LV) -> M {
    RotationWithTranslation::append_rotation_wrt_point(m, amount, center)
}

/// Rotates a copy of `m` by `amount` using `m.translation()` as the pivot point.
#[inline]
pub fn append_rotation_wrt_center<LV: Neg<Output = LV> + Copy,
                                  AV,
                                  M: RotationWithTranslation<LV, AV>>(
                                  m:      &M,
                                  amount: &AV) -> M {
    RotationWithTranslation::append_rotation_wrt_center(m, amount)
}

/*
 * RotationTo
 */
/// Computes the angle of the rotation needed to transfom `a` to `b`.
#[inline]
pub fn angle_between<V: RotationTo>(a: &V, b: &V) -> V::AngleType {
    a.angle_to(b)
}

/// Computes the rotation needed to transform `a` to `b`.
#[inline]
pub fn rotation_between<V: RotationTo>(a: &V, b: &V) -> V::DeltaRotationType {
    a.rotation_to(b)
}

/*
 * RotationMatrix<LV, AV, R>
 */

/// Builds a rotation matrix from `r`.
#[inline]
pub fn to_rotation_matrix<N, LV, AV, R, M>(r: &R) -> M
    where R: RotationMatrix<N, LV, AV, Output = M>,
          M: SquareMatrix<N, LV> + Rotation<AV> + Copy,
          LV: Mul<M, Output = LV>
{
    // FIXME: rust-lang/rust#20413
    r.to_rotation_matrix()
}

/*
 * AbsoluteRotate<V>
 */

/// Applies a rotation using the absolute values of its components.
#[inline]
pub fn absolute_rotate<V, M: AbsoluteRotate<V>>(m: &M, v: &V) -> V {
    m.absolute_rotate(v)
}

/*
 * Transformation<T>
 */

/// Gets the transformation applicable by `m`.
#[inline]
pub fn transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.transformation()
}

/// Gets the inverse transformation applicable by `m`.
#[inline]
pub fn inverse_transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.inverse_transformation()
}

/// Gets a transformed copy of `m`.
#[inline]
pub fn append_transformation<T, M: Transformation<T>>(m: &M, t: &T) -> M {
    Transformation::append_transformation(m, t)
}

/*
 * Transform<V>
 */

/// Applies a transformation to a vector.
#[inline]
pub fn transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    m.transform(v)
}

/// Applies an inverse transformation to a vector.
#[inline]
pub fn inverse_transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    m.inverse_transform(v)
}

/*
 * Dot<N>
 */

/// Computes the dot product of two vectors.
#[inline]
pub fn dot<V: Dot<N>, N>(a: &V, b: &V) -> N {
    Dot::dot(a, b)
}

/*
 * Norm<N>
 */

/// Computes the L2 norm of a vector.
#[inline]
pub fn norm<V: Norm>(v: &V) -> V::NormType {
    Norm::norm(v)
}

/// Computes the squared L2 norm of a vector.
#[inline]
pub fn norm_squared<V: Norm>(v: &V) -> V::NormType {
    Norm::norm_squared(v)
}

/// Gets the normalized version of a vector.
#[inline]
pub fn normalize<V: Norm>(v: &V) -> V {
    Norm::normalize(v)
}

/// Gets the normalized version of a vector or `None` if its norm is smaller than `min_norm`.
#[inline]
pub fn try_normalize<V: Norm>(v: &V, min_norm: V::NormType) -> Option<V> {
    Norm::try_normalize(v, min_norm)
}

/*
 * Determinant<N>
 */
/// Computes the determinant of a square matrix.
#[inline]
pub fn determinant<M: Determinant<N>, N>(m: &M) -> N {
    Determinant::determinant(m)
}

/*
 * Cross<V>
 */

/// Computes the cross product of two vectors.
#[inline]
pub fn cross<LV: Cross>(a: &LV, b: &LV) -> LV::CrossProductType {
    Cross::cross(a, b)
}

/*
 * CrossMatrix<M>
 */

/// Given a vector, computes the matrix which, when multiplied by another vector, computes a cross
/// product.
#[inline]
pub fn cross_matrix<V: CrossMatrix<M>, M>(v: &V) -> M {
    CrossMatrix::cross_matrix(v)
}

/*
 * ToHomogeneous<U>
 */

/// Converts a matrix or vector to homogeneous coordinates.
#[inline]
pub fn to_homogeneous<M: ToHomogeneous<Res>, Res>(m: &M) -> Res {
    ToHomogeneous::to_homogeneous(m)
}

/*
 * FromHomogeneous<U>
 */

/// Converts a matrix or vector from homogeneous coordinates.
///
/// w-normalization is appied.
#[inline]
pub fn from_homogeneous<M, Res: FromHomogeneous<M>>(m: &M) -> Res {
    FromHomogeneous::from(m)
}

/*
 * UniformSphereSample
 */

/// Samples the unit sphere living on the dimension as the samples types.
///
/// The number of sampling point is implementation-specific. It is always uniform.
#[inline]
pub fn sample_sphere<V: UniformSphereSample, F: FnMut(V)>(f: F) {
    UniformSphereSample::sample(f)
}

//
//
// Operations
//
//

/*
 * AproxEq<N>
 */
/// Tests approximate equality.
#[inline]
pub fn approx_eq<T: ApproxEq<N>, N>(a: &T, b: &T) -> bool {
    ApproxEq::approx_eq(a, b)
}

/// Tests approximate equality using a custom epsilon.
#[inline]
pub fn approx_eq_eps<T: ApproxEq<N>, N>(a: &T, b: &T, eps: &N) -> bool {
    ApproxEq::approx_eq_eps(a, b, eps)
}


/*
 * Absolute<A>
 */

/// Computes a component-wise absolute value.
#[inline]
pub fn abs<M: Absolute<Res>, Res>(m: &M) -> Res {
    Absolute::abs(m)
}

/*
 * Inverse
 */

/// Gets an inverted copy of a matrix.
#[inline]
pub fn inverse<M: Inverse>(m: &M) -> Option<M> {
    Inverse::inverse(m)
}

/*
 * Transpose
 */

/// Gets a transposed copy of a matrix.
#[inline]
pub fn transpose<M: Transpose>(m: &M) -> M {
    Transpose::transpose(m)
}

/*
 * Outer<M>
 */

/// Computes the outer product of two vectors.
#[inline]
pub fn outer<V: Outer>(a: &V, b: &V) -> V::OuterProductType {
    Outer::outer(a, b)
}

/*
 * Covariance<M>
 */

/// Computes the covariance of a set of observations.
#[inline]
pub fn covariance<M: Covariance<Res>, Res>(observations: &M) -> Res {
    Covariance::covariance(observations)
}

/*
 * Mean<N>
 */

/// Computes the mean of a set of observations.
#[inline]
pub fn mean<N, M: Mean<N>>(observations: &M) -> N {
    Mean::mean(observations)
}

/*
 * EigenQR<N, V>
 */
/// Computes the eigenvalues and eigenvectors of a square matrix usin the QR algorithm.
#[inline]
pub fn eigen_qr<N, V, M>(m: &M, eps: &N, niter: usize) -> (M, V)
    where V: Mul<M, Output = V>,
          M: EigenQR<N, V> {
    EigenQR::eigen_qr(m, eps, niter)
}

//
//
// Structure
//
//

/*
 * Eye
 */
/// Construct the identity matrix for a given dimension
#[inline]
pub fn new_identity<M: Eye>(dimension: usize) -> M {
    Eye::new_identity(dimension)
}


/*
 * Repeat
 */
/// Create an object by repeating a value.
///
/// Same as `Identity::new()`.
#[inline]
pub fn repeat<N, T: Repeat<N>>(val: N) -> T {
    Repeat::repeat(val)
}

/*
 * Basis
 */

/// Computes the canonical basis for a given dimension.
#[inline]
pub fn canonical_basis<V: Basis, F: FnMut(V) -> bool>(f: F) {
    Basis::canonical_basis(f)
}

/// Computes the basis of the orthonormal subspace of a given vector.
#[inline]
pub fn orthonormal_subspace_basis<V: Basis, F: FnMut(V) -> bool>(v: &V, f: F) {
    Basis::orthonormal_subspace_basis(v, f)
}

/// Gets the (0-based) i-th element of the canonical basis of V.
#[inline]
pub fn canonical_basis_element<V: Basis>(i: usize) -> Option<V> {
    Basis::canonical_basis_element(i)
}

/*
 * Row<R>
 */

/*
 * Column<C>
 */

/*
 * Diagonal<V>
 */
/// Gets the diagonal of a square matrix.
#[inline]
pub fn diagonal<M: Diagonal<V>, V>(m: &M) -> V {
    m.diagonal()
}

/*
 * Dimension
 */
/// Gets the dimension an object lives in.
///
/// Same as `Dimension::dimension::(None::<V>)`.
#[inline]
pub fn dimension<V: Dimension>() -> usize {
    Dimension::dimension(None::<V>)
}

/// Gets the indexable range of an object.
#[inline]
pub fn shape<V: Shape<I>, I>(v: &V) -> I {
    v.shape()
}

/*
 * Cast<T>
 */
/// Converts an object from one type to another.
///
/// For primitive types, this is the same as the `as` keywords.
/// The following properties are preserved by a cast:
///
/// * Type-level geometric invariants cannot be broken (eg. a cast from Rotation3<f64> to Rotation3<i64> is
/// not possible)
/// * A cast to a type with more type-level invariants cannot be done (eg. a cast from Matrix<f64> to
/// Rotation3<f64> is not possible)
/// * For primitive types an unbounded cast is done using the `as` keyword (this is different from
/// the standard library which makes bound-checking to ensure eg. that a i64 is not out of the
/// range of an i32 when a cast from i64 to i32 is done).
/// * A cast does not affect the dimension of an algebraic object. Note that this prevents an
/// isometric transform to be cast to a raw matrix. Use `to_homogeneous` for that special purpose.
#[inline]
pub fn cast<T, U: Cast<T>>(t: T) -> U {
    Cast::from(t)
}

/*
 * Indexable
 */
