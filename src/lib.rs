/*!
# nalgebra

**nalgebra** is a low-dimensional linear algebra library written for Rust targeting:

* general-purpose linear algebra (still lacks a lot of features…).
* real time computer graphics.
* real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org).

## Using **nalgebra**
All the functionality of **nalgebra** is grouped in one place: the root module `nalgebra::`.
This module re-exports everything and includes free functions for all traits methods doing
out-of-place modifications.

* You can import the whole prelude using:

```.ignore
use nalgebra::*;
```

The preferred way to use **nalgebra** is to import types and traits explicitly, and call
free-functions using the `na::` prefix:

```.rust
extern crate "nalgebra" as na;
use na::{Vec3, Rot3, Rotation};

fn main() {
    let     a = Vec3::new(1.0f64, 1.0, 1.0);
    let mut b = Rot3::new(na::zero());

    b.append_rotation(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```

## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* Vectors with static sizes: `Vec0`, `Vec1`, `Vec2`, `Vec3`, `Vec4`, `Vec5`, `Vec6`.
* Points with static sizes: `Pnt0`, `Pnt1`, `Pnt2`, `Pnt3`, `Pnt4`, `Pnt5`, `Pnt6`.
* Square matrices with static sizes: `Mat1`, `Mat2`, `Mat3`, `Mat4`, `Mat5`, `Mat6 `.
* Rotation matrices: `Rot2`, `Rot3`, `Rot4`.
* Quaternions: `Quat`, `UnitQuat`.
* Isometries: `Iso2`, `Iso3`, `Iso4`.
* 3D projections for computer graphics: `Persp3`, `PerspMat3`, `Ortho3`, `OrthoMat3`.
* Dynamically sized vector: `DVec`.
* Dynamically sized (square or rectangular) matrix: `DMat`.
* A few methods for data analysis: `Cov`, `Mean`.
* Almost one trait per functionality: useful for generic programming.
* Operator overloading using multidispatch.

## Compilation
You will need the last nightly build of the [rust compiler](http://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```.ignore
[dependencies.nalgebra]
git = "https://github.com/sebcrozet/nalgebra"
```


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
#![feature(macro_rules)]
#![feature(globs)]
#![doc(html_root_url = "http://nalgebra.org/doc")]

extern crate serialize;

#[cfg(test)]
extern crate test;

use std::cmp;
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
    Col,
    ColSlice, RowSlice,
    Cov,
    Cross,
    CrossMatrix,
    Det,
    Diag,
    Dim,
    Dot,
    EigenQR,
    Eye,
    FloatPnt,
    FloatVec,
    FromHomogeneous,
    Indexable,
    Inv,
    Iterable,
    IterableMut,
    LMul,
    Mat,
    Mean,
    Norm,
    NumPnt,
    NumVec,
    One,
    Orig,
    Outer,
    POrd,
    POrdering,
    PntAsVec,
    RMul,
    Rotate, Rotation, RotationMatrix, RotationWithTranslation,
    Row,
    ScalarAdd, ScalarSub,
    ScalarMul, ScalarDiv,
    Shape,
    SquareMat,
    ToHomogeneous,
    Transform, Transformation,
    Translate, Translation,
    Transpose,
    UniformSphereSample,
    VecAsPnt,
    Zero
};

pub use structs::{
    Identity,
    DMat,
    DVec, DVec1, DVec2,  DVec3,  DVec4,  DVec5,  DVec6,
    Iso2, Iso3, Iso4,
    Mat1, Mat2, Mat3, Mat4,
    Mat5, Mat6,
    Rot2, Rot3, Rot4,
    Vec0, Vec1, Vec2, Vec3, Vec4, Vec5, Vec6,
    Pnt0, Pnt1, Pnt2, Pnt3, Pnt4, Pnt5, Pnt6,
    Persp3, PerspMat3,
    Ortho3, OrthoMat3,
    Quat, UnitQuat
};

pub use linalg::{
    qr,
    householder_matrix
};

mod structs;
mod traits;
mod linalg;
mod macros;

// mod lower_triangular;
// mod chol;

/// Change the input value to ensure it is on the range `[min, max]`.
#[inline(always)]
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
#[inline(always)]
pub fn max<T: Ord>(a: T, b: T) -> T {
    cmp::max(a, b)
}

/// Same as `cmp::min`.
#[inline(always)]
pub fn min<T: Ord>(a: T, b: T) -> T {
    cmp::min(a, b)
}

/// Returns the infimum of `a` and `b`.
#[inline(always)]
pub fn inf<T: POrd>(a: &T, b: &T) -> T {
    POrd::inf(a, b)
}

/// Returns the supremum of `a` and `b`.
#[inline(always)]
pub fn sup<T: POrd>(a: &T, b: &T) -> T {
    POrd::sup(a, b)
}

/// Compare `a` and `b` using a partial ordering relation.
#[inline(always)]
pub fn partial_cmp<T: POrd>(a: &T, b: &T) -> POrdering {
    POrd::partial_cmp(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a < b`.
#[inline(always)]
pub fn partial_lt<T: POrd>(a: &T, b: &T) -> bool {
    POrd::partial_lt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a <= b`.
#[inline(always)]
pub fn partial_le<T: POrd>(a: &T, b: &T) -> bool {
    POrd::partial_le(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a > b`.
#[inline(always)]
pub fn partial_gt<T: POrd>(a: &T, b: &T) -> bool {
    POrd::partial_gt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a >= b`.
#[inline(always)]
pub fn partial_ge<T: POrd>(a: &T, b: &T) -> bool {
    POrd::partial_ge(a, b)
}

/// Return the minimum of `a` and `b` if they are comparable.
#[inline(always)]
pub fn partial_min<'a, T: POrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    POrd::partial_min(a, b)
}

/// Return the maximum of `a` and `b` if they are comparable.
#[inline(always)]
pub fn partial_max<'a, T: POrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    POrd::partial_max(a, b)
}

/// Clamp `value` between `min` and `max`. Returns `None` if `value` is not comparable to
/// `min` or `max`.
#[inline(always)]
pub fn partial_clamp<'a, T: POrd>(value: &'a T, min: &'a T, max: &'a T) -> Option<&'a T> {
    POrd::partial_clamp(value, min, max)
}

//
//
// Constructors
//
//

/// Create a special identity object.
///
/// Same as `Identity::new()`.
#[inline(always)]
pub fn identity() -> Identity {
    Identity::new()
}

/// Create a zero-valued value.
///
/// This is the same as `std::num::zero()`.
#[inline(always)]
pub fn zero<T: Zero>() -> T {
    Zero::zero()
}

/// Tests is a value is iqual to zero.
#[inline(always)]
pub fn is_zero<T: Zero>(val: &T) -> bool {
    val.is_zero()
}

/// Create a one-valued value.
///
/// This is the same as `std::num::one()`.
#[inline(always)]
pub fn one<T: One>() -> T {
    One::one()
}

//
//
// Geometry
//
//

/// Returns the trivial origin of an affine space.
#[inline(always)]
pub fn orig<P: Orig>() -> P {
    Orig::orig()
}

/// Returns the center of two points.
#[inline]
pub fn center<N: BaseFloat, P: FloatPnt<N, V>, V: Copy>(a: &P, b: &P) -> P {
    let _2 = one::<N>() + one();
    (*a + *b.as_vec()) / _2
}

/*
 * FloatPnt
 */
/// Returns the distance between two points.
#[inline(always)]
pub fn dist<N: BaseFloat, P: FloatPnt<N, V>, V: Norm<N>>(a: &P, b: &P) -> N {
    a.dist(b)
}

/// Returns the squared distance between two points.
#[inline(always)]
pub fn sqdist<N: BaseFloat, P: FloatPnt<N, V>, V: Norm<N>>(a: &P, b: &P) -> N {
    a.sqdist(b)
}

/*
 * Perspective
 */
/// Computes a projection matrix given the frustrum near plane width, height, the field of
/// view, and the distance to the clipping planes (`znear` and `zfar`).
#[deprecated = "Use `Persp3::new(width / height, fov, znear, zfar).as_mat()` instead"]
pub fn perspective3d<N: BaseFloat + Cast<f64> + Zero + One>(width: N, height: N, fov: N, znear: N, zfar: N) -> Mat4<N> {
    let aspect = width / height;

    let _1: N = one();
    let sy    = _1 / (fov * cast(0.5)).tan();
    let sx    = -sy / aspect;
    let sz    = -(zfar + znear) / (znear - zfar);
    let tz    = zfar * znear * cast(2.0) / (znear - zfar);

    Mat4::new(
        sx,     zero(), zero(), zero(),
        zero(), sy,     zero(), zero(),
        zero(), zero(), sz,     tz,
        zero(), zero(), one(),  zero())
}

/*
 * Translation<V>
 */

/// Gets the translation applicable by `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Iso3};
///
/// fn main() {
///     let t     = Iso3::new(Vec3::new(1.0f64, 1.0, 1.0), na::zero());
///     let trans = na::translation(&t);
///
///     assert!(trans == Vec3::new(1.0, 1.0, 1.0));
/// }
/// ```
#[inline(always)]
pub fn translation<V, M: Translation<V>>(m: &M) -> V {
    m.translation()
}

/// Gets the inverse translation applicable by `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Iso3};
///
/// fn main() {
///     let t      = Iso3::new(Vec3::new(1.0f64, 1.0, 1.0), na::zero());
///     let itrans = na::inv_translation(&t);
///
///     assert!(itrans == Vec3::new(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline(always)]
pub fn inv_translation<V, M: Translation<V>>(m: &M) -> V {
    m.inv_translation()
}

/// Applies the translation `v` to a copy of `m`.
#[inline(always)]
pub fn append_translation<V, M: Translation<V>>(m: &M, v: &V) -> M {
    Translation::append_translation_cpy(m, v)
}

/*
 * Translate<P>
 */

/// Applies a translation to a point.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Pnt3, Vec3, Iso3};
///
/// fn main() {
///     let t  = Iso3::new(Vec3::new(1.0f64, 1.0, 1.0), na::zero());
///     let p  = Pnt3::new(2.0, 2.0, 2.0);
///
///     let tp = na::translate(&t, &p);
///
///     assert!(tp == Pnt3::new(3.0, 3.0, 3.0))
/// }
/// ```
#[inline(always)]
pub fn translate<P, M: Translate<P>>(m: &M, p: &P) -> P {
    m.translate(p)
}

/// Applies an inverse translation to a point.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Pnt3, Vec3, Iso3};
///
/// fn main() {
///     let t  = Iso3::new(Vec3::new(1.0f64, 1.0, 1.0), na::zero());
///     let p  = Pnt3::new(2.0, 2.0, 2.0);
///
///     let tp = na::inv_translate(&t, &p);
///
///     assert!(na::approx_eq(&tp, &Pnt3::new(1.0, 1.0, 1.0)))
/// }
#[inline(always)]
pub fn inv_translate<P, M: Translate<P>>(m: &M, p: &P) -> P {
    m.inv_translate(p)
}

/*
 * Rotation<V>
 */

/// Gets the rotation applicable by `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Rot3};
///
/// fn main() {
///     let t = Rot3::new(Vec3::new(1.0f64, 1.0, 1.0));
///
///     assert!(na::approx_eq(&na::rotation(&t), &Vec3::new(1.0, 1.0, 1.0)));
/// }
/// ```
#[inline(always)]
pub fn rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.rotation()
}


/// Gets the inverse rotation applicable by `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Rot3};
///
/// fn main() {
///     let t = Rot3::new(Vec3::new(1.0f64, 1.0, 1.0));
///
///     assert!(na::approx_eq(&na::inv_rotation(&t), &Vec3::new(-1.0, -1.0, -1.0)));
/// }
/// ```
#[inline(always)]
pub fn inv_rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.inv_rotation()
}

// FIXME: this example is a bit shity
/// Applies the rotation `v` to a copy of `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Rot3};
///
/// fn main() {
///     let t  = Rot3::new(Vec3::new(0.0f64, 0.0, 0.0));
///     let v  = Vec3::new(1.0, 1.0, 1.0);
///     let rt = na::append_rotation(&t, &v);
///
///     assert!(na::approx_eq(&na::rotation(&rt), &Vec3::new(1.0, 1.0, 1.0)))
/// }
/// ```
#[inline(always)]
pub fn append_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    Rotation::append_rotation_cpy(m, v)
}

// FIXME: this example is a bit shity
/// Pre-applies the rotation `v` to a copy of `m`.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{Vec3, Rot3};
///
/// fn main() {
///     let t  = Rot3::new(Vec3::new(0.0f64, 0.0, 0.0));
///     let v  = Vec3::new(1.0, 1.0, 1.0);
///     let rt = na::prepend_rotation(&t, &v);
///
///     assert!(na::approx_eq(&na::rotation(&rt), &Vec3::new(1.0, 1.0, 1.0)))
/// }
/// ```
#[inline(always)]
pub fn prepend_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    Rotation::prepend_rotation_cpy(m, v)
}

/*
 * Rotate<V>
 */

/// Applies a rotation to a vector.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{BaseFloat, Rot3, Vec3};
///
/// fn main() {
///     let t  = Rot3::new(Vec3::new(0.0f64, 0.0, 0.5 * BaseFloat::pi()));
///     let v  = Vec3::new(1.0, 0.0, 0.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(na::approx_eq(&tv, &Vec3::new(0.0, 1.0, 0.0)))
/// }
/// ```
#[inline(always)]
pub fn rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.rotate(v)
}


/// Applies an inverse rotation to a vector.
///
/// ```rust
/// extern crate "nalgebra" as na;
/// use na::{BaseFloat, Rot3, Vec3};
///
/// fn main() {
///     let t  = Rot3::new(Vec3::new(0.0f64, 0.0, 0.5 * BaseFloat::pi()));
///     let v  = Vec3::new(1.0, 0.0, 0.0);
///
///     let tv = na::inv_rotate(&t, &v);
///
///     assert!(na::approx_eq(&tv, &Vec3::new(0.0, -1.0, 0.0)))
/// }
/// ```
#[inline(always)]
pub fn inv_rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.inv_rotate(v)
}

/*
 * RotationWithTranslation<LV, AV>
 */

/// Rotates a copy of `m` by `amount` using `center` as the pivot point.
#[inline(always)]
pub fn append_rotation_wrt_point<LV: Neg<LV>,
                                 AV,
                                 M: RotationWithTranslation<LV, AV>>(
                                 m:      &M,
                                 amount: &AV,
                                 center: &LV) -> M {
    RotationWithTranslation::append_rotation_wrt_point_cpy(m, amount, center)
}

/// Rotates a copy of `m` by `amount` using `m.translation()` as the pivot point.
#[inline(always)]
pub fn append_rotation_wrt_center<LV: Neg<LV>,
                                  AV,
                                  M: RotationWithTranslation<LV, AV>>(
                                  m:      &M,
                                  amount: &AV) -> M {
    RotationWithTranslation::append_rotation_wrt_center_cpy(m, amount)
}

/*
 * RotationMatrix<LV, AV, R>
 */

/// Builds a rotation matrix from `r`.
#[inline(always)]
pub fn to_rot_mat<N, LV, AV, M: Mat<N, LV, LV> + Rotation<AV>, R: RotationMatrix<N, LV, AV, M>>(r: &R) -> M {
    r.to_rot_mat()
}

/*
 * AbsoluteRotate<V>
 */

/// Applies a rotation using the absolute values of its components.
#[inline(always)]
pub fn absolute_rotate<V, M: AbsoluteRotate<V>>(m: &M, v: &V) -> V {
    m.absolute_rotate(v)
}

/*
 * Transformation<T>
 */

/// Gets the transformation applicable by `m`.
#[inline(always)]
pub fn transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.transformation()
}

/// Gets the inverse transformation applicable by `m`.
#[inline(always)]
pub fn inv_transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.inv_transformation()
}

/// Gets a transformed copy of `m`.
#[inline(always)]
pub fn append_transformation<T, M: Transformation<T>>(m: &M, t: &T) -> M {
    Transformation::append_transformation_cpy(m, t)
}

/*
 * Transform<V>
 */

/// Applies a transformation to a vector.
#[inline(always)]
pub fn transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    m.transform(v)
}

/// Applies an inverse transformation to a vector.
#[inline(always)]
pub fn inv_transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    m.inv_transform(v)
}

/*
 * Dot<N>
 */

/// Computes the dot product of two vectors.
#[inline(always)]
pub fn dot<V: Dot<N>, N>(a: &V, b: &V) -> N {
    Dot::dot(a, b)
}

/*
 * Norm<N>
 */

/// Computes the L2 norm of a vector.
#[inline(always)]
pub fn norm<V: Norm<N>, N: BaseFloat>(v: &V) -> N {
    Norm::norm(v)
}

/// Computes the squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<V: Norm<N>, N: BaseFloat>(v: &V) -> N {
    Norm::sqnorm(v)
}

/// Gets the normalized version of a vector.
#[inline(always)]
pub fn normalize<V: Norm<N>, N: BaseFloat>(v: &V) -> V {
    Norm::normalize_cpy(v)
}

/*
 * Det<N>
 */
/// Computes the determinant of a square matrix.
#[inline(always)]
pub fn det<M: Det<N>, N>(m: &M) -> N {
    Det::det(m)
}

/*
 * Cross<V>
 */

/// Computes the cross product of two vectors.
#[inline(always)]
pub fn cross<LV: Cross<AV>, AV>(a: &LV, b: &LV) -> AV {
    Cross::cross(a, b)
}

/*
 * CrossMatrix<M>
 */

/// Given a vector, computes the matrix which, when multiplied by another vector, computes a cross
/// product.
#[inline(always)]
pub fn cross_matrix<V: CrossMatrix<M>, M>(v: &V) -> M {
    CrossMatrix::cross_matrix(v)
}

/*
 * ToHomogeneous<U>
 */

/// Converts a matrix or vector to homogeneous coordinates.
#[inline(always)]
pub fn to_homogeneous<M: ToHomogeneous<Res>, Res>(m: &M) -> Res {
    ToHomogeneous::to_homogeneous(m)
}

/*
 * FromHomogeneous<U>
 */

/// Converts a matrix or vector from homogeneous coordinates.
///
/// w-normalization is appied.
#[inline(always)]
pub fn from_homogeneous<M, Res: FromHomogeneous<M>>(m: &M) -> Res {
    FromHomogeneous::from(m)
}

/*
 * UniformSphereSample
 */

/// Samples the unit sphere living on the dimension as the samples types.
///
/// The number of sampling point is implementation-specific. It is always uniform.
#[inline(always)]
pub fn sample_sphere<V: UniformSphereSample>(f: |V| -> ()) {
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
#[inline(always)]
pub fn approx_eq<T: ApproxEq<N>, N>(a: &T, b: &T) -> bool {
    ApproxEq::approx_eq(a, b)
}

/// Tests approximate equality using a custom epsilon.
#[inline(always)]
pub fn approx_eq_eps<T: ApproxEq<N>, N>(a: &T, b: &T, eps: &N) -> bool {
    ApproxEq::approx_eq_eps(a, b, eps)
}


/*
 * Absolute<A>
 */

/// Computes a component-wise absolute value.
#[inline(always)]
pub fn abs<M: Absolute<Res>, Res>(m: &M) -> Res {
    Absolute::abs(m)
}

/*
 * Inv
 */

/// Gets an inverted copy of a matrix.
#[inline(always)]
pub fn inv<M: Inv>(m: &M) -> Option<M> {
    Inv::inv_cpy(m)
}

/*
 * Transpose
 */

/// Gets a transposed copy of a matrix.
#[inline(always)]
pub fn transpose<M: Transpose>(m: &M) -> M {
    Transpose::transpose_cpy(m)
}

/*
 * Outer<M>
 */

/// Computes the outer product of two vectors.
#[inline(always)]
pub fn outer<V: Outer<M>, M>(a: &V, b: &V) -> M {
    Outer::outer(a, b)
}

/*
 * Cov<M>
 */

/// Computes the covariance of a set of observations.
#[inline(always)]
pub fn cov<M: Cov<Res>, Res>(observations: &M) -> Res {
    Cov::cov(observations)
}

/*
 * Mean<N>
 */

/// Computes the mean of a set of observations.
#[inline(always)]
pub fn mean<N, M: Mean<N>>(observations: &M) -> N {
    Mean::mean(observations)
}

/*
 * EigenQR<N, V>
 */
/// Computes the eigenvalues and eigenvectors of a square matrix usin the QR algorithm.
#[inline(always)]
pub fn eigen_qr<N, V, M: EigenQR<N, V>>(m: &M, eps: &N, niter: uint) -> (M, V) {
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
#[inline(always)]
pub fn new_identity<M: Eye>(dim: uint) -> M {
    Eye::new_identity(dim)
}

/*
 * Basis
 */

/// Computes the canonical basis for a given dimension.
#[inline(always)]
pub fn canonical_basis<V: Basis>(f: |V| -> bool) {
    Basis::canonical_basis(f)
}

/// Computes the basis of the orthonormal subspace of a given vector.
#[inline(always)]
pub fn orthonormal_subspace_basis<V: Basis>(v: &V, f: |V| -> bool) {
    Basis::orthonormal_subspace_basis(v, f)
}

/// Gets the (0-based) i-th element of the canonical basis of V.
#[inline]
pub fn canonical_basis_element<V: Basis>(i: uint) -> Option<V> {
    Basis::canonical_basis_element(i)
}

/*
 * Row<R>
 */

/*
 * Col<C>
 */

/*
 * Diag<V>
 */
/// Gets the diagonal of a square matrix.
#[inline(always)]
pub fn diag<M: Diag<V>, V>(m: &M) -> V {
    m.diag()
}

/*
 * Dim
 */
/// Gets the dimension an object lives in.
///
/// Same as `Dim::dim::(None::<V>)`.
#[inline(always)]
pub fn dim<V: Dim>() -> uint {
    Dim::dim(None::<V>)
}

/// Gets the indexable range of an object.
#[inline(always)]
pub fn shape<V: Shape<I, N>, I, N>(v: &V) -> I {
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
/// * Type-level geometric invariants cannot be broken (eg. a cast from Rot3<f64> to Rot3<i64> is
/// not possible)
/// * A cast to a type with more type-level invariants cannot be done (eg. a cast from Mat<f64> to
/// Rot3<f64> is not possible)
/// * For primitive types an unbounded cast is done using the `as` keyword (this is different from
/// the standard library which makes bound-checking to ensure eg. that a i64 is not out of the
/// range of an i32 when a cast from i64 to i32 is done).
/// * A cast does not affect the dimension of an algebraic object. Note that this prevents an
/// isometric transform to be cast to a raw matrix. Use `to_homogeneous` for that special purpose.
#[inline(always)]
pub fn cast<T, U: Cast<T>>(t: T) -> U {
    Cast::from(t)
}

/*
 * Indexable
 */
