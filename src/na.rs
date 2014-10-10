//! [DEPRECATED] **nalgebra** prelude.

use std::num::{Zero, One, FloatMath};
use std::cmp;
pub use traits::{PartialLess, PartialEqual, PartialGreater, NotComparable};
pub use traits::{
    Absolute,
    AbsoluteRotate,
    AnyVec,
    ApproxEq,
    Basis,
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
    Eye,
    FloatVec,
    FloatVecExt,
    FromHomogeneous,
    Indexable,
    Inv,
    Iterable,
    IterableMut,
    LMul,
    Mat,
    Mean,
    Norm,
    Outer,
    PartialOrd,
    PartialOrdering,
    RMul,
    Rotate, Rotation, RotationMatrix, RotationWithTranslation,
    Row,
    ScalarAdd, ScalarSub,
    ScalarMul, ScalarDiv,
    ToHomogeneous,
    Transform, Transformation,
    Translate, Translation,
    Transpose,
    UniformSphereSample,
    VecExt
};

pub use structs::{
    Identity,
    DMat,
    DVec, DVec1, DVec2,  DVec3,  DVec4,  DVec5,  DVec6,
    Iso2, Iso3, Iso4,
    Mat1, Mat2, Mat3, Mat4,
    Mat5, Mat6,
    Rot2, Rot3, Rot4,
    Vec0, Vec1, Vec2, Vec3, Vec4, Vec5, Vec6
};

pub use linalg::{
    qr,
    eigen_qr,
    householder_matrix
};

/// Traits to work around the language limitations related to operator overloading.
///
/// The trait names are formed by:
///
/// * a type name (eg. Vec1, Vec2, Mat3, Mat4, etc.).
/// * the name of a binary operation (eg. Mul, Div, Add, Sub, etc.). 
/// * the word `Rhs`.
///
/// When implemented by the type `T`, the trait makes it possible to overload the binary operator
/// between `T` and the type name given by the trait.
///
/// # Examples:
///
/// * `Vec3MulRhs` will allow the overload of the `*` operator between the implementor type and
/// `Vec3`. The `Vec3` being the first argument of the multiplication.
/// * `Mat4DivRhs` will allow the overload of the `/` operator between the implementor type and
/// `Mat4`. The `Mat4` being the first argument of the division.
pub mod overload {
    pub use structs::{Vec1MulRhs, Vec2MulRhs, Vec3MulRhs, Vec4MulRhs, Vec5MulRhs, Vec6MulRhs,
                      Vec1DivRhs, Vec2DivRhs, Vec3DivRhs, Vec4DivRhs, Vec5DivRhs, Vec6DivRhs,
                      Vec1AddRhs, Vec2AddRhs, Vec3AddRhs, Vec4AddRhs, Vec5AddRhs, Vec6AddRhs,
                      Vec1SubRhs, Vec2SubRhs, Vec3SubRhs, Vec4SubRhs, Vec5SubRhs, Vec6SubRhs,
                      Mat1MulRhs, Mat2MulRhs, Mat3MulRhs, Mat4MulRhs, Mat5MulRhs, Mat6MulRhs,
                      Mat1DivRhs, Mat2DivRhs, Mat3DivRhs, Mat4DivRhs, Mat5DivRhs, Mat6DivRhs,
                      Mat1AddRhs, Mat2AddRhs, Mat3AddRhs, Mat4AddRhs, Mat5AddRhs, Mat6AddRhs,
                      Mat1SubRhs, Mat2SubRhs, Mat3SubRhs, Mat4SubRhs, Mat5SubRhs, Mat6SubRhs};
}

/// Change the input value to ensure it is on the range `[min, max]`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn clamp<T: cmp::PartialOrd>(val: T, min: T, max: T) -> T {
    super::clamp(val, min, max)
}

/// Same as `cmp::max`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn max<T: Ord>(a: T, b: T) -> T {
    super::max(a, b)
}

/// Same as `cmp::min`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn min<T: Ord>(a: T, b: T) -> T {
    super::min(a, b)
}

/// Returns the infimum of `a` and `b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inf<T: PartialOrd>(a: &T, b: &T) -> T {
    super::inf(a, b)
}

/// Returns the supremum of `a` and `b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn sup<T: PartialOrd>(a: &T, b: &T) -> T {
    super::sup(a, b)
}

/// Compare `a` and `b` using a partial ordering relation.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_cmp<T: PartialOrd>(a: &T, b: &T) -> PartialOrdering {
    super::partial_cmp(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a < b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_lt<T: PartialOrd>(a: &T, b: &T) -> bool {
    super::partial_lt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a <= b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_le<T: PartialOrd>(a: &T, b: &T) -> bool {
    super::partial_le(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a > b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_gt<T: PartialOrd>(a: &T, b: &T) -> bool {
    super::partial_gt(a, b)
}

/// Returns `true` iff `a` and `b` are comparable and `a >= b`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_ge<T: PartialOrd>(a: &T, b: &T) -> bool {
    super::partial_ge(a, b)
}

/// Return the minimum of `a` and `b` if they are comparable.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_min<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    super::partial_min(a, b)
}

/// Return the maximum of `a` and `b` if they are comparable.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_max<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    super::partial_max(a, b)
}

/// Clamp `value` between `min` and `max`. Returns `None` if `value` is not comparable to
/// `min` or `max`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn partial_clamp<'a, T: PartialOrd>(value: &'a T, min: &'a T, max: &'a T) -> Option<&'a T> {
    super::partial_clamp(value, min, max)
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
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn identity() -> Identity {
    super::identity()
}

/// Create a zero-valued value.
///
/// This is the same as `std::num::zero()`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn zero<T: Zero>() -> T {
    super::zero()
}

/// Create a one-valued value.
///
/// This is the same as `std::num::one()`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn one<T: One>() -> T {
    super::one()
}

//
//
// Geometry
//
//

/*
 * Perspective
 */
/// Computes a projection matrix given the frustrum near plane width, height, the field of
/// view, and the distance to the clipping planes (`znear` and `zfar`).
pub fn perspective3d<N: FloatMath + Cast<f32> + Zero + One>(width: N, height: N, fov: N, znear: N, zfar: N) -> Mat4<N> {
    super::perspective3d(width, height, fov, znear, zfar)
}

/*
 * Translation<V>
 */

/// Gets the translation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn translation<V, M: Translation<V>>(m: &M) -> V {
    super::translation(m)
}

/// Gets the inverse translation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_translation<V, M: Translation<V>>(m: &M) -> V {
    super::inv_translation(m)
}

/// Applies the translation `v` to a copy of `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn append_translation<V, M: Translation<V>>(m: &M, v: &V) -> M {
    super::append_translation(m, v)
}

/*
 * Translate<V>
 */

/// Applies a translation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn translate<V, M: Translate<V>>(m: &M, v: &V) -> V {
    super::translate(m, v)
}

/// Applies an inverse translation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_translate<V, M: Translate<V>>(m: &M, v: &V) -> V {
    super::inv_translate(m, v)
}

/*
 * Rotation<V>
 */

/// Gets the rotation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn rotation<V, M: Rotation<V>>(m: &M) -> V {
    super::rotation(m)
}


/// Gets the inverse rotation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_rotation<V, M: Rotation<V>>(m: &M) -> V {
    super::inv_rotation(m)
}

// FIXME: this example is a bit shity
/// Applies the rotation `v` to a copy of `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn append_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    super::append_rotation(m, v)
}

// FIXME: this example is a bit shity
/// Pre-applies the rotation `v` to a copy of `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn prepend_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    super::prepend_rotation(m, v)
}

/*
 * Rotate<V>
 */

/// Applies a rotation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    super::rotate(m, v)
}


/// Applies an inverse rotation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    super::inv_rotate(m, v)
}

/*
 * RotationWithTranslation<LV, AV>
 */

/// Rotates a copy of `m` by `amount` using `center` as the pivot point.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn append_rotation_wrt_point<LV: Neg<LV>,
                                 AV,
                                 M: RotationWithTranslation<LV, AV>>(
                                 m:      &M,
                                 amount: &AV,
                                 center: &LV) -> M {
    super::append_rotation_wrt_point(m, amount, center)
}

/// Rotates a copy of `m` by `amount` using `m.translation()` as the pivot point.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn append_rotation_wrt_center<LV: Neg<LV>,
                                  AV,
                                  M: RotationWithTranslation<LV, AV>>(
                                  m:      &M,
                                  amount: &AV) -> M {
    super::append_rotation_wrt_center(m, amount)
}

/*
 * RotationMatrix<LV, AV, R>
 */

/// Builds a rotation matrix from `r`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn to_rot_mat<LV, AV, M: Mat<LV, LV> + Rotation<AV>, R: RotationMatrix<LV, AV, M>>(r: &R) -> M {
    super::to_rot_mat(r)
}

/*
 * AbsoluteRotate<V>
 */

/// Applies a rotation using the absolute values of its components.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn absolute_rotate<V, M: AbsoluteRotate<V>>(m: &M, v: &V) -> V {
    super::absolute_rotate(m, v)
}

/*
 * Transformation<T>
 */

/// Gets the transformation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn transformation<T, M: Transformation<T>>(m: &M) -> T {
    super::transformation(m)
}

/// Gets the inverse transformation applicable by `m`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_transformation<T, M: Transformation<T>>(m: &M) -> T {
    super::inv_transformation(m)
}

/// Gets a transformed copy of `m`.
#[inline(always)]
pub fn append_transformation<T, M: Transformation<T>>(m: &M, t: &T) -> M {
    super::append_transformation(m, t)
}

/*
 * Transform<V>
 */

/// Applies a transformation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    super::transform(m, v)
}

/// Applies an inverse transformation to a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv_transform<V, M: Transform<V>>(m: &M, v: &V) -> V {
    super::inv_transform(m, v)
}

/*
 * Dot<N>
 */

/// Computes the dot product of two vectors.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn dot<V: Dot<N>, N>(a: &V, b: &V) -> N {
    super::dot(a, b)
}

/*
 * Norm<N>
 */

/// Computes the L2 norm of a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn norm<V: Norm<N>, N: Float>(v: &V) -> N {
    super::norm(v)
}

/// Computes the squared L2 norm of a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn sqnorm<V: Norm<N>, N: Float>(v: &V) -> N {
    super::sqnorm(v)
}

/// Gets the normalized version of a vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn normalize<V: Norm<N>, N: Float>(v: &V) -> V {
    super::normalize(v)
}

/*
 * Det<N>
 */
/// Computes the determinant of a square matrix.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn det<M: Det<N>, N>(m: &M) -> N {
    super::det(m)
}

/*
 * Cross<V>
 */

/// Computes the cross product of two vectors.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn cross<LV: Cross<AV>, AV>(a: &LV, b: &LV) -> AV {
    super::cross(a, b)
}

/*
 * CrossMatrix<M>
 */

/// Given a vector, computes the matrix which, when multiplied by another vector, computes a cross
/// product.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn cross_matrix<V: CrossMatrix<M>, M>(v: &V) -> M {
    super::cross_matrix(v)
}

/*
 * ToHomogeneous<U>
 */

/// Converts a matrix or vector to homogeneous coordinates.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn to_homogeneous<M: ToHomogeneous<Res>, Res>(m: &M) -> Res {
    super::to_homogeneous(m)
}

/*
 * FromHomogeneous<U>
 */

/// Converts a matrix or vector from homogeneous coordinates.
///
/// w-normalization is appied.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn from_homogeneous<M, Res: FromHomogeneous<M>>(m: &M) -> Res {
    super::from_homogeneous(m)
}

/*
 * UniformSphereSample
 */

/// Samples the unit sphere living on the dimension as the samples types.
///
/// The number of sampling point is implementation-specific. It is always uniform.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn sample_sphere<V: UniformSphereSample>(f: |V| -> ()) {
    super::sample_sphere(f)
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
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn approx_eq<T: ApproxEq<N>, N>(a: &T, b: &T) -> bool {
    super::approx_eq(a, b)
}

/// Tests approximate equality using a custom epsilon.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn approx_eq_eps<T: ApproxEq<N>, N>(a: &T, b: &T, eps: &N) -> bool {
    super::approx_eq_eps(a, b, eps)
}


/*
 * Absolute<A>
 */

/// Computes a component-wise absolute value.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn abs<M: Absolute<Res>, Res>(m: &M) -> Res {
    super::abs(m)
}

/*
 * Inv
 */

/// Gets an inverted copy of a matrix.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn inv<M: Inv>(m: &M) -> Option<M> {
    super::inv(m)
}

/*
 * Transpose
 */

/// Gets a transposed copy of a matrix.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn transpose<M: Transpose>(m: &M) -> M {
    super::transpose(m)
}

/*
 * Outer<M>
 */

/// Computes the outer product of two vectors.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn outer<V: Outer<M>, M>(a: &V, b: &V) -> M {
    super::outer(a, b)
}

/*
 * Cov<M>
 */

/// Computes the covariance of a set of observations.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn cov<M: Cov<Res>, Res>(observations: &M) -> Res {
    super::cov(observations)
}

/*
 * Mean<N>
 */

/// Computes the mean of a set of observations.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn mean<N, M: Mean<N>>(observations: &M) -> N {
    super::mean(observations)
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
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn new_identity<M: Eye>(dim: uint) -> M {
    super::new_identity(dim)
}

/*
 * Basis
 */

/// Computes the canonical basis for a given dimension.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn canonical_basis<V: Basis>(f: |V| -> bool) {
    super::canonical_basis(f)
}

/// Computes the basis of the orthonormal subspace of a given vector.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn orthonormal_subspace_basis<V: Basis>(v: &V, f: |V| -> bool) {
    super::orthonormal_subspace_basis(v, f)
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
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn diag<M: Diag<V>, V>(m: &M) -> V {
    super::diag(m)
}

/*
 * Dim
 */
/// Gets the dimension an object lives in.
///
/// Same as `Dim::dim::(None::<V>)`.
#[inline(always)]
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn dim<V: Dim>() -> uint {
    super::dim::<V>()
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
#[deprecated = "use the root module `nalgebra::` directly instead of the `nalgebra::na::` module (you may create an alias `extern crate \"nalgebra\" as na;` when importing the crate)"]
pub fn cast<T, U: Cast<T>>(t: T) -> U {
    super::cast::<T, U>(t)
}

/*
 * Indexable
 */
