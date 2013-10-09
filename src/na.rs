//! **nalgebra** prelude.

use std::num::{Zero, One};
pub use traits::{
    Absolute,
    AbsoluteRotate,
    AlgebraicVec,
    AlgebraicVecExt,
    Basis,
    Cast,
    Col,
    Cov,
    Cross,
    CrossMatrix,
    Dim,
    Dot,
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
    RMul,
    Rotate, Rotation, RotationMatrix, RotationWithTranslation,
    Row,
    ScalarAdd, ScalarSub,
    ToHomogeneous,
    Transform, Transformation,
    Translate, Translation,
    Transpose,
    UniformSphereSample,
    Vec,
    VecExt
};

pub use structs::{
    Identity,
    DMat, DVec,
    Iso2, Iso3, Iso4,
    Mat1, Mat2, Mat3, Mat4,
    Mat5, Mat6,
    Rot2, Rot3, Rot4,
    Vec0, Vec1, Vec2, Vec3, Vec4, Vec5, Vec6
};

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
/// This is the same as `std::num::Zero::zero()`.
#[inline(always)]
pub fn zero<T: Zero>() -> T {
    Zero::zero()
}

/// Create a one-valued value.
///
/// This is the same as `std::num::One::one()`.
#[inline(always)]
pub fn one<T: One>() -> T {
    One::one()
}

/// Creates a new 1d vector.
///
/// This is the same as `Vec1::new(x)`.
#[inline(always)]
pub fn vec1<N>(x: N) -> Vec1<N> {
    Vec1::new(x)
}

/// Creates a new 2d vector.
///
/// This is the same as `Vec2::new(x, y)`.
#[inline(always)]
pub fn vec2<N>(x: N, y: N) -> Vec2<N> {
    Vec2::new(x, y)
}

/// Creates a new 3d vector.
///
/// This is the same as `Vec3::new(x, y, z)`.
#[inline(always)]
pub fn vec3<N>(x: N, y: N, z: N) -> Vec3<N> {
    Vec3::new(x, y, z)
}

/// Creates a new 4d vector.
///
/// This is the same as `Vec4::new(x, y, z, w)`.
#[inline(always)]
pub fn vec4<N>(x: N, y: N, z: N, w: N) -> Vec4<N> {
    Vec4::new(x, y, z, w)
}

/// Creates a new 1d matrix.
///
/// This is the same as `Mat1::new(...)`.
#[inline(always)]
pub fn mat1<N>(m11: N) -> Mat1<N> {
    Mat1::new(m11)
}

/// Creates a new 2d matrix.
///
/// This is the same as `Mat2::new(...)`.
#[inline(always)]
pub fn mat2<N>(m11: N, m12: N,
               m21: N, m22: N) -> Mat2<N> {
    Mat2::new(
        m11, m12,
        m21, m22)
}

/// Creates a new 3d matrix.
///
/// This is the same as `Mat3::new(...)`.
#[inline(always)]
pub fn mat3<N>(m11: N, m12: N, m13: N,
               m21: N, m22: N, m23: N,
               m31: N, m32: N, m33: N) -> Mat3<N> {
    Mat3::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33)
}

/// Creates a new 4d matrix.
///
/// This is the same as `Mat4::new(...)`.
#[inline(always)]
pub fn mat4<N>(m11: N, m12: N, m13: N, m14: N,
               m21: N, m22: N, m23: N, m24: N,
               m31: N, m32: N, m33: N, m34: N,
               m41: N, m42: N, m43: N, m44: N) -> Mat4<N> {
    Mat4::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44)
}

//
//
// Geometry
//
//

/*
 * Translation<V>
 */

/// Gets the translation applicable by the given object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::types::{Vec3, Affmat};
/// use nalgebra::na;
///
/// pub main() {
///     let t     = Affmat::new_translation3d(1.0, 1.0, 1.0);
///     let trans = na::translation(t);
///
///     assert!(trans == Vec3::new(1.0, 1.0, 1.0));
/// }
/// ```
#[inline(always)]
pub fn translation<V, M: Translation<V>>(m: &M) -> V {
    m.translation()
}

/// Gets the inverse translation applicable by the given object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::types::{Vec3, Affmat};
/// use nalgebra::na;
///
/// pub main() {
///     let t      = Affmat::new_translation3d(1.0, 1.0, 1.0);
///     let itrans = na::inv_translation(t);
///
///     assert!(itrans == Vec3::new(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline(always)]
pub fn inv_translation<V, M: Translation<V>>(m: &M) -> V {
    m.inv_translation()
}

/// In-place version of `translated`.
#[inline(always)]
pub fn translate_by<V, M: Translation<V>>(m: &mut M, v: &V) {
    m.translate_by(v)
}

/// Gets a translated copy of the given object.
#[inline(always)]
pub fn translated<V, M: Translation<V>>(m: &M, v: &V) -> M {
    m.translated(v)
}

/// Sets the translation of the given object.
#[inline(always)]
pub fn set_translation<V, M: Translation<V>>(m: &mut M, v: V) {
    m.set_translation(v)
}

/*
 * Translate<V>
 */

/// Applies a translation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t  = na::translation3d(1.0, 1.0, 1.0);
///     let v  = na::vec3(2.0, 2.0, 2.0);
///
///     let tv = na::translate(&t, &v);
///
///     assert!(tv == na::vec3(3.0, 3.0, 3.0))
/// }
/// ```
#[inline(always)]
pub fn translate<V, M: Translate<V>>(m: &M, v: &V) -> V {
    m.translate(v)
}

/// Applies an inverse translation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t  = na::translation3d(1.0, 1.0, 1.0);
///     let v  = na::vec3(2.0, 2.0, 2.0);
///
///     let tv = na::translate(&t, &v);
///
///     assert!(tv == na::vec3(1.0, 1.0, 1.0))
/// }
#[inline(always)]
pub fn inv_translate<V, M: Translate<V>>(m: &M, v: &V) -> V {
    m.inv_translate(v)
}

/*
 * Rotation<V>
 */

/// Gets the rotation applicable by the given object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t = na::rot3(1.0, 1.0, 1.0);
///
///     assert!(na::rotation(t) == na::vec3(1.0, 1.0, 1.0));
/// }
/// ```
#[inline(always)]
pub fn rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.rotation()
}


/// Gets the rotation applicable by the given object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t = na::rot3(1.0, 1.0, 1.0);
///
///     assert!(na::inv_rotation(t) == na::vec3(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline(always)]
pub fn inv_rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.inv_rotation()
}

/// Rotates an object in-place.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let mut t = na::rot3(0.0, 0.0, 0.0);
///     let v     = na::vec3(1.0, 1.0, 1.0);
///
///     na::rotate_by(&mut t, &v);
///
///     assert!(na::rotation(&t) == na::vec3(1.0, 1.0, 1.0))
/// }
/// ```
#[inline(always)]
pub fn rotate_by<V, M: Rotation<V>>(m: &mut M, v: &V) {
    m.rotate_by(v)
}

/// Creates a rotated copy of an object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t  = na::rot3(0.0, 0.0, 0.0);
///     let v  = na::vec3(1.0, 1.0, 1.0);
///     let rt = na::rotated(&mut t, &v);
///
///     assert!(na::rotation(&rt) == na::vec3(1.0, 1.0, 1.0))
/// }
/// ```
#[inline(always)]
pub fn rotated<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    m.rotated(v)
}

/// Sets the rotation of an object.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let mut t = na::rot3(1.0, 0.5, 0.2);
///     let v     = na::vec3(1.0, 1.0, 1.0);
///
///     na::set_rotation(&mut t, &v);
///
///     assert!(na::rotation(&t) == na::vec3(1.0, 1.0, 1.0))
/// }
/// ```
#[inline(always)]
pub fn set_rotation<V, M: Rotation<V>>(m: &mut M, v: V) {
    m.set_rotation(v)
}

/*
 * Rotate<V>
 */

/// Applies a rotation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t  = na::rot3(1.0, 0.0, 0.0);
///     let v  = na::vec3(0.0, 0.0, na::pi() / 2.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(tv == na::vec3(0.0, 1.0, 0.0))
/// }
/// ```
#[inline(always)]
pub fn rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.rotate(v)
}


/// Applies an inverse rotation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na;
///
/// pub main() {
///     let t  = na::rot3(1.0, 0.0, 0.0);
///     let v  = na::vec3(0.0, 0.0, na::pi() / 2.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(tv == na::vec3(0.0, -1.0, 0.0))
/// }
/// ```
#[inline(always)]
pub fn inv_rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.inv_rotate(v)
}

/*
 * RotationWithTranslation<LV, AV>
 */

/// Creates a rotated copy of an object using a specific center of rotation.
#[inline(always)]
pub fn rotated_wrt_point<LV: Neg<LV>,
                         AV,
                         M: RotationWithTranslation<LV, AV>>(
                         m:      &M,
                         amount: &AV,
                         center: &LV) -> M {
    m.rotated_wrt_point(amount, center)
}

/// In-place version of `rotated_wrt_point`.
#[inline(always)]
pub fn rotate_wrt_point<LV: Neg<LV>,
                        AV,
                        M: RotationWithTranslation<LV, AV>>(
                        m:      &mut M,
                        amount: &AV,
                        center: &LV) {
    m.rotate_wrt_point(amount, center)
}

/// Creates a rotated copy of an object using its own translation as the center of rotation.
#[inline(always)]
pub fn rotated_wrt_center<LV: Neg<LV>,
                          AV,
                          M: RotationWithTranslation<LV, AV>>(
                          m:      &M,
                          amount: &AV) -> M {
    m.rotated_wrt_center(amount)
}

/// In-place version of `rotate_wrt_center`.
#[inline(always)]
pub fn rotate_wrt_center<LV: Neg<LV>,
                         AV,
                         M: RotationWithTranslation<LV, AV>>(
                         m:      &mut M,
                         amount: &AV) {
    m.rotate_wrt_center(amount)
}

/*
 * RotationMatrix<LV, AV, R>
 */

/// Builds a rotation matrix from a rotation-capable object.
#[inline(always)]
pub fn to_rot_mat<LV, AV, M: Mat<LV, LV> + Rotation<AV>, R: RotationMatrix<LV, AV, M>>(r: &R) -> M {
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

/// Gets the transformation applicable by the given object.
#[inline(always)]
pub fn transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.transformation()
}

/// Gets the inverse transformation applicable by the given object.
#[inline(always)]
pub fn inv_transformation<T, M: Transformation<T>>(m: &M) -> T {
    m.inv_transformation()
}

/// In-place version of `transformed`.
#[inline(always)]
pub fn transform_by<T, M: Transformation<T>>(m: &mut M, t: &T) {
    m.transform_by(t)
}

/// Gets a transformed copy of an object.
#[inline(always)]
pub fn transformed<T, M: Transformation<T>>(m: &M, t: &T) -> M {
    m.transformed(t)
}

/// Sets the transformation of an object.
#[inline(always)]
pub fn set_transformation<T, M: Transformation<T>>(m: &mut M, t: T) {
    m.set_transformation(t)
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
    a.dot(b)
}

/// Computes a subtraction followed by a dot product.
#[inline(always)]
pub fn sub_dot<V: Dot<N>, N>(a: &V, b: &V, c: &V) -> N {
    a.sub_dot(b, c)
}

/*
 * Norm<N>
 */

/// Computes the L2 norm of a vector.
#[inline(always)]
pub fn norm<V: Norm<N>, N: Algebraic>(v: &V) -> N {
    v.norm()
}

/// Computes the squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<V: Norm<N>, N: Algebraic>(v: &V) -> N {
    v.sqnorm()
}

/// Gets the normalized version of a vector.
#[inline(always)]
pub fn normalized<V: Norm<N>, N: Algebraic>(v: &V) -> V {
    v.normalized()
}

/// In-place version of `normalized`.
#[inline(always)]
pub fn normalize<V: Norm<N>, N: Algebraic>(v: &mut V) -> N {
    v.normalize()
}

/*
 * Cross<V>
 */

/// Computes the cross product of two vectors.
#[inline(always)]
pub fn cross<LV: Cross<AV>, AV>(a: &LV, b: &LV) -> AV {
    a.cross(b)
}

/*
 * CrossMatrix<M>
 */

/// Given a vector, computes the matrix which, when multiplied by another vector, computes a cross
/// product.
#[inline(always)]
pub fn cross_matrix<V: CrossMatrix<M>, M>(v: &V) -> M {
    v.cross_matrix()
}

/*
 * ToHomogeneous<U>
 */

/// Converts a matrix or vector to homogoneous coordinates.
#[inline(always)]
pub fn to_homogeneous<M: ToHomogeneous<Res>, Res>(m: &M) -> Res {
    m.to_homogeneous()
}

/*
 * FromHomogeneous<U>
 */

/// Converts a matrix or vector from homogoneous coordinates.
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
pub fn sample_sphere<V: UniformSphereSample>(f: &fn(V)) {
    UniformSphereSample::sample(f)
}

//
//
// Operations
//
//


/*
 * Absolute<A>
 */

/// Computes a component-wise absolute value.
#[inline(always)]
pub fn absolute<M: Absolute<Res>, Res>(m: &M) -> Res {
    m.absolute()
}

/*
 * Inv
 */

/// Gets an inverted copy of a matrix.
#[inline(always)]
pub fn inverted<M: Inv>(m: &M) -> Option<M> {
    m.inverted()
}

/// In-place version of `inverted`.
#[inline(always)]
pub fn invert<M: Inv>(m: &mut M) -> bool {
    m.invert()
}

/*
 * Transpose
 */

/// Gets a transposed copy of a matrix.
#[inline(always)]
pub fn transposed<M: Transpose>(m: &M) -> M {
    m.transposed()
}

/// In-place version of `transposed`.
#[inline(always)]
pub fn transpose<M: Transpose>(m: &mut M) {
    m.transpose()
}

/*
 * Outer<M>
 */

/// Computes the outer product of two vectors.
#[inline(always)]
pub fn outer<V: Outer<M>, M>(a: &V, b: &V) -> M {
    a.outer(b)
}

/*
 * Cov<M>
 */

/// Computes the covariance of a set of observations.
#[inline(always)]
pub fn cov<M: Cov<Res>, Res>(observations: &M) -> Res {
    observations.cov()
}

/*
 * Mean<N>
 */

/// Computes the mean of a set of observations.
#[inline(always)]
pub fn mean<N, M: Mean<N>>(observations: &M) -> N {
    observations.mean()
}

//
//
// Structure
//
//

/*
 * Basis
 */

/// Computes the canonical basis for a given dimension.
#[inline(always)]
pub fn canonical_basis<V: Basis>(f: &fn(V) -> bool) {
    Basis::canonical_basis(f)
}

/// Computes the basis of the orthonormal subspace of a given vector.
#[inline(always)]
pub fn orthonormal_subspace_basis<V: Basis>(v: &V, f: &fn(V) -> bool) {
    v.orthonormal_subspace_basis(f)
}

/*
 * Row<R>
 */

/*
 * Col<C>
 */

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

/*
 * Cast<T>
 */
/// Converts an object from one type to another.
///
/// For primitive types, this is the same as the `as` keywords.
/// Those properties are preserved by a cast:
///     * Type-level geometric invariants cannot be broken (eg. a cast from Rot3<f64> to Rot3<i64>
///     is not possible)
///     * A cast to a type with more type-level invariants cannot be done (eg. a cast from
///     Mat<f64> to Rot3<f64> is not possible)
///     * For primitive types an unbounded cast is done using the `as` keyword (this is different
///     from the standard library which makes bound-checking to ensure eg. that a i64 is not out of
///     the range of an i32 when a cast from i64 to i32 is done).
///     * A cast does not affect the dimension of an algebraic object. Note that this prevents an
///     isometric transform to be cast to a raw matrix. Use `to_homogeneous` for that special
///     purpose.
#[inline(always)]
pub fn cast<T, U: Cast<T>>(t: T) -> U {
    Cast::from(t)
}

/*
 * Indexable
 */
