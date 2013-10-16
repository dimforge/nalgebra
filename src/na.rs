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

//
//
// Geometry
//
//

/*
 * Translation<V>
 */

/// Gets the translation applicable by `m`.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::types::{Vec3, Iso3};
/// use nalgebra::na;
///
/// pub main() {
///     let t     = Iso3::new(Vec3::new(1.0, 1.0, 1.0), na::zero());
///     let trans = na::translation(t);
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
/// extern mod nalgebra;
/// use nalgebra::types::{Vec3, Iso3};
/// use nalgebra::na;
///
/// pub main() {
///     let t      = Iso3::new(Vec3::new(1.0, 1.0, 1.0), na::zero());
///     let itrans = na::inv_translation(t);
///
///     assert!(itrans == Vec3::new(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline(always)]
pub fn inv_translation<V, M: Translation<V>>(m: &M) -> V {
    m.inv_translation()
}

/// Appied the translation `v` to a copy of `m`.
#[inline(always)]
pub fn append_translation<V, M: Translation<V>>(m: &M, v: &V) -> M {
    Translation::append_translation_cpy(m, v)
}

/*
 * Translate<V>
 */

/// Applies a translation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na::{Vec3, Iso3};
/// use nalgebra::na;
///
/// pub main() {
///     let t  = Iso3::new(Vec3::new(1.0, 1.0, 1.0), na::zero());
///     let v  = Vec3::new(2.0, 2.0, 2.0);
///
///     let tv = na::translate(&t, &v);
///
///     assert!(tv == Vec3::new(3.0, 3.0, 3.0))
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
/// use nalgebra::na::{Vec3, Iso3};
/// use nalgebra::na;
///
/// pub main() {
///     let t  = Iso3::new(Vec3::new(1.0, 1.0, 1.0), na::zero());
///     let v  = Vec3::new(2.0, 2.0, 2.0);
///
///     let tv = na::translate(&t, &v);
///
///     assert!(tv == Vec3::new(1.0, 1.0, 1.0))
/// }
#[inline(always)]
pub fn inv_translate<V, M: Translate<V>>(m: &M, v: &V) -> V {
    m.inv_translate(v)
}

/*
 * Rotation<V>
 */

/// Gets the rotation applicable by `m`.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na::{Vec3, Rot3};
/// use nalgebra::na;
///
/// pub main() {
///     let t = Rot3::new(Vec3::new(1.0, 1.0, 1.0));
///
///     assert!(na::rotation(t) == Vec3::new(1.0, 1.0, 1.0));
/// }
/// ```
#[inline(always)]
pub fn rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.rotation()
}


/// Gets the inverse rotation applicable by `m`.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na::{Vec3, Rot3};
/// use nalgebra::na;
///
/// pub main() {
///     let t = Rot3::new(Vec3::new(1.0, 1.0, 1.0));
///
///     assert!(na::inv_rotation(t) == Vec3::new(-1.0, -1.0, -1.0));
/// }
/// ```
#[inline(always)]
pub fn inv_rotation<V, M: Rotation<V>>(m: &M) -> V {
    m.inv_rotation()
}

/// Applies the rotation `v` to a copy of `m`.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na::{Vec3, Rot3};
/// use nalgebra::na;
///
/// pub main() {
///     let t  = Rot3::new(Vec3::new(0.0, 0.0, 0.0));
///     let v  = Vec3::new(1.0, 1.0, 1.0);
///     let rt = na::append_rotation(&t, &v);
///
///     assert!(na::rotation(&rt) == Vec3::new(1.0, 1.0, 1.0))
/// }
/// ```
#[inline(always)]
pub fn append_rotation<V, M: Rotation<V>>(m: &M, v: &V) -> M {
    Rotation::append_rotation_cpy(m, v)
}

/*
 * Rotate<V>
 */

/// Applies a rotation to a vector.
///
/// ```rust
/// extern mod nalgebra;
/// use nalgebra::na::{Rot3, Vec3};
/// use nalgebra::na;
///
/// pub main() {
///     let t  = Rot3::new(Vec3::new(1.0, 0.0, 0.0));
///     let v  = Vec3::new(0.0, 0.0, na::pi() / 2.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(tv == Vec3::new(0.0, 1.0, 0.0))
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
///     let t  = Rot3::new(Vec3::new(1.0, 0.0, 0.0));
///     let v  = Vec3::new(0.0, 0.0, na::pi() / 2.0);
///
///     let tv = na::rotate(&t, &v);
///
///     assert!(tv == Vec3::new(0.0, -1.0, 0.0))
/// }
/// ```
#[inline(always)]
pub fn inv_rotate<V, M: Rotate<V>>(m: &M, v: &V) -> V {
    m.inv_rotate(v)
}

/*
 * RotationWithTranslation<LV, AV>
 */

/// Rotates a copy of `m` by `amount` using `center` ase the pivot point.
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

/// Computes a subtraction followed by a dot product.
#[inline(always)]
pub fn sub_dot<V: Dot<N>, N>(a: &V, b: &V, c: &V) -> N {
    Dot::sub_dot(a, b, c)
}

/*
 * Norm<N>
 */

/// Computes the L2 norm of a vector.
#[inline(always)]
pub fn norm<V: Norm<N>, N: Algebraic>(v: &V) -> N {
    Norm::norm(v)
}

/// Computes the squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<V: Norm<N>, N: Algebraic>(v: &V) -> N {
    Norm::sqnorm(v)
}

/// Gets the normalized version of a vector.
#[inline(always)]
pub fn normalize<V: Norm<N>, N: Algebraic>(v: &V) -> V {
    Norm::normalize_cpy(v)
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

/// Converts a matrix or vector to homogoneous coordinates.
#[inline(always)]
pub fn to_homogeneous<M: ToHomogeneous<Res>, Res>(m: &M) -> Res {
    ToHomogeneous::to_homogeneous(m)
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
    Basis::orthonormal_subspace_basis(v, f)
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
