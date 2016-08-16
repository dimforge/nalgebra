//! Traits of operations having a well-known or explicit geometric meaning.

use std::ops::{Neg, Mul};
use num::Float;
use traits::structure::{BaseFloat, SquareMatrix};

/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation<V> {
    // FIXME: add a "from translation: translantion(V) -> Self ?
    /// Gets the translation associated with this object.
    fn translation(&self) -> V;

    /// Gets the inverse translation associated with this object.
    fn inverse_translation(&self) -> V;

    /// Appends a translation to this object.
    fn append_translation_mut(&mut self, &V);

    /// Appends the translation `amount` to a copy of `t`.
    fn append_translation(&self, amount: &V) -> Self;

    /// Prepends a translation to this object.
    fn prepend_translation_mut(&mut self, &V);

    /// Prepends the translation `amount` to a copy of `t`.
    fn prepend_translation(&self, amount: &V) -> Self;

    /// Sets the translation.
    fn set_translation(&mut self, V);
}

/// Trait of objects able to translate other objects. This is typically
/// implemented by vectors to translate points.
pub trait Translate<V> {
    /// Apply a translation to an object.
    fn translate(&self, &V) -> V;

    /// Apply an inverse translation to an object.
    fn inverse_translate(&self, &V) -> V;
}

/// Trait of object which can represent a rotation, and to which new rotations can be appended. A
/// rotation is assumed to be an isometry without translation and without reflexion.
pub trait Rotation<V> {
    /// Gets the rotation associated with `self`.
    fn rotation(&self) -> V;

    /// Gets the inverse rotation associated with `self`.
    fn inverse_rotation(&self) -> V;

    /// Appends a rotation to this object.
    fn append_rotation_mut(&mut self, &V);

    /// Appends the rotation `amount` to a copy of `t`.
    fn append_rotation(&self, amount: &V) -> Self;

    /// Prepends a rotation to this object.
    fn prepend_rotation_mut(&mut self, &V);

    /// Prepends the rotation `amount` to a copy of `t`.
    fn prepend_rotation(&self, amount: &V) -> Self;

    /// Sets the rotation of `self`.
    fn set_rotation(&mut self, V);
}

/// Trait of object that can be rotated to be superimposed with another one of the same nature.
pub trait RotationTo {
    /// Type of the angle between two elements.
    type AngleType;

    /// Type of the rotation between two elements.
    type DeltaRotationType;

    /// Computes an angle nedded to transform the first element to the second one using a
    /// rotation.
    fn angle_to(&self, other: &Self) -> Self::AngleType;

    /// Computes the smallest rotation needed to transform the first element to the second one.
    fn rotation_to(&self, other: &Self) -> Self::DeltaRotationType;
}

/// Trait of objects able to rotate other objects.
///
/// This is typically implemented by matrices which rotate vectors.
pub trait Rotate<V> {
    /// Applies a rotation to `v`.
    fn rotate(&self, v: &V) -> V;

    /// Applies an inverse rotation to `v`.
    fn inverse_rotate(&self, v: &V) -> V;
}

/// Various composition of rotation and translation.
///
/// Utilities to make rotations with regard to a point different than the origin.  All those
/// operations are the composition of rotations and translations.
///
/// Those operations are automatically implemented in term of the `Rotation` and `Translation`
/// traits.
pub trait RotationWithTranslation<LV: Neg<Output = LV> + Copy, AV>: Rotation<AV> + Translation<LV> + Sized {
    /// Applies a rotation centered on a specific point.
    ///
    /// # Arguments
    ///   * `t` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    ///   * `point` - the center of rotation.
    #[inline]
    fn append_rotation_wrt_point(&self, amount: &AV, center: &LV) -> Self {
        let mut res = Translation::append_translation(self, &-*center);

        res.append_rotation_mut(amount);
        res.append_translation_mut(center);

        res
    }

    /// Rotates `self` using a specific center of rotation.
    ///
    /// The rotation is applied in-place.
    ///
    /// # Arguments
    ///   * `amount` - the rotation to be applied
    ///   * `center` - the new center of rotation
    #[inline]
    fn append_rotation_wrt_point_mut(&mut self, amount: &AV, center: &LV) {
        self.append_translation_mut(&-*center);
        self.append_rotation_mut(amount);
        self.append_translation_mut(center);
    }

    /// Applies a rotation centered on the translation of `m`.
    /// 
    /// # Arguments
    ///   * `t` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn append_rotation_wrt_center(&self, amount: &AV) -> Self {
        RotationWithTranslation::append_rotation_wrt_point(self, amount, &self.translation())
    }

    /// Applies a rotation centered on the translation of `m`.
    ///
    /// The rotation os applied on-place.
    ///
    /// # Arguments
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn append_rotation_wrt_center_mut(&mut self, amount: &AV) {
        let center = self.translation();
        self.append_rotation_wrt_point_mut(amount, &center)
    }
}

impl<LV: Neg<Output = LV> + Copy, AV, M: Rotation<AV> + Translation<LV>> RotationWithTranslation<LV, AV> for M {
}

/// Trait of transformation having a rotation extractable as a rotation matrix. This can typically
/// be implemented by quaternions to convert them to a rotation matrix.
pub trait RotationMatrix<N, LV: Mul<Self::Output, Output = LV>, AV> : Rotation<AV> {
    /// The output rotation matrix type.
    type Output: SquareMatrix<N, LV> + Rotation<AV>;

    /// Gets the rotation matrix represented by `self`.
    fn to_rotation_matrix(&self) -> Self::Output;
}

/// Composition of a rotation and an absolute value.
///
/// The operation is accessible using the `RotationMatrix`, `Absolute`, and `RMul` traits, but
/// doing so is not easy in generic code as it can be a cause of type over-parametrization.
pub trait AbsoluteRotate<V> {
    /// This is the same as:
    ///
    /// ```.ignore
    ///     self.rotation_matrix().absolute().rmul(v)
    /// ```
    fn absolute_rotate(&self, v: &V) -> V;
}

/// Trait of object which represent a transformation, and to which new transformations can
/// be appended.
///
/// A transformation is assumed to be an isometry without reflexion.
pub trait Transformation<M> {
    /// Gets the transformation of `self`.
    fn transformation(&self) -> M;

    /// Gets the inverse transformation of `self`.
    fn inverse_transformation(&self) -> M;

    /// Appends a transformation to this object.
    fn append_transformation_mut(&mut self, &M);

    /// Appends the transformation `amount` to a copy of `t`.
    fn append_transformation(&self, amount: &M) -> Self;

    /// Prepends a transformation to this object.
    fn prepend_transformation_mut(&mut self, &M);

    /// Prepends the transformation `amount` to a copy of `t`.
    fn prepend_transformation(&self, amount: &M) -> Self;

    /// Sets the transformation of `self`.
    fn set_transformation(&mut self, M);
}

/// Trait of objects able to transform other objects.
///
/// This is typically implemented by matrices which transform vectors.
pub trait Transform<V> {
    /// Applies a transformation to `v`.
    fn transform(&self, &V) -> V;

    /// Applies an inverse transformation to `v`.
    fn inverse_transform(&self, &V) -> V;
}

/// Traits of objects having a dot product.
pub trait Dot<N> {
    /// Computes the dot (inner) product of two vectors.
    #[inline]
    fn dot(&self, other: &Self) -> N;
}

/// Traits of objects having an euclidian norm.
pub trait Norm: Sized {
    /// The scalar type for the norm (i.e. the undelying field).
    type NormType : BaseFloat;

    /// Computes the norm of `self`.
    #[inline]
    fn norm(&self) -> Self::NormType {
        self.norm_squared().sqrt()
    }

    /// Computes the squared norm of `self`.
    ///
    /// This is usually faster than computing the norm itself.
    fn norm_squared(&self) -> Self::NormType;

    /// Gets the normalized version of a copy of `v`.
    ///
    /// Might return an invalid result if the vector is zero or close to zero.
    fn normalize(&self) -> Self;

    /// Normalizes `self`.
    ///
    /// The result might be invalid if the vector is zero or close to zero.
    fn normalize_mut(&mut self) -> Self::NormType;

    /// Gets the normalized version of a copy of `v` or `None` if the vector has a norm smaller
    /// or equal to `min_norm`. In particular, `.try_normalize(0.0)` returns `None` if the norm is
    /// exactly zero.
    fn try_normalize(&self, min_norm: Self::NormType) -> Option<Self>;

    /// Normalized `v` or does nothing if the vector has a norm smaller
    /// or equal to `min_norm`.
    ///
    /// Returns the old norm or `None` if the normalization failed.
    fn try_normalize_mut(&mut self, min_norm: Self::NormType) -> Option<Self::NormType>;
}

/**
 * Trait of elements having a cross product.
 */
pub trait Cross {
    /// The cross product output.
    type CrossProductType;

    /// Computes the cross product between two elements (usually vectors).
    fn cross(&self, other: &Self) -> Self::CrossProductType;
}

/**
 * Trait of elements having a cross product operation which can be expressed as a matrix.
 */
pub trait CrossMatrix<M> {
    /// The matrix associated to any cross product with this vector. I.e. `v.cross(anything)` =
    /// `v.cross_matrix().rmul(anything)`.
    fn cross_matrix(&self) -> M;
}

/// Traits of objects which can be put in homogeneous coordinates form.
pub trait ToHomogeneous<U> {
    /// Gets the homogeneous coordinates form of this object.
    fn to_homogeneous(&self) -> U;
}

/// Traits of objects which can be build from an homogeneous coordinate form.
pub trait FromHomogeneous<U> {
    /// Builds an object from its homogeneous coordinate form.
    ///
    /// Note that this this is not required that `from` is the inverse of `to_homogeneous`.
    /// Typically, `from` will remove some informations unrecoverable by `to_homogeneous`.
    fn from(&U) -> Self;
}

/// Trait of vectors able to sample a unit sphere.
///
/// The number of sample must be sufficient to approximate a sphere using a support mapping
/// function.
pub trait UniformSphereSample : Sized {
    /// Iterate through the samples.
    fn sample<F: FnMut(Self)>(F);
}

/// The zero element of a vector space, seen as an element of its embeding affine space.
// XXX: once associated types are suported, move this to the `AnyPoint` trait.
pub trait Origin {
    /// The trivial origin.
    fn origin() -> Self;
    /// Returns true if this points is exactly the trivial origin.
    fn is_origin(&self) -> bool;
}
