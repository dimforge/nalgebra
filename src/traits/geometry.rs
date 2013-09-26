use traits::structure::Mat;

/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation<V> {
    // FIXME: add a "from translation: translantion(V) -> Self ?
    /// Gets the translation associated with this object.
    fn translation(&self) -> V;

    /// Gets the inverse translation associated with this object.
    fn inv_translation(&self) -> V;

    /// In-place version of `translated`.
    fn translate_by(&mut self, &V);

    /// Appends a translation.
    fn translated(&self, &V) -> Self;

    /// Sets the translation.
    fn set_translation(&mut self, V);
}

/// Trait of objects able to rotate other objects. This is typically implemented by matrices which
/// rotate vectors.
pub trait Translate<V> {
    /// Apply a translation to an object.
    fn translate(&self, &V)     -> V;
    /// Apply an inverse translation to an object.
    fn inv_translate(&self, &V) -> V;
}

/// Trait of object which can represent a rotation, and to which new rotations can be appended. A
/// rotation is assumed to be an isometry without translation and without reflexion.
pub trait Rotation<V> {
    /// Gets the rotation associated with `self`.
    fn rotation(&self) -> V;

    /// Gets the inverse rotation associated with `self`.
    fn inv_rotation(&self) -> V;

    /// In-place version of `rotated`.
    fn rotate_by(&mut self, &V);

    /// Appends a rotation to `self`.
    fn rotated(&self, &V) -> Self;

    /// Sets the rotation of `self`.
    fn set_rotation(&mut self, V);
}

/// Trait of objects able to rotate other objects.
///
/// This is typically implemented by matrices which rotate vectors.
pub trait Rotate<V> {
    /// Applies a rotation to `v`.
    fn rotate(&self, v: &V) -> V;
    /// Applies an inverse rotation to `v`.
    fn inv_rotate(&self, v: &V) -> V;
}

/// Trait of object which represent a transformation, and to which new transformations can
/// be appended.
///
/// A transformation is assumed to be an isometry without reflexion.
pub trait Transformation<M> {
    /// Gets the transformation of `self`.
    fn transformation(&self) -> M;

    /// Gets the inverse transformation of `self`.
    fn inv_transformation(&self) -> M;

    /// In-place version of `transformed`.
    fn transform_by(&mut self, &M);

    /// Appends a transformation to `self`.
    fn transformed(&self, &M) -> Self;

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
    fn inv_transform(&self, &V) -> V;
}

/// Trait of transformation having a rotation extractable as a rotation matrix. This can typically
/// be implemented by quaternions to convert them to a rotation matrix.
pub trait RotationMatrix<LV, AV, R: Mat<LV, LV> + Rotation<AV>> : Rotation<AV> {
    /// Gets the rotation matrix represented by `self`.
    fn to_rot_mat(&self) -> R;
}

/// Traits of objects having a dot product.
pub trait Dot<N> {
    /// Computes the dot (inner) product of two vectors.
    #[inline]
    fn dot(&self, &Self) -> N;

    /**
     * Short-cut to compute the projection of a point on a vector, but without
     * computing intermediate vectors.
     * This must be equivalent to:
     *
     *   (a - b).dot(c)
     *
     */
    #[inline]
    fn sub_dot(&self, b: &Self, c: &Self) -> N;
}

/// Traits of objects having an euclidian norm.
pub trait Norm<N: Algebraic> {
    /// Computes the norm of `self`.
    #[inline]
    fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    /// Computes the squared norm of `self`.
    ///
    /// This is usually faster than computing the norm itself.
    #[inline]
    fn sqnorm(&self) -> N;

    /// Gets the normalized version of `self`.
    #[inline]
    fn normalized(&self) -> Self;

    /// In-place version of `normalized`.
    #[inline]
    fn normalize(&mut self) -> N;
}

/**
 * Trait of elements having a cross product.
 */
pub trait Cross<V> {
    /// Computes the cross product between two elements (usually vectors).
    fn cross(&self, other: &Self) -> V;
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

/**
 * Trait of elements having a cross product operation which can be expressed as a matrix.
 */
pub trait CrossMatrix<M> {
    /// The matrix associated to any cross product with this vector. I.e. `v.cross(anything)` =
    /// `v.cross_matrix().rmul(anything)`.
    fn cross_matrix(&self) -> M;
}

/// Composition of a rotation and an absolute value.
///
/// The operation is accessible using the `RotationMatrix`, `Absolute`, and `RMul` traits, but
/// doing so is not easy in generic code as it can be a cause of type over-parametrization.
pub trait AbsoluteRotate<V> {
    /// This is the same as:
    ///
    /// ~~~{.rust}
    ///     self.rotation_matrix().absolute().rmul(v)
    /// ~~~
    fn absolute_rotate(&self, v: &V) -> V;
}

/// Trait of vectors able to sample a unit sphere.
///
/// The number of sample must be sufficient to approximate a sphere using a support mapping
/// function.
pub trait UniformSphereSample {
    /// Iterate through the samples.
    fn sample(&fn(Self));
}

/// Various composition of rotation and translation.
///
/// Utilities to make rotations with regard to a point different than the origin.  All those
/// operations are the composition of rotations and translations.
///
/// Those operations are automatically implemented in term of the `Rotation` and `Translation`
/// traits.
pub trait RotationWithTranslation<LV: Neg<LV>, AV>: Rotation<AV> + Translation<LV> {
    /// Applies a rotation centered on a specific point.
    ///
    /// # Arguments
    ///   * `m` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    ///   * `point` - the center of rotation.
    #[inline]
    fn rotated_wrt_point(&self, amount: &AV, center: &LV) -> Self {
        let mut res = self.translated(&-center);

        res.rotate_by(amount);
        res.translate_by(center);

        res
    }

    /// Rotates `self` using a specific center of rotation.
    ///
    /// The rotation is applied in-place.
    ///
    /// # Arguments
    ///   * `m` - the object to be rotated
    ///   * `amount` - the rotation to be applied
    ///   * `center` - the new center of rotation
    #[inline]
    fn rotate_wrt_point(&mut self, amount: &AV, center: &LV) {
        self.translate_by(&-center);
        self.rotate_by(amount);
        self.translate_by(center);
    }

    /// Applies a rotation centered on the translation of `m`.
    /// 
    /// # Arguments
    ///   * `m` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn rotated_wrt_center(&self, amount: &AV) -> Self {
        self.rotated_wrt_point(amount, &self.translation())
    }

    /// Applies a rotation centered on the translation of `m`.
    ///
    /// The rotation os applied on-place.
    ///
    /// # Arguments
    ///   * `m` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn rotate_wrt_center(&mut self, amount: &AV) {
        let center = self.translation();
        self.rotate_wrt_point(amount, &center)
    }
}

impl<LV: Neg<LV>, AV, M: Rotation<AV> + Translation<LV>> RotationWithTranslation<LV, AV> for M {
}
