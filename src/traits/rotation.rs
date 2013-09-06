use traits::translation::Translation;

/// Trait of object which represent a rotation, and to wich new rotations can
/// be appended. A rotation is assumed to be an isomitry without translation
/// and without reflexion.
pub trait Rotation<V> {
    /// Gets the rotation associated with this object.
    fn rotation(&self) -> V;

    /// Gets the inverse rotation associated with this object.
    fn inv_rotation(&self) -> V;

    /// In-place version of `rotated`.
    fn rotate_by(&mut self, &V);

    /// Appends a rotation.
    fn rotated(&self, &V) -> Self;

    /// Sets the rotation.
    fn set_rotation(&mut self, V);
}

/// Trait of objects able to rotate other objects. This is typically implemented by matrices which
/// rotate vectors.
pub trait Rotate<V> {
    /// Apply a rotation to an object.
    fn rotate(&self, &V)     -> V;
    /// Apply an inverse rotation to an object.
    fn inv_rotate(&self, &V) -> V;
}

/// Utilities to make rotations with regard to a point different than the origin.
// NOTE: we cannot call this an Isometry since an isometry really does not need to have a rotation
// nor a translation (this can be a reflexion).
pub trait RotationWithTranslation<LV: Neg<LV>, AV>: Rotation<AV> + Translation<LV> {
    /**
     * Applies a rotation centered on a specific point.
     *
     *   - `m`:       the object to be rotated.
     *   - `ammount`: the rotation to apply.
     *   - `point`:   the center of rotation.
     */
    #[inline]
    fn rotated_wrt_point(&self, ammount: &AV, center: &LV) -> Self {
        let mut res = self.translated(&-center);

        res.rotate_by(ammount);
        res.translate_by(center);

        res
    }

    /// Rotates an object using a specific center of rotation.
    ///
    /// # Arguments
    ///   * `m` - the object to be rotated
    ///   * `ammount` - the rotation to be applied
    ///   * `center` - the new center of rotation
    #[inline]
    fn rotate_wrt_point(&mut self, ammount: &AV, center: &LV) {
        self.translate_by(&-center);
        self.rotate_by(ammount);
        self.translate_by(center);
    }

    /**
     * Applies a rotation centered on the input translation.
     *
     * # Arguments
     *   * `m` - the object to be rotated.
     *   * `ammount` - the rotation to apply.
     */
    #[inline]
    fn rotated_wrt_center(&self, ammount: &AV) -> Self {
        self.rotated_wrt_point(ammount, &self.translation())
    }

    /**
     * Applies a rotation centered on the input translation.
     *
     * # Arguments
     *   * `m` - the object to be rotated.
     *   * `ammount` - the rotation to apply.
     */
    #[inline]
    fn rotate_wrt_center(&mut self, ammount: &AV) {
        let center = self.translation();
        self.rotate_wrt_point(ammount, &center)
    }
}

impl<LV: Neg<LV>, AV, M: Rotation<AV> + Translation<LV>> RotationWithTranslation<LV, AV> for M;
