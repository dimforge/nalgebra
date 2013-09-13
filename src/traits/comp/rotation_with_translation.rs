use traits::rotation::Rotation;
use traits::translation::Translation;

// NOTE: we cannot call this an Isometry since an isometry really does not need to have a rotation
// nor a translation (this can be a reflexion).
/// Utilities to make rotations with regard to a point different than the origin.
/// All those operations are the composition of rotations and translations.
/// This could be implemented in term of the `Rotation` and `Translation` traits, but having those
/// here make it easier to use.
///
/// # Known use case:
///     * to change the center of a rotation.
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
