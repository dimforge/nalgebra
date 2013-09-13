use traits::mat::Mat;

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

/// Trait of transformation having a rotation extractable as a rotation matrix.  This can typically
/// be implemented by quaternions to convert them 
pub trait RotationMatrix<LV, AV, R: Mat<LV, LV> + Rotation<AV>> : Rotation<AV> {
    /// Gets the rotation matrix from this object.
    fn to_rot_mat(&self) -> R;
}
