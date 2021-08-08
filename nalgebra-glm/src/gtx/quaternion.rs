use na::{Rotation3, Unit, UnitQuaternion};

use crate::aliases::{Qua, TMat3, TMat4, TVec3, TVec4};
use crate::RealNumber;

/// Rotate the vector `v` by the quaternion `q` assumed to be normalized.
pub fn quat_cross_vec<T: RealNumber>(q: &Qua<T>, v: &TVec3<T>) -> TVec3<T> {
    UnitQuaternion::new_unchecked(*q) * v
}

/// Rotate the vector `v` by the inverse of the quaternion `q` assumed to be normalized.
pub fn quat_inv_cross_vec<T: RealNumber>(v: &TVec3<T>, q: &Qua<T>) -> TVec3<T> {
    UnitQuaternion::new_unchecked(*q).inverse() * v
}

/// The quaternion `w` component.
pub fn quat_extract_real_component<T: RealNumber>(q: &Qua<T>) -> T {
    q.w
}

/// Normalized linear interpolation between two quaternions.
pub fn quat_fast_mix<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    Unit::new_unchecked(*x)
        .nlerp(&Unit::new_unchecked(*y), a)
        .into_inner()
}

//pub fn quat_intermediate<T: RealNumber>(prev: &Qua<T>, curr: &Qua<T>, next: &Qua<T>) -> Qua<T> {
//    unimplemented!()
//}

/// The squared magnitude of a quaternion `q`.
pub fn quat_length2<T: RealNumber>(q: &Qua<T>) -> T {
    q.norm_squared()
}

/// The squared magnitude of a quaternion `q`.
pub fn quat_magnitude2<T: RealNumber>(q: &Qua<T>) -> T {
    q.norm_squared()
}

/// The quaternion representing the identity rotation.
pub fn quat_identity<T: RealNumber>() -> Qua<T> {
    UnitQuaternion::identity().into_inner()
}

/// Rotates a vector by a quaternion assumed to be normalized.
pub fn quat_rotate_vec3<T: RealNumber>(q: &Qua<T>, v: &TVec3<T>) -> TVec3<T> {
    UnitQuaternion::new_unchecked(*q) * v
}

/// Rotates a vector in homogeneous coordinates by a quaternion assumed to be normalized.
pub fn quat_rotate_vec<T: RealNumber>(q: &Qua<T>, v: &TVec4<T>) -> TVec4<T> {
    let rotated = Unit::new_unchecked(*q) * v.fixed_rows::<3>(0);
    TVec4::new(rotated.x, rotated.y, rotated.z, v.w)
}

/// The rotation required to align `orig` to `dest`.
pub fn quat_rotation<T: RealNumber>(orig: &TVec3<T>, dest: &TVec3<T>) -> Qua<T> {
    UnitQuaternion::rotation_between(orig, dest)
        .unwrap_or_else(UnitQuaternion::identity)
        .into_inner()
}

/// The spherical linear interpolation between two quaternions.
pub fn quat_short_mix<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    Unit::new_normalize(*x)
        .slerp(&Unit::new_normalize(*y), a)
        .into_inner()
}

//pub fn quat_squad<T: RealNumber>(q1: &Qua<T>, q2: &Qua<T>, s1: &Qua<T>, s2: &Qua<T>, h: T) -> Qua<T> {
//    unimplemented!()
//}

/// Converts a quaternion to a rotation matrix.
pub fn quat_to_mat3<T: RealNumber>(x: &Qua<T>) -> TMat3<T> {
    UnitQuaternion::new_unchecked(*x)
        .to_rotation_matrix()
        .into_inner()
}

/// Converts a quaternion to a rotation matrix in homogenous coordinates.
pub fn quat_to_mat4<T: RealNumber>(x: &Qua<T>) -> TMat4<T> {
    UnitQuaternion::new_unchecked(*x).to_homogeneous()
}

/// Converts a rotation matrix to a quaternion.
pub fn mat3_to_quat<T: RealNumber>(x: &TMat3<T>) -> Qua<T> {
    let r = Rotation3::from_matrix_unchecked(*x);
    UnitQuaternion::from_rotation_matrix(&r).into_inner()
}

/// Converts a rotation matrix in homogeneous coordinates to a quaternion.
pub fn to_quat<T: RealNumber>(x: &TMat4<T>) -> Qua<T> {
    let rot = x.fixed_slice::<3, 3>(0, 0).into_owned();
    mat3_to_quat(&rot)
}
