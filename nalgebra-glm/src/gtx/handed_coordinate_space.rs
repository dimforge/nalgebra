use crate::aliases::TVec3;
use crate::traits::Number;

/// Returns `true` if `{a, b, c}` forms a left-handed trihedron.
///
/// # See also:
///
/// * [`right_handed`](fn.right_handed.html)
pub fn left_handed<T: Number>(a: &TVec3<T>, b: &TVec3<T>, c: &TVec3<T>) -> bool {
    a.cross(b).dot(c) < T::zero()
}

/// Returns `true` if `{a, b, c}` forms a right-handed trihedron.
///
/// # See also:
///
/// * [`left_handed`](fn.left_handed.html)
pub fn right_handed<T: Number>(a: &TVec3<T>, b: &TVec3<T>, c: &TVec3<T>) -> bool {
    a.cross(b).dot(c) > T::zero()
}
