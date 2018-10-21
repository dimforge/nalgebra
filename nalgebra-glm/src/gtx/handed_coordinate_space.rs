use aliases::TVec3;
use traits::Number;

/// Returns `true` if `{a, b, c}` forms a left-handed trihedron.
///
/// # See also:
///
/// * [`right_handed`](fn.right_handed.html)
pub fn left_handed<N: Number>(a: &TVec3<N>, b: &TVec3<N>, c: &TVec3<N>) -> bool {
    a.cross(b).dot(c) < N::zero()
}

/// Returns `true` if `{a, b, c}` forms a right-handed trihedron.
///
/// # See also:
///
/// * [`left_handed`](fn.left_handed.html)
pub fn right_handed<N: Number>(a: &TVec3<N>, b: &TVec3<N>, c: &TVec3<N>) -> bool {
    a.cross(b).dot(c) > N::zero()
}
