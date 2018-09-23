use traits::Number;
use aliases::TVec3;

/// Returns `true` if `{a, b, c}` forms a left-handed trihedron.
pub fn left_handed<N: Number>(a: &TVec3<N>, b: &TVec3<N>, c: &TVec3<N>) -> bool {
    a.cross(b).dot(c) < N::zero()
}

/// Returns `true` if `{a, b, c}` forms a right-handed trihedron.
pub fn right_handed<N: Number>(a: &TVec3<N>, b: &TVec3<N>, c: &TVec3<N>) -> bool {
    a.cross(b).dot(c) > N::zero()
}
