use na::U3;

use traits::Number;
use aliases::Vec;

/// Returns `true` if `{a, b, c}` forms a left-handed trihedron.
pub fn left_handed<N: Number>(a: &Vec<N, U3>, b: &Vec<N, U3>, c: &Vec<N, U3>) -> bool {
    a.cross(b).dot(c) < N::zero()
}

/// Returns `true` if `{a, b, c}` forms a right-handed trihedron.
pub fn right_handed<N: Number>(a: &Vec<N, U3>, b: &Vec<N, U3>, c: &Vec<N, U3>) -> bool {
    a.cross(b).dot(c) > N::zero()
}
