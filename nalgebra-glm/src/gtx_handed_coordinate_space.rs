use na::U3;

use traits::Number;
use aliases::Vec;

pub fn left_handed<N: Number>(a: &Vec<N, U3>, b: &Vec<N, U3>, c: &Vec<N, U3>) -> bool {
    a.cross(b).dot(c) < N::zero()
}

pub fn right_handed<N: Number>(a: &Vec<N, U3>, b: &Vec<N, U3>, c: &Vec<N, U3>) -> bool {
    a.cross(b).dot(c) > N::zero()
}
