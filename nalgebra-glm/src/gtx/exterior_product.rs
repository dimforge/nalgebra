use na::U2;

use traits::Number;
use aliases::Vec;

/// The 2D perpendicular product between two vectors.
pub fn cross<N: Number>(v: &Vec<N, U2>, u: &Vec<N, U2>) -> N {
    v.perp(u)
}