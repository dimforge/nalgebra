use na::U2;

use traits::Number;
use aliases::Vec;

pub fn cross<N: Number>(v: &Vec<N, U2>, u: &Vec<N, U2>) -> N {
    v.perp(u)
}