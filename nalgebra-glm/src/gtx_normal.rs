use na::{Real, U3};

use aliases::Vec;

pub fn triangleNormal<N: Real>(p1: &Vec<N, U3>, p2: &Vec<N, U3>, p3: &Vec<N, U3>) -> Vec<N, U3> {
    (p2 - p1).cross(&(p3 - p1)).normalize()
}