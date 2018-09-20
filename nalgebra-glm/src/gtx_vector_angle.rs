use na::{self, DefaultAllocator, Real, U2, U3};

use traits::{Dimension, Alloc};
use aliases::Vec;


pub fn angle<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.angle(y)
}

pub fn oriented_angle<N: Real>(x: &Vec<N, U2>, y: &Vec<N, U2>) -> N {
    unimplemented!()
}

pub fn oriented_angle_ref<N: Real>(x: &Vec<N, U3>, y: &Vec<N, U3>, refv: &Vec<N, U3>) -> N {
    unimplemented!()
}
