extern crate alga;
extern crate nalgebra as na;

use alga::general::{Real, RingCommutative};
use na::{Scalar, Vector3};

fn print_vector<N: Scalar>(m: &Vector3<N>) {
    println!("{:?}", m)
}

fn print_squared_norm<N: Scalar + RingCommutative>(v: &Vector3<N>) {
    // NOTE: alternatively, nalgebra already defines `v.squared_norm()`.
    let sqnorm = v.dot(v);
    println!("{:?}", sqnorm);
}

fn print_norm<N: Real>(v: &Vector3<N>) {
    // NOTE: alternatively, nalgebra already defines `v.norm()`.
    let norm = v.dot(v).sqrt();

    // The Real bound implies that N is Display so we can
    // use "{}" instead of "{:?}" for the format string.
    println!("{}", norm)
}

fn main() {
    let v1 = Vector3::new(1, 2, 3);
    let v2 = Vector3::new(1.0, 2.0, 3.0);

    print_vector(&v1);
    print_squared_norm(&v1);
    print_norm(&v2);
}
