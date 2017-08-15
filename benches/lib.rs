#![feature(test)]
#![allow(unused_macros)]

extern crate test;
extern crate rand;
extern crate typenum;
extern crate nalgebra as na;


use rand::{Rng, IsaacRng};
use na::DMatrix;


mod core;
mod linalg;
mod geometry;

fn reproductible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    let mut rng = IsaacRng::new_unseeded();
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}
