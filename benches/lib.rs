#![feature(test)]
#![allow(unused_macros)]

extern crate nalgebra as na;
extern crate rand;
extern crate test;
extern crate typenum;

use na::DMatrix;
use rand::{IsaacRng, Rng};

mod core;
mod geometry;
mod linalg;

fn reproductible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}
