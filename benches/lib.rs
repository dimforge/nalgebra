#![feature(test)]
#![allow(unused_macros)]

extern crate nalgebra as na;
extern crate rand;
extern crate test;
extern crate typenum;

#[macro_use]
extern crate criterion;

use na::DMatrix;
use rand::{IsaacRng, Rng};

pub mod core;
pub mod geometry;
pub mod linalg;

fn reproductible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}

criterion_main!(core::matrix::matrix, core::vector::vector);