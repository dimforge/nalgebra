#![feature(test)]
#![allow(unused_macros)]

extern crate nalgebra as na;

#[macro_use]
extern crate criterion;

use na::DMatrix;
use rand::{Rng, SeedableRng};

pub mod core;
pub mod geometry;
pub mod linalg;

fn reproductible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    let mut rng = rand_isaac::IsaacRng::seed_from_u64(0);
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}

criterion_main!(
    core::matrix,
    core::vector,
    geometry::quaternion,
    linalg::bidiagonal,
    linalg::cholesky,
    linalg::full_piv_lu,
    linalg::hessenberg,
    linalg::lu,
    linalg::qr,
    linalg::schur,
    linalg::solve,
    linalg::svd,
    linalg::symmetric_eigen,
);
