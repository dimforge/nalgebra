#![allow(unused_macros)]

extern crate nalgebra as na;
extern crate rand_package as rand;

#[macro_use]
extern crate criterion;

use na::{DMatrix, Matrix, Scalar};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use rand_isaac::IsaacRng;

pub mod core;
pub mod geometry;
pub mod linalg;

fn reproducible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}

fn reproducible_matrix<T: Scalar + Default, const R: usize, const C: usize>(
) -> Matrix<T, na::Const<R>, na::Const<C>, na::ArrayStorage<T, R, C>>
where
    [[T; R]; C]: Default,
    Standard: Distribution<T>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    let mut m: Matrix<T, na::Const<R>, na::Const<C>, na::ArrayStorage<T, R, C>> = Matrix::default();
    m.iter_mut().for_each(|x| *x = rng.gen());
    return m;
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
    linalg::matrix_inversion
);
