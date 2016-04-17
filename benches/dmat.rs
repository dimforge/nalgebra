#![feature(test)]

extern crate test;
extern crate nalgebra as na;

use test::Bencher;
use na::{DVector, DMatrix};

macro_rules! bench_mul_dmatrix(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {
            $bh.iter(|| {
                let a:     DMatrix<f64> = DMatrix::new_random($nrows, $ncols);
                let mut b: DMatrix<f64> = DMatrix::new_random($nrows, $ncols);

                for _ in 0usize .. 1000 {
                    // XXX: the clone here is highly undesirable!
                    b = a.clone() * b;
                }
            })
        }
    }
);

#[bench]
fn bench_mul_dmat2(bh: &mut Bencher) {
    bench_mul_dmatrix!(bh, 2, 2);
}

#[bench]
fn bench_mul_dmat3(bh: &mut Bencher) {
    bench_mul_dmatrix!(bh, 3, 3);
}

#[bench]
fn bench_mul_dmat4(bh: &mut Bencher) {
    bench_mul_dmatrix!(bh, 4, 4);
}

#[bench]
fn bench_mul_dmat5(bh: &mut Bencher) {
    bench_mul_dmatrix!(bh, 5, 5);
}

#[bench]
fn bench_mul_dmat6(bh: &mut Bencher) {
    bench_mul_dmatrix!(bh, 6, 6);
}

macro_rules! bench_mul_dmat_dvector(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {

            $bh.iter(|| {
                let m : DMatrix<f64>     = DMatrix::new_random($nrows, $ncols);
                let mut v : DVector<f64> = DVector::new_random($ncols);

                for _ in 0usize .. 1000 {
                    // XXX: the clone here is highly undesirable!
                    v = m.clone() * v
                }
            })
        }
    }
);

#[bench]
fn bench_mul_dmat_dvec2(bh: &mut Bencher) {
    bench_mul_dmat_dvector!(bh, 2, 2);
}

#[bench]
fn bench_mul_dmat_dvec3(bh: &mut Bencher) {
    bench_mul_dmat_dvector!(bh, 3, 3);
}

#[bench]
fn bench_mul_dmat_dvec4(bh: &mut Bencher) {
    bench_mul_dmat_dvector!(bh, 4, 4);
}

#[bench]
fn bench_mul_dmat_dvec5(bh: &mut Bencher) {
    bench_mul_dmat_dvector!(bh, 5, 5);
}

#[bench]
fn bench_mul_dmat_dvec6(bh: &mut Bencher) {
    bench_mul_dmat_dvector!(bh, 6, 6);
}
