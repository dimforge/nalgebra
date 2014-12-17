#![feature(macro_rules)]

extern crate test;
extern crate "nalgebra" as na;

use test::Bencher;
use na::{DVec, DMat};

macro_rules! bench_mul_dmat(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {
            $bh.iter(|| {
                let a:     DMat<f64> = DMat::new_random($nrows, $ncols);
                let mut b: DMat<f64> = DMat::new_random($nrows, $ncols);

                for _ in range(0u, 1000) {
                    // XXX: the clone here is highly undesirable!
                    b = a.clone() * b;
                }
            })
        }
    }
)

#[bench]
fn bench_mul_dmat2(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 2, 2)
}

#[bench]
fn bench_mul_dmat3(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 3, 3)
}

#[bench]
fn bench_mul_dmat4(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 4, 4)
}

#[bench]
fn bench_mul_dmat5(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 5, 5)
}

#[bench]
fn bench_mul_dmat6(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 6, 6)
}

macro_rules! bench_mul_dmat_dvec(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {

            $bh.iter(|| {
                let m : DMat<f64>     = DMat::new_random($nrows, $ncols);
                let mut v : DVec<f64> = DVec::new_random($ncols);

                for _ in range(0u, 1000) {
                    // XXX: the clone here is highly undesirable!
                    v = m.clone() * v
                }
            })
        }
    }
)

#[bench]
fn bench_mul_dmat_dvec2(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 2, 2)
}

#[bench]
fn bench_mul_dmat_dvec3(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 3, 3)
}

#[bench]
fn bench_mul_dmat_dvec4(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 4, 4)
}

#[bench]
fn bench_mul_dmat_dvec5(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 5, 5)
}

#[bench]
fn bench_mul_dmat_dvec6(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 6, 6)
}
