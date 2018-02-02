use test::{self, Bencher};
use na::{Matrix4, RealSchur};

#[bench]
fn schur_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(RealSchur::new(m.clone())))
}

#[bench]
fn schur_decompose_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(RealSchur::new(m.clone())))
}

#[bench]
fn schur_decompose_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(RealSchur::new(m.clone())))
}

#[bench]
fn schur_decompose_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(RealSchur::new(m.clone())))
}

#[bench]
fn eigenvalues_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(m.complex_eigenvalues()))
}

#[bench]
fn eigenvalues_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(m.complex_eigenvalues()))
}

#[bench]
fn eigenvalues_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(m.complex_eigenvalues()))
}

#[bench]
fn eigenvalues_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(m.complex_eigenvalues()))
}
