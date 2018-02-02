use test::{self, Bencher};
use na::{Matrix4, SymmetricEigen};

#[bench]
fn symmetric_eigen_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(SymmetricEigen::new(m.clone())))
}

#[bench]
fn symmetric_eigen_decompose_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(SymmetricEigen::new(m.clone())))
}

#[bench]
fn symmetric_eigen_decompose_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(SymmetricEigen::new(m.clone())))
}

#[bench]
fn symmetric_eigen_decompose_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(SymmetricEigen::new(m.clone())))
}
