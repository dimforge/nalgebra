use test::{self, Bencher};
use na::{Matrix4, SVD};

#[bench]
fn svd_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(SVD::new(m.clone(), true, true)))
}

#[bench]
fn svd_decompose_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(SVD::new(m.clone(), true, true)))
}

#[bench]
fn svd_decompose_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(SVD::new(m.clone(), true, true)))
}

#[bench]
fn svd_decompose_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(SVD::new(m.clone(), true, true)))
}

#[bench]
fn rank_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(m.rank(1.0e-10)))
}

#[bench]
fn rank_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(m.rank(1.0e-10)))
}

#[bench]
fn rank_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(m.rank(1.0e-10)))
}

#[bench]
fn rank_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(m.rank(1.0e-10)))
}

#[bench]
fn singular_values_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(m.singular_values()))
}

#[bench]
fn singular_values_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(m.singular_values()))
}

#[bench]
fn singular_values_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(m.singular_values()))
}

#[bench]
fn singular_values_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(m.singular_values()))
}

#[bench]
fn pseudo_inverse_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(m.clone().pseudo_inverse(1.0e-10)))
}

#[bench]
fn pseudo_inverse_10x10(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(10, 10);
    bh.iter(|| test::black_box(m.clone().pseudo_inverse(1.0e-10)))
}

#[bench]
fn pseudo_inverse_100x100(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(100, 100);
    bh.iter(|| test::black_box(m.clone().pseudo_inverse(1.0e-10)))
}

#[bench]
fn pseudo_inverse_200x200(bh: &mut Bencher) {
    let m = ::reproductible_dmatrix(200, 200);
    bh.iter(|| test::black_box(m.clone().pseudo_inverse(1.0e-10)))
}
