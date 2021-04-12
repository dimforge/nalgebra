use na::{DMatrix, Matrix4};
use nl::QR;
use test::{self, Bencher};

#[bench]
fn qr_decompose_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| std::hint::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_100x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.iter(|| std::hint::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| std::hint::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_500x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.iter(|| std::hint::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| std::hint::black_box(QR::new(m.clone())))
}
