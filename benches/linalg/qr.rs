use test::{self, Bencher};
use na::{DMatrix, DVector, Matrix4, QR};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
#[bench]
fn qr_decompose_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| test::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_100x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.iter(|| test::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_500x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.iter(|| test::black_box(QR::new(m.clone())))
}

#[bench]
fn qr_decompose_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| test::black_box(QR::new(m.clone())))
}

// With unpack.
#[bench]
fn qr_decompose_unpack_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| {
        let qr = QR::new(m.clone());
        let _ = qr.unpack();
    })
}

#[bench]
fn qr_decompose_unpack_100x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.iter(|| {
        let qr = QR::new(m.clone());
        let _ = qr.unpack();
    })
}

#[bench]
fn qr_decompose_unpack_500x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.iter(|| {
        let qr = QR::new(m.clone());
        let _ = qr.unpack();
    })
}

#[bench]
fn qr_decompose_unpack_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| {
        let qr = QR::new(m.clone());
        let _ = qr.unpack();
    })
}

#[bench]
fn qr_solve_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let qr = QR::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(10, 1.0);
        qr.solve(&mut b);
    })
}

#[bench]
fn qr_solve_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let qr = QR::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(100, 1.0);
        qr.solve(&mut b);
    })
}

#[bench]
fn qr_solve_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let qr = QR::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(500, 1.0);
        qr.solve(&mut b);
    })
}

#[bench]
fn qr_inverse_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let qr = QR::new(m.clone());

    bh.iter(|| test::black_box(qr.try_inverse()))
}

#[bench]
fn qr_inverse_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let qr = QR::new(m.clone());

    bh.iter(|| test::black_box(qr.try_inverse()))
}

#[bench]
fn qr_inverse_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let qr = QR::new(m.clone());

    bh.iter(|| test::black_box(qr.try_inverse()))
}
