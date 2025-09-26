use na::{DMatrix, DVector, Matrix4, QR};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
fn qr_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("qr_decompose_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(QR::new(m.clone())))
    });
}

fn qr_decompose_100x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.bench_function("qr_decompose_100x500", move |bh| {
        bh.iter(|| std::hint::black_box(QR::new(m.clone())))
    });
}

fn qr_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("qr_decompose_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(QR::new(m.clone())))
    });
}

fn qr_decompose_500x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.bench_function("qr_decompose_500x100", move |bh| {
        bh.iter(|| std::hint::black_box(QR::new(m.clone())))
    });
}

// With unpack.
fn qr_decompose_unpack_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("qr_decompose_unpack_100x100", move |bh| {
        bh.iter(|| {
            let qr = QR::new(m.clone());
            let _ = qr.unpack();
        })
    });
}

fn qr_decompose_unpack_100x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.bench_function("qr_decompose_unpack_100x500", move |bh| {
        bh.iter(|| {
            let qr = QR::new(m.clone());
            let _ = qr.unpack();
        })
    });
}

fn qr_decompose_unpack_500x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.bench_function("qr_decompose_unpack_500x100", move |bh| {
        bh.iter(|| {
            let qr = QR::new(m.clone());
            let _ = qr.unpack();
        })
    });
}

fn qr_solve_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let qr = QR::new(m.clone());

    bh.bench_function("qr_solve_10x10", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(10, 1.0);
            qr.solve_mut(&mut b);
            b
        })
    });
}

fn qr_solve_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let qr = QR::new(m.clone());

    bh.bench_function("qr_solve_100x100", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(100, 1.0);
            qr.solve_mut(&mut b);
            b
        })
    });
}

fn qr_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let qr = QR::new(m.clone());

    bh.bench_function("qr_inverse_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(qr.try_inverse()))
    });
}

fn qr_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let qr = QR::new(m.clone());

    bh.bench_function("qr_inverse_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(qr.try_inverse()))
    });
}

criterion_group!(
    qr,
    qr_decompose_100x100,
    qr_decompose_100x500,
    qr_decompose_4x4,
    qr_decompose_500x100,
    qr_decompose_unpack_100x100,
    qr_decompose_unpack_100x500,
    qr_decompose_unpack_500x100,
    qr_solve_10x10,
    qr_solve_100x100,
    qr_inverse_10x10,
    qr_inverse_100x100,
);
