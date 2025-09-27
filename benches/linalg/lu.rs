use na::{DMatrix, DVector, LU};

// Without unpack.
fn lu_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    bh.bench_function("lu_decompose_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(LU::new(m.clone())))
    });
}

fn lu_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("lu_decompose_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(LU::new(m.clone())))
    });
}

fn lu_solve_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_solve_10x10", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(10, 1.0);
            lu.solve_mut(&mut b);
            b
        })
    });
}

fn lu_solve_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_solve_100x100", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(100, 1.0);
            lu.solve_mut(&mut b);
            b
        })
    });
}

fn lu_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_inverse_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(lu.try_inverse()))
    });
}

fn lu_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_inverse_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(lu.try_inverse()))
    });
}

fn lu_determinant_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_determinant_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(lu.determinant()))
    });
}

fn lu_determinant_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_determinant_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(lu.determinant()))
    });
}

criterion_group!(
    lu,
    lu_decompose_10x10,
    lu_decompose_100x100,
    lu_solve_10x10,
    lu_solve_100x100,
    lu_inverse_10x10,
    lu_inverse_100x100,
    lu_determinant_10x10,
    lu_determinant_100x100,
);
