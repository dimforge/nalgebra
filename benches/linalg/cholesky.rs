use na::{Cholesky, DMatrix, DVector};

fn cholesky_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();

    bh.bench_function("cholesky_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(Cholesky::new(m.clone())))
    });
}

fn cholesky_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();

    bh.bench_function("cholesky_500x500", move |bh| {
        bh.iter(|| std::hint::black_box(Cholesky::new(m.clone())))
    });
}

// With unpack.
fn cholesky_decompose_unpack_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();

    bh.bench_function("cholesky_decompose_unpack_100x100", move |bh| {
        bh.iter(|| {
            let chol = Cholesky::new(m.clone()).unwrap();
            let _ = chol.unpack();
        })
    });
}
fn cholesky_decompose_unpack_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();

    bh.bench_function("cholesky_decompose_unpack_500x500", move |bh| {
        bh.iter(|| {
            let chol = Cholesky::new(m.clone()).unwrap();
            let _ = chol.unpack();
        })
    });
}

fn cholesky_solve_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(10);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_solve_10x10", move |bh| {
        bh.iter(|| {
            let _ = chol.solve(&v);
        })
    });
}

fn cholesky_solve_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(100);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_solve_100x100", move |bh| {
        bh.iter(|| {
            let _ = chol.solve(&v);
        })
    });
}

fn cholesky_solve_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(500);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_solve_500x500", move |bh| {
        bh.iter(|| {
            let _ = chol.solve(&v);
        })
    });
}

fn cholesky_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_inverse_10x10", move |bh| {
        bh.iter(|| {
            let _ = chol.inverse();
        })
    });
}

fn cholesky_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_inverse_100x100", move |bh| {
        bh.iter(|| {
            let _ = chol.inverse();
        })
    });
}

fn cholesky_inverse_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.bench_function("cholesky_inverse_500x500", move |bh| {
        bh.iter(|| {
            let _ = chol.inverse();
        })
    });
}

criterion_group!(
    cholesky,
    cholesky_100x100,
    cholesky_500x500,
    cholesky_decompose_unpack_100x100,
    cholesky_decompose_unpack_500x500,
    cholesky_solve_10x10,
    cholesky_solve_100x100,
    cholesky_solve_500x500,
    cholesky_inverse_10x10,
    cholesky_inverse_100x100,
    cholesky_inverse_500x500
);
