use na::{DMatrix, DVector, LU};

// Without unpack.
fn lu_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    bh.bench_function("lu_decompose_10x10", move |bh| {
        bh.iter(|| test::black_box(LU::new(m.clone())))
    });
}

fn lu_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("lu_decompose_100x100", move |bh| {
        bh.iter(|| test::black_box(LU::new(m.clone())))
    });
}

fn lu_decompose_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("lu_decompose_500x500", move |bh| {
        bh.iter(|| test::black_box(LU::new(m.clone())))
    });
}

fn lu_solve_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_solve_10x10", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(10, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn lu_solve_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_solve_100x100", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(100, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn lu_solve_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = LU::new(m.clone());

    bh.bench_function("", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(500, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn lu_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_inverse_10x10", move |bh| {
        bh.iter(|| test::black_box(lu.try_inverse()))
    });
}

fn lu_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_inverse_100x100", move |bh| {
        bh.iter(|| test::black_box(lu.try_inverse()))
    });
}

fn lu_inverse_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_inverse_500x500", move |bh| {
        bh.iter(|| test::black_box(lu.try_inverse()))
    });
}

fn lu_determinant_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_determinant_10x10", move |bh| {
        bh.iter(|| test::black_box(lu.determinant()))
    });
}

fn lu_determinant_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = LU::new(m.clone());

    bh.bench_function("lu_determinant_100x100", move |bh| {
        bh.iter(|| test::black_box(lu.determinant()))
    });
}

fn lu_determinant_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = LU::new(m.clone());

    bh.bench_function("", move |bh| bh.iter(|| test::black_box(lu.determinant())));
}

criterion_group!(
    lu,
    lu_decompose_10x10,
    lu_decompose_100x100,
    //    lu_decompose_500x500,
    lu_solve_10x10,
    lu_solve_100x100,
    lu_inverse_10x10,
    lu_inverse_100x100,
    //    lu_inverse_500x500,
    lu_determinant_10x10,
    lu_determinant_100x100
);
