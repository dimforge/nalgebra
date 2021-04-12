use na::{DMatrix, DVector, FullPivLU};

// Without unpack.
fn full_piv_lu_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    bh.bench_function("full_piv_lu_decompose_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(FullPivLU::new(m.clone())))
    });
}

fn full_piv_lu_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("full_piv_lu_decompose_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(FullPivLU::new(m.clone())))
    });
}

fn full_piv_lu_decompose_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("full_piv_lu_decompose_500x500", move |bh| {
        bh.iter(|| std::hint::black_box(FullPivLU::new(m.clone())))
    });
}

fn full_piv_lu_solve_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_solve_10x10", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(10, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn full_piv_lu_solve_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_solve_100x100", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(100, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn full_piv_lu_solve_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_solve_500x500", move |bh| {
        bh.iter(|| {
            let mut b = DVector::<f64>::from_element(500, 1.0);
            lu.solve(&mut b);
        })
    });
}

fn full_piv_lu_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_inverse_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(lu.try_inverse()))
    });
}

fn full_piv_lu_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_inverse_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(lu.try_inverse()))
    });
}

fn full_piv_lu_inverse_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_inverse_500x500", move |bh| {
        bh.iter(|| std::hint::black_box(lu.try_inverse()))
    });
}

fn full_piv_lu_determinant_10x10(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_determinant_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(lu.determinant()))
    });
}

fn full_piv_lu_determinant_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_determinant_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(lu.determinant()))
    });
}

fn full_piv_lu_determinant_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.bench_function("full_piv_lu_determinant_500x500", move |bh| {
        bh.iter(|| std::hint::black_box(lu.determinant()))
    });
}

criterion_group!(
    full_piv_lu,
    full_piv_lu_decompose_10x10,
    full_piv_lu_decompose_100x100,
    //    full_piv_lu_decompose_500x500,
    full_piv_lu_solve_10x10,
    full_piv_lu_solve_100x100,
    //    full_piv_lu_solve_500x500,
    full_piv_lu_inverse_10x10,
    full_piv_lu_inverse_100x100,
    //    full_piv_lu_inverse_500x500,
    full_piv_lu_determinant_10x10,
    full_piv_lu_determinant_100x100,
    //    full_piv_lu_determinant_500x500
);
