use na::{DMatrix, DVector};

fn solve_l_triangular_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.bench_function("solve_l_triangular_100x100", move |bh| {
        bh.iter(|| {
            let _ = m.solve_lower_triangular(&v);
        })
    });
}

fn solve_l_triangular_1000x1000(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.bench_function("solve_l_triangular_1000x1000", move |bh| {
        bh.iter(|| {
            let _ = m.solve_lower_triangular(&v);
        })
    });
}

fn tr_solve_l_triangular_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.bench_function("tr_solve_l_triangular_100x100", move |bh| {
        bh.iter(|| {
            let _ = m.tr_solve_lower_triangular(&v);
        })
    });
}

fn tr_solve_l_triangular_1000x1000(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.bench_function("tr_solve_l_triangular_1000x1000", move |bh| {
        bh.iter(|| {
            let _ = m.tr_solve_lower_triangular(&v);
        })
    });
}

fn solve_u_triangular_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.bench_function("solve_u_triangular_100x100", move |bh| {
        bh.iter(|| {
            let _ = m.solve_upper_triangular(&v);
        })
    });
}

fn solve_u_triangular_1000x1000(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.bench_function("solve_u_triangular_1000x1000", move |bh| {
        bh.iter(|| {
            let _ = m.solve_upper_triangular(&v);
        })
    });
}

fn tr_solve_u_triangular_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.bench_function("tr_solve_u_triangular_100x100", move |bh| {
        bh.iter(|| {
            let _ = m.tr_solve_upper_triangular(&v);
        })
    });
}

fn tr_solve_u_triangular_1000x1000(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.bench_function("tr_solve_u_triangular_1000x1000", move |bh| {
        bh.iter(|| {
            let _ = m.tr_solve_upper_triangular(&v);
        })
    });
}

criterion_group!(
    solve,
    solve_l_triangular_100x100,
    solve_l_triangular_1000x1000,
    tr_solve_l_triangular_100x100,
    tr_solve_l_triangular_1000x1000,
    solve_u_triangular_100x100,
    solve_u_triangular_1000x1000,
    tr_solve_u_triangular_100x100,
    tr_solve_u_triangular_1000x1000
);
