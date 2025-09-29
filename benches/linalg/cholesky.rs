use na::{Cholesky, DVector};

fn cholesky_100x100(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_100x100", |bh| {
        bh.iter_batched(
            || {
                let m = crate::reproducible_dmatrix(100, 100);
                &m * m.transpose()
            },
            |m| Cholesky::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_500x500(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_500x500", |bh| {
        bh.iter_batched(
            || {
                let m = crate::reproducible_dmatrix(500, 500);
                &m * m.transpose()
            },
            |m| Cholesky::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

// With unpack.
fn cholesky_decompose_unpack_100x100(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_decompose_unpack_100x100", |bh| {
        bh.iter_batched(
            || {
                let m = crate::reproducible_dmatrix(100, 100);
                &m * m.transpose()
            },
            |m| {
                let chol = Cholesky::new(m).unwrap();
                chol.unpack()
            },
            criterion::BatchSize::SmallInput,
        )
    });
}
fn cholesky_decompose_unpack_500x500(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_decompose_unpack_500x500", |bh| {
        bh.iter_batched(
            || {
                let m = crate::reproducible_dmatrix(500, 500);
                &m * m.transpose()
            },
            |m| {
                let chol = Cholesky::new(m).unwrap();
                chol.unpack()
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_solve_10x10(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_solve_10x10", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(10, 10);
                let m = &m * m.transpose();
                let v = DVector::<f64>::new_random(10);
                let chol = Cholesky::new(m).unwrap();
                (chol, v)
            },
            |(chol, v)| chol.solve(v),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_solve_100x100(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_solve_100x100", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(100, 100);
                let m = &m * m.transpose();
                let v = DVector::<f64>::new_random(100);
                let chol = Cholesky::new(m).unwrap();
                (chol, v)
            },
            |(chol, v)| chol.solve(v),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_solve_500x500(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_solve_500x500", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(500, 500);
                let m = &m * m.transpose();
                let v = DVector::<f64>::new_random(500);
                let chol = Cholesky::new(m).unwrap();
                (chol, v)
            },
            |(chol, v)| chol.solve(v),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_inverse_10x10(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_inverse_10x10", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(10, 10);
                let m = &m * m.transpose();
                Cholesky::new(m).unwrap()
            },
            |chol| chol.inverse(),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_inverse_100x100(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_inverse_100x100", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(100, 100);
                let m = &m * m.transpose();
                Cholesky::new(m).unwrap()
            },
            |chol| chol.inverse(),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn cholesky_inverse_500x500(bh: &mut criterion::Criterion) {
    bh.bench_function("cholesky_inverse_500x500", |bh| {
        bh.iter_batched_ref(
            || {
                let m = crate::reproducible_dmatrix(500, 500);
                let m = &m * m.transpose();
                Cholesky::new(m).unwrap()
            },
            |chol| chol.inverse(),
            criterion::BatchSize::SmallInput,
        )
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
