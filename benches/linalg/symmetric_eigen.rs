use na::SymmetricEigen;

fn symmetric_eigen_decompose_4x4(bh: &mut criterion::Criterion) {
    bh.bench_function("symmetric_eigen_decompose_4x4", |bh| {
        bh.iter_batched(
            || crate::reproducible_smatrix::<f64, 4, 4>(),
            |m| SymmetricEigen::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn symmetric_eigen_decompose_10x10(bh: &mut criterion::Criterion) {
    bh.bench_function("symmetric_eigen_decompose_10x10", |bh| {
        bh.iter_batched(
            || crate::reproducible_dmatrix(10, 10),
            |m| SymmetricEigen::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn symmetric_eigen_decompose_100x100(bh: &mut criterion::Criterion) {
    bh.bench_function("symmetric_eigen_decompose_100x100", |bh| {
        bh.iter_batched(
            || crate::reproducible_dmatrix(100, 100),
            |m| SymmetricEigen::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn symmetric_eigen_decompose_200x200(bh: &mut criterion::Criterion) {
    bh.bench_function("symmetric_eigen_decompose_200x200", |bh| {
        bh.iter_batched(
            || crate::reproducible_dmatrix(200, 200),
            |m| SymmetricEigen::new(m),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    symmetric_eigen,
    symmetric_eigen_decompose_4x4,
    symmetric_eigen_decompose_10x10,
    symmetric_eigen_decompose_100x100,
    symmetric_eigen_decompose_200x200
);
