use na::{Matrix4, SymmetricEigen};
use std::hint::black_box;

fn symmetric_eigen_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("symmetric_eigen_decompose_4x4", move |bh| {
        bh.iter(|| SymmetricEigen::new(black_box(m.clone())))
    });
}

fn symmetric_eigen_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproducible_dmatrix(10, 10);
    bh.bench_function("symmetric_eigen_decompose_10x10", move |bh| {
        bh.iter(|| SymmetricEigen::new(black_box(m.clone())))
    });
}

fn symmetric_eigen_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproducible_dmatrix(100, 100);
    bh.bench_function("symmetric_eigen_decompose_100x100", move |bh| {
        bh.iter(|| SymmetricEigen::new(black_box(m.clone())))
    });
}

fn symmetric_eigen_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproducible_dmatrix(200, 200);
    bh.bench_function("symmetric_eigen_decompose_200x200", move |bh| {
        bh.iter(|| SymmetricEigen::new(black_box(m.clone())))
    });
}

criterion_group!(
    symmetric_eigen,
    symmetric_eigen_decompose_4x4,
    symmetric_eigen_decompose_10x10,
    symmetric_eigen_decompose_100x100,
    symmetric_eigen_decompose_200x200
);
