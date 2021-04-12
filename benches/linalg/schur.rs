use na::{Matrix4, Schur};

fn schur_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("schur_decompose_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(Schur::new(m.clone())))
    });
}

fn schur_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("schur_decompose_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(Schur::new(m.clone())))
    });
}

fn schur_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("schur_decompose_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(Schur::new(m.clone())))
    });
}

fn schur_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("schur_decompose_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(Schur::new(m.clone())))
    });
}

fn eigenvalues_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("eigenvalues_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(m.complex_eigenvalues()))
    });
}

fn eigenvalues_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("eigenvalues_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(m.complex_eigenvalues()))
    });
}

fn eigenvalues_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("eigenvalues_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(m.complex_eigenvalues()))
    });
}

fn eigenvalues_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("eigenvalues_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(m.complex_eigenvalues()))
    });
}

criterion_group!(
    schur,
    schur_decompose_4x4,
    schur_decompose_10x10,
    schur_decompose_100x100,
    schur_decompose_200x200,
    eigenvalues_4x4,
    eigenvalues_10x10,
    eigenvalues_100x100,
    eigenvalues_200x200
);
