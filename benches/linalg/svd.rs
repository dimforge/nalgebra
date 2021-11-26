use na::{Matrix2, Matrix3, Matrix4, SVD};

fn svd_decompose_2x2_f32(bh: &mut criterion::Criterion) {
    let m = Matrix2::<f32>::new_random();
    bh.bench_function("svd_decompose_2x2", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new_unordered(m.clone(), true, true)))
    });
}

fn svd_decompose_3x3_f32(bh: &mut criterion::Criterion) {
    let m = Matrix3::<f32>::new_random();
    bh.bench_function("svd_decompose_3x3", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new_unordered(m.clone(), true, true)))
    });
}

fn svd_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("svd_decompose_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new(m.clone(), true, true)))
    });
}

fn svd_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("svd_decompose_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new(m.clone(), true, true)))
    });
}

fn svd_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("svd_decompose_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new(m.clone(), true, true)))
    });
}

fn svd_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("svd_decompose_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(SVD::new(m.clone(), true, true)))
    });
}

fn rank_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("rank_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(m.rank(1.0e-10)))
    });
}

fn rank_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("rank_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(m.rank(1.0e-10)))
    });
}

fn rank_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("rank_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(m.rank(1.0e-10)))
    });
}

fn rank_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("rank_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(m.rank(1.0e-10)))
    });
}

fn singular_values_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("singular_values_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(m.singular_values()))
    });
}

fn singular_values_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("singular_values_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(m.singular_values()))
    });
}

fn singular_values_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("singular_values_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(m.singular_values()))
    });
}

fn singular_values_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("singular_values_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(m.singular_values()))
    });
}

fn pseudo_inverse_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("pseudo_inverse_4x4", move |bh| {
        bh.iter(|| std::hint::black_box(m.clone().pseudo_inverse(1.0e-10)))
    });
}

fn pseudo_inverse_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("pseudo_inverse_10x10", move |bh| {
        bh.iter(|| std::hint::black_box(m.clone().pseudo_inverse(1.0e-10)))
    });
}

fn pseudo_inverse_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("pseudo_inverse_100x100", move |bh| {
        bh.iter(|| std::hint::black_box(m.clone().pseudo_inverse(1.0e-10)))
    });
}

fn pseudo_inverse_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("pseudo_inverse_200x200", move |bh| {
        bh.iter(|| std::hint::black_box(m.clone().pseudo_inverse(1.0e-10)))
    });
}

criterion_group!(
    svd,
    svd_decompose_2x2_f32,
    svd_decompose_3x3_f32,
    svd_decompose_4x4,
    svd_decompose_10x10,
    svd_decompose_100x100,
    svd_decompose_200x200,
    rank_4x4,
    rank_10x10,
    rank_100x100,
    rank_200x200,
    singular_values_4x4,
    singular_values_10x10,
    singular_values_100x100,
    singular_values_200x200,
    pseudo_inverse_4x4,
    pseudo_inverse_10x10,
    pseudo_inverse_100x100,
    pseudo_inverse_200x200
);
