use na::LU;

use crate::reproducible_matrix;

fn inverse_4(bh: &mut criterion::Criterion) {
    let m = reproducible_matrix::<f64, 4, 4>();
    bh.bench_function("4x4_matrix_inverse_det", move |bh| {
        bh.iter(|| std::hint::black_box(m.try_inverse().unwrap()))
    });
}

fn inverse_lu(bh: &mut criterion::Criterion) {
    let m = reproducible_matrix::<f64, 4, 4>();
    bh.bench_function("4x4_matrix_inverse_lu", move |bh| {
        bh.iter(|| std::hint::black_box(LU::new(m).try_inverse().unwrap()))
    });
}

criterion_group!(matrix_inversion, inverse_4, inverse_lu);
