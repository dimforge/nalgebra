use na::{DMatrix, Hessenberg, Matrix4};
use std::hint::black_box;

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
fn hessenberg_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("hessenberg_decompose_4x4", move |bh| {
        bh.iter(|| Hessenberg::new(black_box(m.clone())))
    });
}

fn hessenberg_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("hessenberg_decompose_100x100", move |bh| {
        bh.iter(|| Hessenberg::new(black_box(m.clone())))
    });
}

fn hessenberg_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.bench_function("hessenberg_decompose_200x200", move |bh| {
        bh.iter(|| Hessenberg::new(black_box(m.clone())))
    });
}

// With unpack.
fn hessenberg_decompose_unpack_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("hessenberg_decompose_unpack_100x100", move |bh| {
        bh.iter(|| {
            let hess = Hessenberg::new(black_box(m.clone()));
            hess.unpack()
        })
    });
}

fn hessenberg_decompose_unpack_200x200(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.bench_function("hessenberg_decompose_unpack_200x200", move |bh| {
        bh.iter(|| {
            let hess = Hessenberg::new(black_box(m.clone()));
            hess.unpack()
        })
    });
}

criterion_group!(
    hessenberg,
    hessenberg_decompose_4x4,
    hessenberg_decompose_100x100,
    hessenberg_decompose_200x200,
    hessenberg_decompose_unpack_100x100,
    hessenberg_decompose_unpack_200x200,
);
