use na::{DMatrix, Hessenberg, Matrix4};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
fn hessenberg_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("hessenberg_decompose_4x4", move |bh| {
        bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
    });
}

fn hessenberg_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("hessenberg_decompose_100x100", move |bh| {
        bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
    });
}

fn hessenberg_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.bench_function("hessenberg_decompose_200x200", move |bh| {
        bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
    });
}

fn hessenberg_decompose_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("hessenberg_decompose_500x500", move |bh| {
        bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
    });
}

// With unpack.
fn hessenberg_decompose_unpack_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("hessenberg_decompose_unpack_100x100", move |bh| {
        bh.iter(|| {
            let hess = Hessenberg::new(m.clone());
            let _ = hess.unpack();
        })
    });
}

fn hessenberg_decompose_unpack_200x200(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.bench_function("hessenberg_decompose_unpack_200x200", move |bh| {
        bh.iter(|| {
            let hess = Hessenberg::new(m.clone());
            let _ = hess.unpack();
        })
    });
}

fn hessenberg_decompose_unpack_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("hessenberg_decompose_unpack_500x500", move |bh| {
        bh.iter(|| {
            let hess = Hessenberg::new(m.clone());
            let _ = hess.unpack();
        })
    });
}

criterion_group!(
    hessenberg,
    hessenberg_decompose_4x4,
    hessenberg_decompose_100x100,
    hessenberg_decompose_200x200,
    //    hessenberg_decompose_500x500,
    hessenberg_decompose_unpack_100x100,
    hessenberg_decompose_unpack_200x200,
    //    hessenberg_decompose_unpack_500x500
);
