use na::{Bidiagonal, DMatrix, Matrix4};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
fn bidiagonalize_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("bidiagonalize_100x100", move |bh| {
        bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
    });
}

fn bidiagonalize_100x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.bench_function("bidiagonalize_100x500", move |bh| {
        bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
    });
}

fn bidiagonalize_4x4(bh: &mut criterion::Criterion) {
    let m = Matrix4::<f64>::new_random();
    bh.bench_function("bidiagonalize_4x4", move |bh| {
        bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
    });
}

fn bidiagonalize_500x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.bench_function("bidiagonalize_500x100", move |bh| {
        bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
    });
}

fn bidiagonalize_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("bidiagonalize_500x500", move |bh| {
        bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
    });
}

// With unpack.
fn bidiagonalize_unpack_100x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.bench_function("bidiagonalize_unpack_100x100", move |bh| {
        bh.iter(|| {
            let bidiag = Bidiagonal::new(m.clone());
            let _ = bidiag.unpack();
        })
    });
}

fn bidiagonalize_unpack_100x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.bench_function("bidiagonalize_unpack_100x500", move |bh| {
        bh.iter(|| {
            let bidiag = Bidiagonal::new(m.clone());
            let _ = bidiag.unpack();
        })
    });
}

fn bidiagonalize_unpack_500x100(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.bench_function("bidiagonalize_unpack_500x100", move |bh| {
        bh.iter(|| {
            let bidiag = Bidiagonal::new(m.clone());
            let _ = bidiag.unpack();
        })
    });
}

fn bidiagonalize_unpack_500x500(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.bench_function("bidiagonalize_unpack_500x500", move |bh| {
        bh.iter(|| {
            let bidiag = Bidiagonal::new(m.clone());
            let _ = bidiag.unpack();
        })
    });
}

criterion_group!(
    bidiagonal,
    bidiagonalize_100x100,
    bidiagonalize_100x500,
    bidiagonalize_4x4,
    bidiagonalize_500x100,
    //    bidiagonalize_500x500, // too long
    bidiagonalize_unpack_100x100,
    bidiagonalize_unpack_100x500,
    bidiagonalize_unpack_500x100,
    //    bidiagonalize_unpack_500x500 // too long
);
