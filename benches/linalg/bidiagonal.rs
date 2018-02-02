use test::{self, Bencher};
use na::{Bidiagonal, DMatrix, Matrix4};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
#[bench]
fn bidiagonalize_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
}

#[bench]
fn bidiagonalize_100x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
}

#[bench]
fn bidiagonalize_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
}

#[bench]
fn bidiagonalize_500x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
}

#[bench]
fn bidiagonalize_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| test::black_box(Bidiagonal::new(m.clone())))
}

// With unpack.
#[bench]
fn bidiagonalize_unpack_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| {
        let bidiag = Bidiagonal::new(m.clone());
        let _ = bidiag.unpack();
    })
}

#[bench]
fn bidiagonalize_unpack_100x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 500);
    bh.iter(|| {
        let bidiag = Bidiagonal::new(m.clone());
        let _ = bidiag.unpack();
    })
}

#[bench]
fn bidiagonalize_unpack_500x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 100);
    bh.iter(|| {
        let bidiag = Bidiagonal::new(m.clone());
        let _ = bidiag.unpack();
    })
}

#[bench]
fn bidiagonalize_unpack_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| {
        let bidiag = Bidiagonal::new(m.clone());
        let _ = bidiag.unpack();
    })
}
