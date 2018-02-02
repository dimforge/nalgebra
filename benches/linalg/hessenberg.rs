use test::{self, Bencher};
use na::{DMatrix, Hessenberg, Matrix4};

#[path = "../common/macros.rs"]
mod macros;

// Without unpack.
#[bench]
fn hessenberg_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
}

#[bench]
fn hessenberg_decompose_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
}

#[bench]
fn hessenberg_decompose_200x200(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
}

#[bench]
fn hessenberg_decompose_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| test::black_box(Hessenberg::new(m.clone())))
}

// With unpack.
#[bench]
fn hessenberg_decompose_unpack_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| {
        let hess = Hessenberg::new(m.clone());
        let _ = hess.unpack();
    })
}

#[bench]
fn hessenberg_decompose_unpack_200x200(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(200, 200);
    bh.iter(|| {
        let hess = Hessenberg::new(m.clone());
        let _ = hess.unpack();
    })
}

#[bench]
fn hessenberg_decompose_unpack_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| {
        let hess = Hessenberg::new(m.clone());
        let _ = hess.unpack();
    })
}
