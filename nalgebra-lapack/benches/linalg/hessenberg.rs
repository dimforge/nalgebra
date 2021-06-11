use na::{DMatrix, Matrix4};
use nl::Hessenberg;
use test::{self, Bencher};

#[bench]
fn hessenberg_decompose_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| std::hint::black_box(Hessenberg::new(m.clone())))
}

#[bench]
fn hessenberg_decompose_4x4(bh: &mut Bencher) {
    let m = Matrix4::<f64>::new_random();
    bh.iter(|| std::hint::black_box(Hessenberg::new(m.clone())))
}

#[bench]
fn hessenberg_decompose_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| std::hint::black_box(Hessenberg::new(m.clone())))
}
