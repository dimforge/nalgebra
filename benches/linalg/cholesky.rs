use test::{self, Bencher};
use na::{Cholesky, DMatrix, DVector};

#[bench]
fn cholesky_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();

    bh.iter(|| test::black_box(Cholesky::new(m.clone())))
}

#[bench]
fn cholesky_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();

    bh.iter(|| test::black_box(Cholesky::new(m.clone())))
}

// With unpack.
#[bench]
fn cholesky_decompose_unpack_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();

    bh.iter(|| {
        let chol = Cholesky::new(m.clone()).unwrap();
        let _ = chol.unpack();
    })
}
#[bench]
fn cholesky_decompose_unpack_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();

    bh.iter(|| {
        let chol = Cholesky::new(m.clone()).unwrap();
        let _ = chol.unpack();
    })
}

#[bench]
fn cholesky_solve_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(10);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.solve(&v);
    })
}

#[bench]
fn cholesky_solve_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(100);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.solve(&v);
    })
}

#[bench]
fn cholesky_solve_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();
    let v = DVector::<f64>::new_random(500);
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.solve(&v);
    })
}

#[bench]
fn cholesky_inverse_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.inverse();
    })
}

#[bench]
fn cholesky_inverse_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.inverse();
    })
}

#[bench]
fn cholesky_inverse_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let m = &m * m.transpose();
    let chol = Cholesky::new(m.clone()).unwrap();

    bh.iter(|| {
        let _ = chol.inverse();
    })
}
