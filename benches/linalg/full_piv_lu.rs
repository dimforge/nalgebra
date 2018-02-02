use test::{self, Bencher};
use na::{DMatrix, DVector, FullPivLU};

// Without unpack.
#[bench]
fn full_piv_lu_decompose_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    bh.iter(|| test::black_box(FullPivLU::new(m.clone())))
}

#[bench]
fn full_piv_lu_decompose_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    bh.iter(|| test::black_box(FullPivLU::new(m.clone())))
}

#[bench]
fn full_piv_lu_decompose_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    bh.iter(|| test::black_box(FullPivLU::new(m.clone())))
}

#[bench]
fn full_piv_lu_solve_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(10, 1.0);
        lu.solve(&mut b);
    })
}

#[bench]
fn full_piv_lu_solve_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(100, 1.0);
        lu.solve(&mut b);
    })
}

#[bench]
fn full_piv_lu_solve_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| {
        let mut b = DVector::<f64>::from_element(500, 1.0);
        lu.solve(&mut b);
    })
}

#[bench]
fn full_piv_lu_inverse_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.try_inverse()))
}

#[bench]
fn full_piv_lu_inverse_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.try_inverse()))
}

#[bench]
fn full_piv_lu_inverse_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.try_inverse()))
}

#[bench]
fn full_piv_lu_determinant_10x10(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(10, 10);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.determinant()))
}

#[bench]
fn full_piv_lu_determinant_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.determinant()))
}

#[bench]
fn full_piv_lu_determinant_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);
    let lu = FullPivLU::new(m.clone());

    bh.iter(|| test::black_box(lu.determinant()))
}
