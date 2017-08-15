use test::Bencher;
use na::{DMatrix, DVector};

#[bench]
fn solve_l_triangular_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.iter(|| {
        let _ = m.solve_lower_triangular(&v);
    })
}

#[bench]
fn solve_l_triangular_1000x1000(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.iter(|| {
        let _ = m.solve_lower_triangular(&v);
    })
}

#[bench]
fn tr_solve_l_triangular_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.iter(|| {
        let _ = m.tr_solve_lower_triangular(&v);
    })
}

#[bench]
fn tr_solve_l_triangular_1000x1000(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.iter(|| {
        let _ = m.tr_solve_lower_triangular(&v);
    })
}

#[bench]
fn solve_u_triangular_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.iter(|| {
        let _ = m.solve_upper_triangular(&v);
    })
}

#[bench]
fn solve_u_triangular_1000x1000(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.iter(|| {
        let _ = m.solve_upper_triangular(&v);
    })
}

#[bench]
fn tr_solve_u_triangular_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);
    let v = DVector::<f64>::new_random(100);

    bh.iter(|| {
        let _ = m.tr_solve_upper_triangular(&v);
    })
}

#[bench]
fn tr_solve_u_triangular_1000x1000(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(1000, 1000);
    let v = DVector::<f64>::new_random(1000);

    bh.iter(|| {
        let _ = m.tr_solve_upper_triangular(&v);
    })
}
