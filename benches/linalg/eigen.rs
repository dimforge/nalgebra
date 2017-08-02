use test::Bencher;
use na::{DMatrix, Eigen};

#[bench]
fn eigen_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);

    bh.iter(|| Eigen::new(m.clone(), 1.0e-7, 0))
}

#[bench]
fn eigen_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);

    bh.iter(|| Eigen::new(m.clone(), 1.0e-7, 0))
}

#[bench]
fn eigenvalues_100x100(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(100, 100);

    bh.iter(|| m.clone().eigenvalues(1.0e-7, 0))
}

#[bench]
fn eigenvalues_500x500(bh: &mut Bencher) {
    let m = DMatrix::<f64>::new_random(500, 500);

    bh.iter(|| m.clone().eigenvalues(1.0e-7, 0))
}
