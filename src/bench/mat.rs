use rand::random;
use test::Bencher;
use na::{Vec2, Vec3, Vec4, Vec5, Vec6, DVec, Mat2, Mat3, Mat4, Mat5, Mat6, DMat};

macro_rules! bench_mul_mat(
    ($bh: expr, $t: ty) => {
        {
            let a: $t     = random();
            let mut b: $t = random();

            $bh.iter(|| {
                for _ in range(0, 1000) {
                    b = a * b;
                }
            })
        }
    }
)

#[bench]
fn bench_mul_mat2(bh: &mut Bencher) {
    bench_mul_mat!(bh, Mat2<f64>)
}

#[bench]
fn bench_mul_mat3(bh: &mut Bencher) {
    bench_mul_mat!(bh, Mat3<f64>)
}

#[bench]
fn bench_mul_mat4(bh: &mut Bencher) {
    bench_mul_mat!(bh, Mat4<f64>)
}

#[bench]
fn bench_mul_mat5(bh: &mut Bencher) {
    bench_mul_mat!(bh, Mat5<f64>)
}

#[bench]
fn bench_mul_mat6(bh: &mut Bencher) {
    bench_mul_mat!(bh, Mat6<f64>)
}

macro_rules! bench_mul_dmat(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {
            let a:     DMat<f64> = DMat::new_random($nrows, $ncols);
            let mut b: DMat<f64> = DMat::new_random($nrows, $ncols);

            $bh.iter(|| {
                for _ in range(0, 1000) {
                    b = a * b;
                }
            })
        }
    }
)

#[bench]
fn bench_mul_dmat2(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 2, 2)
}

#[bench]
fn bench_mul_dmat3(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 3, 3)
}

#[bench]
fn bench_mul_dmat4(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 4, 4)
}

#[bench]
fn bench_mul_dmat5(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 5, 5)
}

#[bench]
fn bench_mul_dmat6(bh: &mut Bencher) {
    bench_mul_dmat!(bh, 6, 6)
}

macro_rules! bench_mul_mat_vec(
    ($bh: expr, $tm: ty, $tv: ty) => {
        {
            let m : $tm     = random();
            let mut v : $tv = random();

            $bh.iter(|| {
                for _ in range(0, 1000) {
                    v = m * v
                }
            })
        }
    }
)

#[bench]
fn bench_mul_mat_vec2(bh: &mut Bencher) {
    bench_mul_mat_vec!(bh, Mat2<f64>, Vec2<f64>)
}

#[bench]
fn bench_mul_mat_vec3(bh: &mut Bencher) {
    bench_mul_mat_vec!(bh, Mat3<f64>, Vec3<f64>)
}

#[bench]
fn bench_mul_mat_vec4(bh: &mut Bencher) {
    bench_mul_mat_vec!(bh, Mat4<f64>, Vec4<f64>)
}

#[bench]
fn bench_mul_mat_vec5(bh: &mut Bencher) {
    bench_mul_mat_vec!(bh, Mat5<f64>, Vec5<f64>)
}

#[bench]
fn bench_mul_mat_vec6(bh: &mut Bencher) {
    bench_mul_mat_vec!(bh, Mat6<f64>, Vec6<f64>)
}

macro_rules! bench_mul_dmat_dvec(
    ($bh: expr, $nrows: expr, $ncols: expr) => {
        {
            let m : DMat<f64>     = DMat::new_random($nrows, $ncols);
            let mut v : DVec<f64> = DVec::new_random($ncols);

            $bh.iter(|| {
                for _ in range(0, 1000) {
                    v = m * v
                }
            })
        }
    }
)

#[bench]
fn bench_mul_dmat_dvec2(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 2, 2)
}

#[bench]
fn bench_mul_dmat_dvec3(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 3, 3)
}

#[bench]
fn bench_mul_dmat_dvec4(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 4, 4)
}

#[bench]
fn bench_mul_dmat_dvec5(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 5, 5)
}

#[bench]
fn bench_mul_dmat_dvec6(bh: &mut Bencher) {
    bench_mul_dmat_dvec!(bh, 6, 6)
}
