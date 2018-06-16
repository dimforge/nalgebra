use rand::{IsaacRng, Rng};
use test::{self, Bencher};
use na::{DMatrix, DVector, Matrix2, Matrix3, Matrix4, MatrixN, U10, Vector2, Vector3, Vector4};
use std::ops::{Add, Div, Mul, Sub};

#[path = "../common/macros.rs"]
mod macros;

bench_binop!(mat2_mul_m, Matrix2<f32>, Matrix2<f32>, mul);
bench_binop!(mat3_mul_m, Matrix3<f32>, Matrix3<f32>, mul);
bench_binop!(mat4_mul_m, Matrix4<f32>, Matrix4<f32>, mul);

bench_binop_ref!(mat2_tr_mul_m, Matrix2<f32>, Matrix2<f32>, tr_mul);
bench_binop_ref!(mat3_tr_mul_m, Matrix3<f32>, Matrix3<f32>, tr_mul);
bench_binop_ref!(mat4_tr_mul_m, Matrix4<f32>, Matrix4<f32>, tr_mul);

bench_binop!(mat2_add_m, Matrix2<f32>, Matrix2<f32>, add);
bench_binop!(mat3_add_m, Matrix3<f32>, Matrix3<f32>, add);
bench_binop!(mat4_add_m, Matrix4<f32>, Matrix4<f32>, add);

bench_binop!(mat2_sub_m, Matrix2<f32>, Matrix2<f32>, sub);
bench_binop!(mat3_sub_m, Matrix3<f32>, Matrix3<f32>, sub);
bench_binop!(mat4_sub_m, Matrix4<f32>, Matrix4<f32>, sub);

bench_binop!(mat2_mul_v, Matrix2<f32>, Vector2<f32>, mul);
bench_binop!(mat3_mul_v, Matrix3<f32>, Vector3<f32>, mul);
bench_binop!(mat4_mul_v, Matrix4<f32>, Vector4<f32>, mul);

bench_binop_ref!(mat2_tr_mul_v, Matrix2<f32>, Vector2<f32>, tr_mul);
bench_binop_ref!(mat3_tr_mul_v, Matrix3<f32>, Vector3<f32>, tr_mul);
bench_binop_ref!(mat4_tr_mul_v, Matrix4<f32>, Vector4<f32>, tr_mul);

bench_binop!(mat2_mul_s, Matrix2<f32>, f32, mul);
bench_binop!(mat3_mul_s, Matrix3<f32>, f32, mul);
bench_binop!(mat4_mul_s, Matrix4<f32>, f32, mul);

bench_binop!(mat2_div_s, Matrix2<f32>, f32, div);
bench_binop!(mat3_div_s, Matrix3<f32>, f32, div);
bench_binop!(mat4_div_s, Matrix4<f32>, f32, div);

bench_unop!(mat2_inv, Matrix2<f32>, try_inverse);
bench_unop!(mat3_inv, Matrix3<f32>, try_inverse);
bench_unop!(mat4_inv, Matrix4<f32>, try_inverse);

bench_unop!(mat2_transpose, Matrix2<f32>, transpose);
bench_unop!(mat3_transpose, Matrix3<f32>, transpose);
bench_unop!(mat4_transpose, Matrix4<f32>, transpose);

#[bench]
fn mat_div_scalar(b: &mut Bencher) {
    let a = DMatrix::from_row_slice(1000, 1000, &vec![2.0; 1000000]);
    let n = 42.0;

    b.iter(|| {
        let mut aa = a.clone();
        let mut b = aa.slice_mut((0, 0), (1000, 1000));
        b /= n
    })
}

#[bench]
fn mat100_add_mat100(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);

    bench.iter(|| &a + &b)
}

#[bench]
fn mat4_mul_mat4(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(4, 4);
    let b = DMatrix::<f64>::new_random(4, 4);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat5_mul_mat5(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(5, 5);
    let b = DMatrix::<f64>::new_random(5, 5);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat6_mul_mat6(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(6, 6);
    let b = DMatrix::<f64>::new_random(6, 6);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat7_mul_mat7(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(7, 7);
    let b = DMatrix::<f64>::new_random(7, 7);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat8_mul_mat8(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(8, 8);
    let b = DMatrix::<f64>::new_random(8, 8);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat9_mul_mat9(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(9, 9);
    let b = DMatrix::<f64>::new_random(9, 9);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat10_mul_mat10(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(10, 10);
    let b = DMatrix::<f64>::new_random(10, 10);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat10_mul_mat10_static(bench: &mut Bencher) {
    let a = MatrixN::<f64, U10>::new_random();
    let b = MatrixN::<f64, U10>::new_random();

    bench.iter(|| &a * &b)
}

#[bench]
fn mat100_mul_mat100(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);

    bench.iter(|| &a * &b)
}

#[bench]
fn mat500_mul_mat500(bench: &mut Bencher) {
    let a = DMatrix::<f64>::from_element(500, 500, 5f64);
    let b = DMatrix::<f64>::from_element(500, 500, 6f64);

    bench.iter(|| &a * &b)
}

#[bench]
fn copy_from(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(1000, 1000);
    let mut b = DMatrix::<f64>::new_random(1000, 1000);

    bench.iter(|| {
        b.copy_from(&a);
    })
}

#[bench]
fn axpy(bench: &mut Bencher) {
    let x = DVector::<f64>::from_element(100000, 2.0);
    let mut y = DVector::<f64>::from_element(100000, 3.0);
    let a = 42.0;

    bench.iter(|| {
        y.axpy(a, &x, 1.0);
    })
}

#[bench]
fn tr_mul_to(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(1000, 1000);
    let b = DVector::<f64>::new_random(1000);
    let mut c = DVector::from_element(1000, 0.0);

    bench.iter(|| a.tr_mul_to(&b, &mut c))
}

#[bench]
fn mat_mul_mat(bench: &mut Bencher) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);
    let mut ab = DMatrix::<f64>::from_element(100, 100, 0.0);

    bench.iter(|| {
        test::black_box(a.mul_to(&b, &mut ab));
    })
}

#[bench]
fn mat100_from_fn(bench: &mut Bencher) {
    bench.iter(|| {
        DMatrix::from_fn(100, 100, |a, b| {
            a + b
        })
    })
}

#[bench]
fn mat500_from_fn(bench: &mut Bencher) {
    bench.iter(|| {
        DMatrix::from_fn(500, 500, |a, b| {
            a + b
        })
    })
}
