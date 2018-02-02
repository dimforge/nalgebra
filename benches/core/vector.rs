use rand::{IsaacRng, Rng};
use test::{self, Bencher};
use typenum::U10000;
use na::{DVector, Vector2, Vector3, Vector4, VectorN};
use std::ops::{Add, Div, Mul, Sub};

#[path = "../common/macros.rs"]
mod macros;

bench_binop!(vec2_add_v_f32, Vector2<f32>, Vector2<f32>, add);
bench_binop!(vec3_add_v_f32, Vector3<f32>, Vector3<f32>, add);
bench_binop!(vec4_add_v_f32, Vector4<f32>, Vector4<f32>, add);

bench_binop!(vec2_add_v_f64, Vector2<f64>, Vector2<f64>, add);
bench_binop!(vec3_add_v_f64, Vector3<f64>, Vector3<f64>, add);
bench_binop!(vec4_add_v_f64, Vector4<f64>, Vector4<f64>, add);

bench_binop!(vec2_sub_v, Vector2<f32>, Vector2<f32>, sub);
bench_binop!(vec3_sub_v, Vector3<f32>, Vector3<f32>, sub);
bench_binop!(vec4_sub_v, Vector4<f32>, Vector4<f32>, sub);

bench_binop!(vec2_mul_s, Vector2<f32>, f32, mul);
bench_binop!(vec3_mul_s, Vector3<f32>, f32, mul);
bench_binop!(vec4_mul_s, Vector4<f32>, f32, mul);

bench_binop!(vec2_div_s, Vector2<f32>, f32, div);
bench_binop!(vec3_div_s, Vector3<f32>, f32, div);
bench_binop!(vec4_div_s, Vector4<f32>, f32, div);

bench_binop_ref!(vec2_dot_f32, Vector2<f32>, Vector2<f32>, dot);
bench_binop_ref!(vec3_dot_f32, Vector3<f32>, Vector3<f32>, dot);
bench_binop_ref!(vec4_dot_f32, Vector4<f32>, Vector4<f32>, dot);

bench_binop_ref!(vec2_dot_f64, Vector2<f64>, Vector2<f64>, dot);
bench_binop_ref!(vec3_dot_f64, Vector3<f64>, Vector3<f64>, dot);
bench_binop_ref!(vec4_dot_f64, Vector4<f64>, Vector4<f64>, dot);

bench_binop_ref!(vec3_cross, Vector3<f32>, Vector3<f32>, cross);

bench_unop!(vec2_norm, Vector2<f32>, norm);
bench_unop!(vec3_norm, Vector3<f32>, norm);
bench_unop!(vec4_norm, Vector4<f32>, norm);

bench_unop!(vec2_normalize, Vector2<f32>, normalize);
bench_unop!(vec3_normalize, Vector3<f32>, normalize);
bench_unop!(vec4_normalize, Vector4<f32>, normalize);

bench_binop_ref!(vec10000_dot_f64, VectorN<f64, U10000>, VectorN<f64, U10000>, dot);
bench_binop_ref!(vec10000_dot_f32, VectorN<f32, U10000>, VectorN<f32, U10000>, dot);

#[bench]
fn vec10000_axpy_f64(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = DVector::new_random(10000);
    let b = DVector::new_random(10000);
    let n = rng.gen::<f64>();

    bh.iter(|| a.axpy(n, &b, 1.0))
}

#[bench]
fn vec10000_axpy_beta_f64(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = DVector::new_random(10000);
    let b = DVector::new_random(10000);
    let n = rng.gen::<f64>();
    let beta = rng.gen::<f64>();

    bh.iter(|| a.axpy(n, &b, beta))
}

#[bench]
fn vec10000_axpy_f64_slice(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = DVector::new_random(10000);
    let b = DVector::new_random(10000);
    let n = rng.gen::<f64>();

    bh.iter(|| {
        let mut a = a.fixed_rows_mut::<U10000>(0);
        let b = b.fixed_rows::<U10000>(0);

        a.axpy(n, &b, 1.0)
    })
}

#[bench]
fn vec10000_axpy_f64_static(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = VectorN::<f64, U10000>::new_random();
    let b = VectorN::<f64, U10000>::new_random();
    let n = rng.gen::<f64>();

    // NOTE: for some reasons, it is much faster if the arument are boxed (Box::new(VectorN...)).
    bh.iter(|| a.axpy(n, &b, 1.0))
}

#[bench]
fn vec10000_axpy_f32(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = DVector::new_random(10000);
    let b = DVector::new_random(10000);
    let n = rng.gen::<f32>();

    bh.iter(|| a.axpy(n, &b, 1.0))
}

#[bench]
fn vec10000_axpy_beta_f32(bh: &mut Bencher) {
    let mut rng = IsaacRng::new_unseeded();
    let mut a = DVector::new_random(10000);
    let b = DVector::new_random(10000);
    let n = rng.gen::<f32>();
    let beta = rng.gen::<f32>();

    bh.iter(|| a.axpy(n, &b, beta))
}
