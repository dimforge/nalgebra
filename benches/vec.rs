#![feature(test)]

#[cfg(feature = "generic_sizes")]
extern crate typenum;

extern crate test;
extern crate rand;
extern crate nalgebra as na;

use rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Vector2, Vector3, Vector4};
use std::ops::{Add, Sub, Mul, Div};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_vec2_add_v, Vector2<f32>, Vector2<f32>, add);
bench_binop!(_bench_vec3_add_v, Vector3<f32>, Vector3<f32>, add);
bench_binop!(_bench_vec4_add_v, Vector4<f32>, Vector4<f32>, add);

bench_binop!(_bench_vec2_sub_v, Vector2<f32>, Vector2<f32>, sub);
bench_binop!(_bench_vec3_sub_v, Vector3<f32>, Vector3<f32>, sub);
bench_binop!(_bench_vec4_sub_v, Vector4<f32>, Vector4<f32>, sub);

bench_binop!(_bench_vec2_mul_v, Vector2<f32>, Vector2<f32>, mul);
bench_binop!(_bench_vec3_mul_v, Vector3<f32>, Vector3<f32>, mul);
bench_binop!(_bench_vec4_mul_v, Vector4<f32>, Vector4<f32>, mul);

bench_binop!(_bench_vec2_div_v, Vector2<f32>, Vector2<f32>, div);
bench_binop!(_bench_vec3_div_v, Vector3<f32>, Vector3<f32>, div);
bench_binop!(_bench_vec4_div_v, Vector4<f32>, Vector4<f32>, div);

bench_binop!(_bench_vec2_add_s, Vector2<f32>, f32, add);
bench_binop!(_bench_vec3_add_s, Vector3<f32>, f32, add);
bench_binop!(_bench_vec4_add_s, Vector4<f32>, f32, add);

bench_binop!(_bench_vec2_sub_s, Vector2<f32>, f32, sub);
bench_binop!(_bench_vec3_sub_s, Vector3<f32>, f32, sub);
bench_binop!(_bench_vec4_sub_s, Vector4<f32>, f32, sub);

bench_binop!(_bench_vec2_mul_s, Vector2<f32>, f32, mul);
bench_binop!(_bench_vec3_mul_s, Vector3<f32>, f32, mul);
bench_binop!(_bench_vec4_mul_s, Vector4<f32>, f32, mul);

bench_binop!(_bench_vec2_div_s, Vector2<f32>, f32, div);
bench_binop!(_bench_vec3_div_s, Vector3<f32>, f32, div);
bench_binop!(_bench_vec4_div_s, Vector4<f32>, f32, div);

bench_binop_na!(_bench_vec2_dot, Vector2<f32>, Vector2<f32>, dot);
bench_binop_na!(_bench_vec3_dot, Vector3<f32>, Vector3<f32>, dot);
bench_binop_na!(_bench_vec4_dot, Vector4<f32>, Vector4<f32>, dot);

bench_binop_na!(_bench_vec3_cross, Vector3<f32>, Vector3<f32>, cross);

bench_unop!(_bench_vec2_norm, Vector2<f32>, norm);
bench_unop!(_bench_vec3_norm, Vector3<f32>, norm);
bench_unop!(_bench_vec4_norm, Vector4<f32>, norm);

bench_unop!(_bench_vec2_normalize, Vector2<f32>, normalize);
bench_unop!(_bench_vec3_normalize, Vector3<f32>, normalize);
bench_unop!(_bench_vec4_normalize, Vector4<f32>, normalize);

#[cfg(feature = "generic_sizes")]
mod bench_vecn {
    use typenum::{U2, U3, U4};
    use na::VectorN;

    bench_binop!(_bench_vecn2_add_v, VectorN<f32, U2>, VectorN<f32, U2>, add);
    bench_binop!(_bench_vecn3_add_v, VectorN<f32, U3>, VectorN<f32, U3>, add);
    bench_binop!(_bench_vecn4_add_v, VectorN<f32, U4>, VectorN<f32, U4>, add);

    bench_binop!(_bench_vecn2_sub_v, VectorN<f32, U2>, VectorN<f32, U2>, sub);
    bench_binop!(_bench_vecn3_sub_v, VectorN<f32, U3>, VectorN<f32, U3>, sub);
    bench_binop!(_bench_vecn4_sub_v, VectorN<f32, U4>, VectorN<f32, U4>, sub);

    bench_binop!(_bench_vecn2_mul_v, VectorN<f32, U2>, VectorN<f32, U2>, mul);
    bench_binop!(_bench_vecn3_mul_v, VectorN<f32, U3>, VectorN<f32, U3>, mul);
    bench_binop!(_bench_vecn4_mul_v, VectorN<f32, U4>, VectorN<f32, U4>, mul);

    bench_binop!(_bench_vecn2_div_v, VectorN<f32, U2>, VectorN<f32, U2>, div);
    bench_binop!(_bench_vecn3_div_v, VectorN<f32, U3>, VectorN<f32, U3>, div);
    bench_binop!(_bench_vecn4_div_v, VectorN<f32, U4>, VectorN<f32, U4>, div);

    bench_binop!(_bench_vecn2_add_s, VectorN<f32, U2>, f32, add);
    bench_binop!(_bench_vecn3_add_s, VectorN<f32, U3>, f32, add);
    bench_binop!(_bench_vecn4_add_s, VectorN<f32, U4>, f32, add);

    bench_binop!(_bench_vecn2_sub_s, VectorN<f32, U2>, f32, sub);
    bench_binop!(_bench_vecn3_sub_s, VectorN<f32, U3>, f32, sub);
    bench_binop!(_bench_vecn4_sub_s, VectorN<f32, U4>, f32, sub);

    bench_binop!(_bench_vecn2_mul_s, VectorN<f32, U2>, f32, mul);
    bench_binop!(_bench_vecn3_mul_s, VectorN<f32, U3>, f32, mul);
    bench_binop!(_bench_vecn4_mul_s, VectorN<f32, U4>, f32, mul);

    bench_binop!(_bench_vecn2_div_s, VectorN<f32, U2>, f32, div);
    bench_binop!(_bench_vecn3_div_s, VectorN<f32, U3>, f32, div);
    bench_binop!(_bench_vecn4_div_s, VectorN<f32, U4>, f32, div);

    bench_binop_na!(_bench_vecn2_dot, VectorN<f32, U2>, VectorN<f32, U2>, dot);
    bench_binop_na!(_bench_vecn3_dot, VectorN<f32, U3>, VectorN<f32, U3>, dot);
    bench_binop_na!(_bench_vecn4_dot, VectorN<f32, U4>, VectorN<f32, U4>, dot);

    bench_unop!(_bench_vecn2_norm, VectorN<f32, U2>, norm);
    bench_unop!(_bench_vecn3_norm, VectorN<f32, U3>, norm);
    bench_unop!(_bench_vecn4_norm, VectorN<f32, U4>, norm);

    bench_unop!(_bench_vecn2_normalize, VectorN<f32, U2>, normalize);
    bench_unop!(_bench_vecn3_normalize, VectorN<f32, U3>, normalize);
    bench_unop!(_bench_vecn4_normalize, VectorN<f32, U4>, normalize);
}
