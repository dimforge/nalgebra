#![feature(test)]

#[cfg(feature = "generic_sizes")]
extern crate typenum;

extern crate test;
extern crate rand;
extern crate nalgebra as na;

use rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Vec2, Vec3, Vec4};
use std::ops::{Add, Sub, Mul, Div};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_vec2_add_v, Vec2<f32>, Vec2<f32>, add);
bench_binop!(_bench_vec3_add_v, Vec3<f32>, Vec3<f32>, add);
bench_binop!(_bench_vec4_add_v, Vec4<f32>, Vec4<f32>, add);

bench_binop!(_bench_vec2_sub_v, Vec2<f32>, Vec2<f32>, sub);
bench_binop!(_bench_vec3_sub_v, Vec3<f32>, Vec3<f32>, sub);
bench_binop!(_bench_vec4_sub_v, Vec4<f32>, Vec4<f32>, sub);

bench_binop!(_bench_vec2_mul_v, Vec2<f32>, Vec2<f32>, mul);
bench_binop!(_bench_vec3_mul_v, Vec3<f32>, Vec3<f32>, mul);
bench_binop!(_bench_vec4_mul_v, Vec4<f32>, Vec4<f32>, mul);

bench_binop!(_bench_vec2_div_v, Vec2<f32>, Vec2<f32>, div);
bench_binop!(_bench_vec3_div_v, Vec3<f32>, Vec3<f32>, div);
bench_binop!(_bench_vec4_div_v, Vec4<f32>, Vec4<f32>, div);

bench_binop!(_bench_vec2_add_s, Vec2<f32>, f32, add);
bench_binop!(_bench_vec3_add_s, Vec3<f32>, f32, add);
bench_binop!(_bench_vec4_add_s, Vec4<f32>, f32, add);

bench_binop!(_bench_vec2_sub_s, Vec2<f32>, f32, sub);
bench_binop!(_bench_vec3_sub_s, Vec3<f32>, f32, sub);
bench_binop!(_bench_vec4_sub_s, Vec4<f32>, f32, sub);

bench_binop!(_bench_vec2_mul_s, Vec2<f32>, f32, mul);
bench_binop!(_bench_vec3_mul_s, Vec3<f32>, f32, mul);
bench_binop!(_bench_vec4_mul_s, Vec4<f32>, f32, mul);

bench_binop!(_bench_vec2_div_s, Vec2<f32>, f32, div);
bench_binop!(_bench_vec3_div_s, Vec3<f32>, f32, div);
bench_binop!(_bench_vec4_div_s, Vec4<f32>, f32, div);

bench_binop_na!(_bench_vec2_dot, Vec2<f32>, Vec2<f32>, dot);
bench_binop_na!(_bench_vec3_dot, Vec3<f32>, Vec3<f32>, dot);
bench_binop_na!(_bench_vec4_dot, Vec4<f32>, Vec4<f32>, dot);

bench_binop_na!(_bench_vec3_cross, Vec3<f32>, Vec3<f32>, cross);

bench_unop!(_bench_vec2_norm, Vec2<f32>, norm);
bench_unop!(_bench_vec3_norm, Vec3<f32>, norm);
bench_unop!(_bench_vec4_norm, Vec4<f32>, norm);

bench_unop!(_bench_vec2_normalize, Vec2<f32>, normalize);
bench_unop!(_bench_vec3_normalize, Vec3<f32>, normalize);
bench_unop!(_bench_vec4_normalize, Vec4<f32>, normalize);

#[cfg(feature = "generic_sizes")]
mod bench_vecn {
    use typenum::{U2, U3, U4};
    use na::VecN;

    bench_binop!(_bench_vecn2_add_v, VecN<f32, U2>, VecN<f32, U2>, add);
    bench_binop!(_bench_vecn3_add_v, VecN<f32, U3>, VecN<f32, U3>, add);
    bench_binop!(_bench_vecn4_add_v, VecN<f32, U4>, VecN<f32, U4>, add);

    bench_binop!(_bench_vecn2_sub_v, VecN<f32, U2>, VecN<f32, U2>, sub);
    bench_binop!(_bench_vecn3_sub_v, VecN<f32, U3>, VecN<f32, U3>, sub);
    bench_binop!(_bench_vecn4_sub_v, VecN<f32, U4>, VecN<f32, U4>, sub);

    bench_binop!(_bench_vecn2_mul_v, VecN<f32, U2>, VecN<f32, U2>, mul);
    bench_binop!(_bench_vecn3_mul_v, VecN<f32, U3>, VecN<f32, U3>, mul);
    bench_binop!(_bench_vecn4_mul_v, VecN<f32, U4>, VecN<f32, U4>, mul);

    bench_binop!(_bench_vecn2_div_v, VecN<f32, U2>, VecN<f32, U2>, div);
    bench_binop!(_bench_vecn3_div_v, VecN<f32, U3>, VecN<f32, U3>, div);
    bench_binop!(_bench_vecn4_div_v, VecN<f32, U4>, VecN<f32, U4>, div);

    bench_binop!(_bench_vecn2_add_s, VecN<f32, U2>, f32, add);
    bench_binop!(_bench_vecn3_add_s, VecN<f32, U3>, f32, add);
    bench_binop!(_bench_vecn4_add_s, VecN<f32, U4>, f32, add);

    bench_binop!(_bench_vecn2_sub_s, VecN<f32, U2>, f32, sub);
    bench_binop!(_bench_vecn3_sub_s, VecN<f32, U3>, f32, sub);
    bench_binop!(_bench_vecn4_sub_s, VecN<f32, U4>, f32, sub);

    bench_binop!(_bench_vecn2_mul_s, VecN<f32, U2>, f32, mul);
    bench_binop!(_bench_vecn3_mul_s, VecN<f32, U3>, f32, mul);
    bench_binop!(_bench_vecn4_mul_s, VecN<f32, U4>, f32, mul);

    bench_binop!(_bench_vecn2_div_s, VecN<f32, U2>, f32, div);
    bench_binop!(_bench_vecn3_div_s, VecN<f32, U3>, f32, div);
    bench_binop!(_bench_vecn4_div_s, VecN<f32, U4>, f32, div);

    bench_binop_na!(_bench_vecn2_dot, VecN<f32, U2>, VecN<f32, U2>, dot);
    bench_binop_na!(_bench_vecn3_dot, VecN<f32, U3>, VecN<f32, U3>, dot);
    bench_binop_na!(_bench_vecn4_dot, VecN<f32, U4>, VecN<f32, U4>, dot);

    bench_unop!(_bench_vecn2_norm, VecN<f32, U2>, norm);
    bench_unop!(_bench_vecn3_norm, VecN<f32, U3>, norm);
    bench_unop!(_bench_vecn4_norm, VecN<f32, U4>, norm);

    bench_unop!(_bench_vecn2_normalize, VecN<f32, U2>, normalize);
    bench_unop!(_bench_vecn3_normalize, VecN<f32, U3>, normalize);
    bench_unop!(_bench_vecn4_normalize, VecN<f32, U4>, normalize);
}
