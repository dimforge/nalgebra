#![feature(macro_rules)]

extern crate test;
extern crate "nalgebra" as na;

use std::rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Vec2, Vec3, Vec4};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_vec2_add_v, Vec2<f32>, Vec2<f32>, add)
bench_binop!(_bench_vec3_add_v, Vec3<f32>, Vec3<f32>, add)
bench_binop!(_bench_vec4_add_v, Vec4<f32>, Vec4<f32>, add)

bench_binop!(_bench_vec2_sub_v, Vec2<f32>, Vec2<f32>, sub)
bench_binop!(_bench_vec3_sub_v, Vec3<f32>, Vec3<f32>, sub)
bench_binop!(_bench_vec4_sub_v, Vec4<f32>, Vec4<f32>, sub)

bench_binop!(_bench_vec2_mul_v, Vec2<f32>, Vec2<f32>, mul)
bench_binop!(_bench_vec3_mul_v, Vec3<f32>, Vec3<f32>, mul)
bench_binop!(_bench_vec4_mul_v, Vec4<f32>, Vec4<f32>, mul)

bench_binop!(_bench_vec2_div_v, Vec2<f32>, Vec2<f32>, div)
bench_binop!(_bench_vec3_div_v, Vec3<f32>, Vec3<f32>, div)
bench_binop!(_bench_vec4_div_v, Vec4<f32>, Vec4<f32>, div)

bench_binop!(_bench_vec2_add_s, Vec2<f32>, f32, add)
bench_binop!(_bench_vec3_add_s, Vec3<f32>, f32, add)
bench_binop!(_bench_vec4_add_s, Vec4<f32>, f32, add)

bench_binop!(_bench_vec2_sub_s, Vec2<f32>, f32, sub)
bench_binop!(_bench_vec3_sub_s, Vec3<f32>, f32, sub)
bench_binop!(_bench_vec4_sub_s, Vec4<f32>, f32, sub)

bench_binop!(_bench_vec2_mul_s, Vec2<f32>, f32, mul)
bench_binop!(_bench_vec3_mul_s, Vec3<f32>, f32, mul)
bench_binop!(_bench_vec4_mul_s, Vec4<f32>, f32, mul)

bench_binop!(_bench_vec2_div_s, Vec2<f32>, f32, div)
bench_binop!(_bench_vec3_div_s, Vec3<f32>, f32, div)
bench_binop!(_bench_vec4_div_s, Vec4<f32>, f32, div)

bench_binop_na!(_bench_vec2_dot, Vec2<f32>, Vec2<f32>, dot)
bench_binop_na!(_bench_vec3_dot, Vec3<f32>, Vec3<f32>, dot)
bench_binop_na!(_bench_vec4_dot, Vec4<f32>, Vec4<f32>, dot)

bench_binop_na!(_bench_vec3_cross, Vec3<f32>, Vec3<f32>, cross)

bench_unop!(_bench_vec2_norm, Vec2<f32>, norm)
bench_unop!(_bench_vec3_norm, Vec3<f32>, norm)
bench_unop!(_bench_vec4_norm, Vec4<f32>, norm)

bench_unop!(_bench_vec2_normalize, Vec2<f32>, normalize)
bench_unop!(_bench_vec3_normalize, Vec3<f32>, normalize)
bench_unop!(_bench_vec4_normalize, Vec4<f32>, normalize)
