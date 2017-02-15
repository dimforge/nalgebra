#![feature(test)]

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

bench_binop!(_bench_vec2_mul_s, Vector2<f32>, f32, mul);
bench_binop!(_bench_vec3_mul_s, Vector3<f32>, f32, mul);
bench_binop!(_bench_vec4_mul_s, Vector4<f32>, f32, mul);

bench_binop!(_bench_vec2_div_s, Vector2<f32>, f32, div);
bench_binop!(_bench_vec3_div_s, Vector3<f32>, f32, div);
bench_binop!(_bench_vec4_div_s, Vector4<f32>, f32, div);

bench_binop_ref!(_bench_vec2_dot, Vector2<f32>, Vector2<f32>, dot);
bench_binop_ref!(_bench_vec3_dot, Vector3<f32>, Vector3<f32>, dot);
bench_binop_ref!(_bench_vec4_dot, Vector4<f32>, Vector4<f32>, dot);

bench_binop_ref!(_bench_vec3_cross, Vector3<f32>, Vector3<f32>, cross);

bench_unop!(_bench_vec2_norm, Vector2<f32>, norm);
bench_unop!(_bench_vec3_norm, Vector3<f32>, norm);
bench_unop!(_bench_vec4_norm, Vector4<f32>, norm);

bench_unop!(_bench_vec2_normalize, Vector2<f32>, normalize);
bench_unop!(_bench_vec3_normalize, Vector3<f32>, normalize);
bench_unop!(_bench_vec4_normalize, Vector4<f32>, normalize);
