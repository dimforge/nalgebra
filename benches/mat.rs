#![feature(macro_rules)]

extern crate test;
extern crate "nalgebra" as na;

use std::rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Vec2, Vec3, Vec4, Mat2, Mat3, Mat4};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_mat2_mul_m, Mat2<f32>, Mat2<f32>, mul)
bench_binop!(_bench_mat3_mul_m, Mat3<f32>, Mat3<f32>, mul)
bench_binop!(_bench_mat4_mul_m, Mat4<f32>, Mat4<f32>, mul)

bench_binop!(_bench_mat2_add_m, Mat2<f32>, Mat2<f32>, add)
bench_binop!(_bench_mat3_add_m, Mat3<f32>, Mat3<f32>, add)
bench_binop!(_bench_mat4_add_m, Mat4<f32>, Mat4<f32>, add)

bench_binop!(_bench_mat2_sub_m, Mat2<f32>, Mat2<f32>, sub)
bench_binop!(_bench_mat3_sub_m, Mat3<f32>, Mat3<f32>, sub)
bench_binop!(_bench_mat4_sub_m, Mat4<f32>, Mat4<f32>, sub)

bench_binop!(_bench_mat2_mul_v, Mat2<f32>, Vec2<f32>, mul)
bench_binop!(_bench_mat3_mul_v, Mat3<f32>, Vec3<f32>, mul)
bench_binop!(_bench_mat4_mul_v, Mat4<f32>, Vec4<f32>, mul)

bench_binop!(_bench_mat2_mul_s, Mat2<f32>, f32, mul)
bench_binop!(_bench_mat3_mul_s, Mat3<f32>, f32, mul)
bench_binop!(_bench_mat4_mul_s, Mat4<f32>, f32, mul)

bench_binop!(_bench_mat2_div_s, Mat2<f32>, f32, div)
bench_binop!(_bench_mat3_div_s, Mat3<f32>, f32, div)
bench_binop!(_bench_mat4_div_s, Mat4<f32>, f32, div)

bench_unop!(_bench_mat2_inv, Mat2<f32>, inv)
bench_unop!(_bench_mat3_inv, Mat3<f32>, inv)
bench_unop!(_bench_mat4_inv, Mat4<f32>, inv)

bench_unop!(_bench_mat2_transpose, Mat2<f32>, transpose)
bench_unop!(_bench_mat3_transpose, Mat3<f32>, transpose)
bench_unop!(_bench_mat4_transpose, Mat4<f32>, transpose)
