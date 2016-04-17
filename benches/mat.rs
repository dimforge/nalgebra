#![feature(test)]

extern crate test;
extern crate rand;
extern crate nalgebra as na;

use rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Vector2, Vector3, Vector4, Matrix2, Matrix3, Matrix4};
use std::ops::{Add, Sub, Mul, Div};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_mat2_mul_m, Matrix2<f32>, Matrix2<f32>, mul);
bench_binop!(_bench_mat3_mul_m, Matrix3<f32>, Matrix3<f32>, mul);
bench_binop!(_bench_mat4_mul_m, Matrix4<f32>, Matrix4<f32>, mul);

bench_binop!(_bench_mat2_add_m, Matrix2<f32>, Matrix2<f32>, add);
bench_binop!(_bench_mat3_add_m, Matrix3<f32>, Matrix3<f32>, add);
bench_binop!(_bench_mat4_add_m, Matrix4<f32>, Matrix4<f32>, add);

bench_binop!(_bench_mat2_sub_m, Matrix2<f32>, Matrix2<f32>, sub);
bench_binop!(_bench_mat3_sub_m, Matrix3<f32>, Matrix3<f32>, sub);
bench_binop!(_bench_mat4_sub_m, Matrix4<f32>, Matrix4<f32>, sub);

bench_binop!(_bench_mat2_mul_v, Matrix2<f32>, Vector2<f32>, mul);
bench_binop!(_bench_mat3_mul_v, Matrix3<f32>, Vector3<f32>, mul);
bench_binop!(_bench_mat4_mul_v, Matrix4<f32>, Vector4<f32>, mul);

bench_binop!(_bench_mat2_mul_s, Matrix2<f32>, f32, mul);
bench_binop!(_bench_mat3_mul_s, Matrix3<f32>, f32, mul);
bench_binop!(_bench_mat4_mul_s, Matrix4<f32>, f32, mul);

bench_binop!(_bench_mat2_div_s, Matrix2<f32>, f32, div);
bench_binop!(_bench_mat3_div_s, Matrix3<f32>, f32, div);
bench_binop!(_bench_mat4_div_s, Matrix4<f32>, f32, div);

bench_unop!(_bench_mat2_inv, Matrix2<f32>, inverse);
bench_unop!(_bench_mat3_inv, Matrix3<f32>, inverse);
bench_unop!(_bench_mat4_inv, Matrix4<f32>, inverse);

bench_unop!(_bench_mat2_transpose, Matrix2<f32>, transpose);
bench_unop!(_bench_mat3_transpose, Matrix3<f32>, transpose);
bench_unop!(_bench_mat4_transpose, Matrix4<f32>, transpose);
