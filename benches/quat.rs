#![feature(macro_rules)]

extern crate test;
extern crate "nalgebra" as na;

use std::rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Quat, UnitQuat, Vec3};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_quat_add_q, Quat<f32>, Quat<f32>, add)
bench_binop!(_bench_quat_sub_q, Quat<f32>, Quat<f32>, sub)
bench_binop!(_bench_quat_mul_q, Quat<f32>, Quat<f32>, mul)
// bench_binop!(_bench_quat_div_q, Quat<f32>, Quat<f32>, div)
bench_binop!(_bench_quat_mul_v, UnitQuat<f32>, Vec3<f32>, mul)
bench_binop!(_bench_quat_mul_s, Quat<f32>, f32, mul)
bench_binop!(_bench_quat_div_s, Quat<f32>, f32, div)
bench_unop!(_bench_quat_inv, Quat<f32>, inv)
bench_unop_self!(_bench_quat_conjugate, Quat<f32>, conjugate)
bench_unop!(_bench_quat_normalize, Quat<f32>, normalize)
