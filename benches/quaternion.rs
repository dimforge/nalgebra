#![feature(test)]

extern crate test;
extern crate rand;
extern crate nalgebra as na;

use rand::{IsaacRng, Rng};
use test::Bencher;
use na::{Quaternion, UnitQuaternion, Vector3};
use std::ops::{Add, Sub, Mul, Div};

#[path="common/macros.rs"]
mod macros;

bench_binop!(_bench_quaternion_add_q, Quaternion<f32>, Quaternion<f32>, add);
bench_binop!(_bench_quaternion_sub_q, Quaternion<f32>, Quaternion<f32>, sub);
bench_binop!(_bench_quaternion_mul_q, Quaternion<f32>, Quaternion<f32>, mul);

bench_binop!(_bench_unit_quaternion_mul_v, UnitQuaternion<f32>, Vector3<f32>, mul);

bench_binop!(_bench_quaternion_mul_s, Quaternion<f32>, f32, mul);
bench_binop!(_bench_quaternion_div_s, Quaternion<f32>, f32, div);

bench_unop!(_bench_quaternion_inv, Quaternion<f32>, try_inverse);
bench_unop!(_bench_unit_quaternion_inv, UnitQuaternion<f32>, inverse);

// bench_unop_self!(_bench_quaternion_conjugate, Quaternion<f32>, conjugate);
// bench_unop!(_bench_quaternion_normalize, Quaternion<f32>, normalize);
