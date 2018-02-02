use rand::{IsaacRng, Rng};
use test::{self, Bencher};
use na::{Quaternion, UnitQuaternion, Vector3};
use std::ops::{Add, Div, Mul, Sub};

#[path = "../common/macros.rs"]
mod macros;

bench_binop!(quaternion_add_q, Quaternion<f32>, Quaternion<f32>, add);
bench_binop!(quaternion_sub_q, Quaternion<f32>, Quaternion<f32>, sub);
bench_binop!(quaternion_mul_q, Quaternion<f32>, Quaternion<f32>, mul);

bench_binop!(
    unit_quaternion_mul_v,
    UnitQuaternion<f32>,
    Vector3<f32>,
    mul
);

bench_binop!(quaternion_mul_s, Quaternion<f32>, f32, mul);
bench_binop!(quaternion_div_s, Quaternion<f32>, f32, div);

bench_unop!(quaternion_inv, Quaternion<f32>, try_inverse);
bench_unop!(unit_quaternion_inv, UnitQuaternion<f32>, inverse);

// bench_unop_self!(quaternion_conjugate, Quaternion<f32>, conjugate);
// bench_unop!(quaternion_normalize, Quaternion<f32>, normalize);
