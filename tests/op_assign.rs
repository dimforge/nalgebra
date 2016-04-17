extern crate nalgebra as na;
extern crate rand;

use std::ops::{Mul, Div, Add, Sub, MulAssign, DivAssign, AddAssign, SubAssign};
use rand::random;
use na::{Pnt3, Vec3, Mat3, Rot3, Iso3, Sim3, Quat, UnitQuat};

// NOTE: we test only the 3D version because the others share the same code anyway.

macro_rules! test_op_vs_op_assign(
    ($name: ident, $t1: ty, $t2: ty, $op: ident, $op_assign: ident) => (
        #[test]
        fn $name() {
            for _ in 0usize .. 10000 {
                let rand1 = random::<$t1>();
                let rand2 = random::<$t2>();
                let mut res = rand1;

                res.$op_assign(rand2);

                assert_eq!(rand1.$op(rand2), res)
            }
        }
    )
);

// Multiplication.
test_op_vs_op_assign!(test_vec3_f32_mul_assign, Vec3<f32>, f32, mul, mul_assign);
test_op_vs_op_assign!(test_mat3_f32_mul_assign, Mat3<f32>, f32, mul, mul_assign);
test_op_vs_op_assign!(test_quat_f32_mul_assign, Quat<f32>, f32, mul, mul_assign);

test_op_vs_op_assign!(test_vec3_vec3_mul_assign, Vec3<f32>, Vec3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_quat_quat_mul_assign, Quat<f32>, Quat<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_unit_quat_unit_quat_mul_assign, UnitQuat<f32>, UnitQuat<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_vec3_unit_quat_mul_assign, Vec3<f32>, UnitQuat<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_pnt3_unit_quat_mul_assign, Pnt3<f32>, UnitQuat<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_mat3_mat3_mul_assign, Mat3<f32>, Mat3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_vec3_mat3_mul_assign, Vec3<f32>, Mat3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_pnt3_mat3_mul_assign, Pnt3<f32>, Mat3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_rot3_rot3_mul_assign, Rot3<f32>, Rot3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_vec3_rot3_mul_assign, Vec3<f32>, Rot3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_pnt3_rot3_mul_assign, Pnt3<f32>, Rot3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_iso3_iso3_mul_assign, Iso3<f32>, Iso3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_iso3_rot3_mul_assign, Iso3<f32>, Rot3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_sim3_sim3_mul_assign, Sim3<f32>, Sim3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_sim3_iso3_mul_assign, Sim3<f32>, Iso3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_sim3_rot3_mul_assign, Sim3<f32>, Rot3<f32>, mul, mul_assign);

// Division.
test_op_vs_op_assign!(test_vec3_vec3_div_assign, Vec3<f32>, Vec3<f32>, div, div_assign);
test_op_vs_op_assign!(test_quat_quat_div_assign, Quat<f32>, Quat<f32>, div, div_assign);
test_op_vs_op_assign!(test_unit_quat_unit_quat_div_assign, UnitQuat<f32>, UnitQuat<f32>, div, div_assign);

test_op_vs_op_assign!(test_vec3_f32_div_assign, Vec3<f32>, f32, div, div_assign);
test_op_vs_op_assign!(test_mat3_f32_div_assign, Mat3<f32>, f32, div, div_assign);

// Addition.
test_op_vs_op_assign!(test_vec3_vec3_add_assign, Vec3<f32>, Vec3<f32>, add, add_assign);
test_op_vs_op_assign!(test_mat3_mat3_add_assign, Mat3<f32>, Mat3<f32>, add, add_assign);
test_op_vs_op_assign!(test_quat_quat_add_assign, Quat<f32>, Quat<f32>, add, add_assign);

test_op_vs_op_assign!(test_vec3_f32_add_assign, Vec3<f32>, f32, add, add_assign);
test_op_vs_op_assign!(test_mat3_f32_add_assign, Mat3<f32>, f32, add, add_assign);

// Subtraction.
test_op_vs_op_assign!(test_vec3_vec3_sub_assign, Vec3<f32>, Vec3<f32>, sub, sub_assign);
test_op_vs_op_assign!(test_mat3_mat3_sub_assign, Mat3<f32>, Mat3<f32>, sub, sub_assign);
test_op_vs_op_assign!(test_quat_quat_sub_assign, Quat<f32>, Quat<f32>, sub, sub_assign);

test_op_vs_op_assign!(test_vec3_f32_sub_assign, Vec3<f32>, f32, sub, sub_assign);
test_op_vs_op_assign!(test_mat3_f32_sub_assign, Mat3<f32>, f32, sub, sub_assign);
