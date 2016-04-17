extern crate nalgebra as na;
extern crate rand;

use std::ops::{Mul, Div, Add, Sub, MulAssign, DivAssign, AddAssign, SubAssign};
use rand::random;
use na::{Point3, Vector3, Matrix3, Rotation3, Isometry3, Similarity3, Quaternion, UnitQuaternion};

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
test_op_vs_op_assign!(test_vec3_f32_mul_assign, Vector3<f32>, f32, mul, mul_assign);
test_op_vs_op_assign!(test_mat3_f32_mul_assign, Matrix3<f32>, f32, mul, mul_assign);
test_op_vs_op_assign!(test_quaternion_f32_mul_assign, Quaternion<f32>, f32, mul, mul_assign);

test_op_vs_op_assign!(test_vec3_vec3_mul_assign, Vector3<f32>, Vector3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_quaternion_quaternion_mul_assign, Quaternion<f32>, Quaternion<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_unit_quaternion_unit_quaternion_mul_assign, UnitQuaternion<f32>, UnitQuaternion<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_vec3_unit_quaternion_mul_assign, Vector3<f32>, UnitQuaternion<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_point3_unit_quaternion_mul_assign, Point3<f32>, UnitQuaternion<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_mat3_mat3_mul_assign, Matrix3<f32>, Matrix3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_vec3_mat3_mul_assign, Vector3<f32>, Matrix3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_point3_mat3_mul_assign, Point3<f32>, Matrix3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_rot3_rot3_mul_assign, Rotation3<f32>, Rotation3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_vec3_rot3_mul_assign, Vector3<f32>, Rotation3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_point3_rot3_mul_assign, Point3<f32>, Rotation3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_iso3_iso3_mul_assign, Isometry3<f32>, Isometry3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_iso3_rot3_mul_assign, Isometry3<f32>, Rotation3<f32>, mul, mul_assign);

test_op_vs_op_assign!(test_sim3_sim3_mul_assign, Similarity3<f32>, Similarity3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_sim3_iso3_mul_assign, Similarity3<f32>, Isometry3<f32>, mul, mul_assign);
test_op_vs_op_assign!(test_sim3_rot3_mul_assign, Similarity3<f32>, Rotation3<f32>, mul, mul_assign);

// Division.
test_op_vs_op_assign!(test_vec3_vec3_div_assign, Vector3<f32>, Vector3<f32>, div, div_assign);
test_op_vs_op_assign!(test_quaternion_quaternion_div_assign, Quaternion<f32>, Quaternion<f32>, div, div_assign);
test_op_vs_op_assign!(test_unit_quaternion_unit_quaternion_div_assign, UnitQuaternion<f32>, UnitQuaternion<f32>, div, div_assign);

test_op_vs_op_assign!(test_vec3_f32_div_assign, Vector3<f32>, f32, div, div_assign);
test_op_vs_op_assign!(test_mat3_f32_div_assign, Matrix3<f32>, f32, div, div_assign);

// Addition.
test_op_vs_op_assign!(test_vec3_vec3_add_assign, Vector3<f32>, Vector3<f32>, add, add_assign);
test_op_vs_op_assign!(test_mat3_mat3_add_assign, Matrix3<f32>, Matrix3<f32>, add, add_assign);
test_op_vs_op_assign!(test_quaternion_quaternion_add_assign, Quaternion<f32>, Quaternion<f32>, add, add_assign);

test_op_vs_op_assign!(test_vec3_f32_add_assign, Vector3<f32>, f32, add, add_assign);
test_op_vs_op_assign!(test_mat3_f32_add_assign, Matrix3<f32>, f32, add, add_assign);

// Subtraction.
test_op_vs_op_assign!(test_vec3_vec3_sub_assign, Vector3<f32>, Vector3<f32>, sub, sub_assign);
test_op_vs_op_assign!(test_mat3_mat3_sub_assign, Matrix3<f32>, Matrix3<f32>, sub, sub_assign);
test_op_vs_op_assign!(test_quaternion_quaternion_sub_assign, Quaternion<f32>, Quaternion<f32>, sub, sub_assign);

test_op_vs_op_assign!(test_vec3_f32_sub_assign, Vector3<f32>, f32, sub, sub_assign);
test_op_vs_op_assign!(test_mat3_f32_sub_assign, Matrix3<f32>, f32, sub, sub_assign);
