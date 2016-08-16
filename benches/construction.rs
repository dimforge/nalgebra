#![feature(test)]

extern crate test;
extern crate rand;
extern crate nalgebra as na;

use rand::{IsaacRng, Rng};
use test::Bencher;
use na::{UnitQuaternion, Rotation2, Rotation3, Vector1, Vector3};

#[path="common/macros.rs"]
mod macros;

bench_construction!(_bench_quaternion_from_axisangle, UnitQuaternion::from_scaled_axis, axisangle: Vector3<f32>);
bench_construction!(_bench_rot2_from_axisangle, Rotation2::new, axisangle: Vector1<f32>);
bench_construction!(_bench_rot3_from_axisangle, Rotation3::new, axisangle: Vector3<f32>);

bench_construction!(_bench_quaternion_from_euler_angles, UnitQuaternion::from_euler_angles, roll: f32, pitch: f32, yaw: f32);
bench_construction!(_bench_rot3_from_euler_angles, Rotation3::from_euler_angles, roll: f32, pitch: f32, yaw: f32);
