#![feature(macro_rules)]

extern crate test;
extern crate "nalgebra" as na;

use std::rand::{IsaacRng, Rng};
use test::Bencher;
use na::{UnitQuat, Rot2, Rot3, Vec1, Vec3};

#[path="common/macros.rs"]
mod macros;

bench_construction!(_bench_quat_from_axisangle, UnitQuat::new, axisangle: Vec3<f32>)
bench_construction!(_bench_rot2_from_axisangle, Rot2::new, axisangle: Vec1<f32>)
bench_construction!(_bench_rot3_from_axisangle, Rot3::new, axisangle: Vec3<f32>)

bench_construction!(_bench_quat_from_euler_angles, UnitQuat::new_with_euler_angles, roll: f32, pitch: f32, yaw: f32)
bench_construction!(_bench_rot3_from_euler_angles, Rot3::new_with_euler_angles, roll: f32, pitch: f32, yaw: f32)
