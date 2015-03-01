extern crate "nalgebra" as na;
extern crate rand;

use na::{Pnt3, Vec3, Rot3, UnitQuat, Rotation};
use rand::random;

#[test]
fn test_quat_as_mat() {
    for _ in (0usize .. 10000) {
        let axis_angle: Vec3<f64> = random();

        assert!(na::approx_eq(&UnitQuat::new(axis_angle).to_rot(), &Rot3::new(axis_angle)))
    }
}

#[test]
fn test_quat_mul_vec_or_pnt_as_mat() {
    for _ in (0usize .. 10000) {
        let axis_angle: Vec3<f64> = random();
        let vec: Vec3<f64> = random();
        let pnt: Pnt3<f64> = random();

        let mat  = Rot3::new(axis_angle);
        let quat = UnitQuat::new(axis_angle);

        assert!(na::approx_eq(&(mat * vec), &(quat * vec)));
        assert!(na::approx_eq(&(mat * pnt), &(quat * pnt)));
        assert!(na::approx_eq(&(vec * mat), &(vec * quat)));
        assert!(na::approx_eq(&(pnt * mat), &(pnt * quat)));
    }
}

#[test]
fn test_quat_div_quat() {
    for _ in (0usize .. 10000) {
        let axis_angle1: Vec3<f64> = random();
        let axis_angle2: Vec3<f64> = random();

        let r1 = Rot3::new(axis_angle1);
        let r2 = na::inv(&Rot3::new(axis_angle2)).unwrap();

        let q1 = UnitQuat::new(axis_angle1);
        let q2 = UnitQuat::new(axis_angle2);

        assert!(na::approx_eq(&(q1 / q2).to_rot(), &(r1 * r2)))
    }
}

#[test]
fn test_quat_to_axis_angle() {
    for _ in (0usize .. 10000) {
        let axis_angle: Vec3<f64> = random();

        let q = UnitQuat::new(axis_angle);

        println!("{:?} {:?}", q.rotation(), axis_angle);
        assert!(na::approx_eq(&q.rotation(), &axis_angle))
    }
}

#[test]
fn test_quat_euler_angles() {
    for _ in (0usize .. 10000) {
        let angles: Vec3<f64> = random();

        let q = UnitQuat::new_with_euler_angles(angles.x, angles.y, angles.z);
        let m = Rot3::new_with_euler_angles(angles.x, angles.y, angles.z);

        assert!(na::approx_eq(&q.to_rot(), &m))
    }
}
