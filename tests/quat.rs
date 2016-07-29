extern crate nalgebra as na;
extern crate rand;

use na::{Point3, Quaternion, Vector3, Rotation3, UnitQuaternion, Rotation, one};
use rand::random;

#[test]
fn test_quaternion_as_matrix() {
    for _ in 0usize .. 10000 {
        let axis_angle: Vector3<f64> = random();

        assert!(na::approx_eq(&UnitQuaternion::new(axis_angle).to_rotation_matrix(), &Rotation3::new(axis_angle)))
    }
}

#[test]
fn test_quaternion_mul_vec_or_point_as_matrix() {
    for _ in 0usize .. 10000 {
        let axis_angle: Vector3<f64> = random();
        let vector: Vector3<f64> = random();
        let point: Point3<f64> = random();

        let matrix  = Rotation3::new(axis_angle);
        let quaternion = UnitQuaternion::new(axis_angle);

        assert!(na::approx_eq(&(matrix * vector), &(quaternion * vector)));
        assert!(na::approx_eq(&(matrix * point), &(quaternion * point)));
        assert!(na::approx_eq(&(vector * matrix), &(vector * quaternion)));
        assert!(na::approx_eq(&(point * matrix), &(point * quaternion)));
    }
}

#[test]
fn test_quaternion_div_quaternion() {
    for _ in 0usize .. 10000 {
        let axis_angle1: Vector3<f64> = random();
        let axis_angle2: Vector3<f64> = random();

        let r1 = Rotation3::new(axis_angle1);
        let r2 = na::inverse(&Rotation3::new(axis_angle2)).unwrap();

        let q1 = UnitQuaternion::new(axis_angle1);
        let q2 = UnitQuaternion::new(axis_angle2);

        assert!(na::approx_eq(&(q1 / q2).to_rotation_matrix(), &(r1 * r2)))
    }
}

#[test]
fn test_quaternion_to_axis_angle() {
    for _ in 0usize .. 10000 {
        let axis_angle: Vector3<f64> = random();

        let q = UnitQuaternion::new(axis_angle);

        println!("{:?} {:?}", q.rotation(), axis_angle);
        assert!(na::approx_eq(&q.rotation(), &axis_angle))
    }
}

#[test]
fn test_quaternion_euler_angles() {
    for _ in 0usize .. 10000 {
        let angles: Vector3<f64> = random();

        let q = UnitQuaternion::new_with_euler_angles(angles.x, angles.y, angles.z);
        let m = Rotation3::new_with_euler_angles(angles.x, angles.y, angles.z);

        assert!(na::approx_eq(&q.to_rotation_matrix(), &m))
    }
}

#[test]
fn test_quaternion_rotation_between() {
    let q1: UnitQuaternion<f64> = random();
    let q2: UnitQuaternion<f64> = random();

    let delta = na::rotation_between(&q1, &q2);

    assert!(na::approx_eq(&(delta * q1), &q2))
}

#[test]
fn test_quaternion_angle_between() {
    let q1: UnitQuaternion<f64> = random();
    let q2: UnitQuaternion<f64> = random();

    let delta = na::rotation_between(&q1, &q2);
    let delta_angle = na::angle_between(&q1, &q2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}

#[test]
fn test_quaternion_exp_zero_is_one() {
    let q = Quaternion::new(0., 0., 0., 0.);
    assert!(na::approx_eq(&q.exp(), &one()))
}
