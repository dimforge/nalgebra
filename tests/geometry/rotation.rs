use na::{Quaternion, RealField, UnitQuaternion, Vector2, Vector3};

#[test]
fn angle_2() {
    let a = Vector2::new(4.0, 0.0);
    let b = Vector2::new(9.0, 0.0);

    assert_eq!(a.angle(&b), 0.0);
}

#[test]
fn angle_3() {
    let a = Vector3::new(4.0, 0.0, 0.5);
    let b = Vector3::new(8.0, 0.0, 1.0);

    assert_eq!(a.angle(&b), 0.0);
}

#[test]
fn quaternion_euler_angles_issue_494() {
    let quat = UnitQuaternion::from_quaternion(Quaternion::new(
        -0.10405792,
        -0.6993922f32,
        -0.10406871,
        0.69942284,
    ));
    let angs = quat.euler_angles();
    assert_eq!(angs.0, 2.8461843);
    assert_eq!(angs.1, f32::frac_pi_2());
    assert_eq!(angs.2, 0.0);
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    use na::{self, Rotation2, Rotation3, Unit, Vector2, Vector3};
    use simba::scalar::RealField;
    use std::f64;

    quickcheck! {
        /*
         *
         * Euler angles.
         *
         */
        fn from_euler_angles(r: f64, p: f64, y: f64) -> bool {
            let roll  = Rotation3::from_euler_angles(r, 0.0, 0.0);
            let pitch = Rotation3::from_euler_angles(0.0, p, 0.0);
            let yaw   = Rotation3::from_euler_angles(0.0, 0.0, y);

            let rpy = Rotation3::from_euler_angles(r, p, y);

            roll[(0, 0)]  == 1.0 && // rotation wrt. x axis.
            pitch[(1, 1)] == 1.0 && // rotation wrt. y axis.
            yaw[(2, 2)]   == 1.0 && // rotation wrt. z axis.
            yaw * pitch * roll == rpy
        }

        fn euler_angles(r: f64, p: f64, y: f64) -> bool {
            let rpy = Rotation3::from_euler_angles(r, p, y);
            let (roll, pitch, yaw) = rpy.euler_angles();
            relative_eq!(Rotation3::from_euler_angles(roll, pitch, yaw), rpy, epsilon = 1.0e-7)
        }

        fn euler_angles_gimble_lock(r: f64, y: f64) -> bool {
            let pos = Rotation3::from_euler_angles(r,  f64::frac_pi_2(), y);
            let neg = Rotation3::from_euler_angles(r, -f64::frac_pi_2(), y);
            let (pos_r, pos_p, pos_y) = pos.euler_angles();
            let (neg_r, neg_p, neg_y) = neg.euler_angles();
            relative_eq!(Rotation3::from_euler_angles(pos_r, pos_p, pos_y), pos, epsilon = 1.0e-7) &&
            relative_eq!(Rotation3::from_euler_angles(neg_r, neg_p, neg_y), neg, epsilon = 1.0e-7)
        }

        /*
         *
         * Inversion is transposition.
         *
         */
        fn rotation_inv_3(a: Rotation3<f64>) -> bool {
            let ta = a.transpose();
            let ia = a.inverse();

            ta == ia &&
            relative_eq!(&ta * &a,  Rotation3::identity(), epsilon = 1.0e-7) &&
            relative_eq!(&ia *  a,  Rotation3::identity(), epsilon = 1.0e-7) &&
            relative_eq!(  a * &ta, Rotation3::identity(), epsilon = 1.0e-7) &&
            relative_eq!(  a *  ia, Rotation3::identity(), epsilon = 1.0e-7)
        }

        fn rotation_inv_2(a: Rotation2<f64>) -> bool {
            let ta = a.transpose();
            let ia = a.inverse();

            ta == ia &&
            relative_eq!(&ta * &a,  Rotation2::identity(), epsilon = 1.0e-7) &&
            relative_eq!(&ia *  a,  Rotation2::identity(), epsilon = 1.0e-7) &&
            relative_eq!(  a * &ta, Rotation2::identity(), epsilon = 1.0e-7) &&
            relative_eq!(  a *  ia, Rotation2::identity(), epsilon = 1.0e-7)
        }

        /*
         *
         * Angle between vectors.
         *
         */
        fn angle_is_commutative_2(a: Vector2<f64>, b: Vector2<f64>) -> bool {
            a.angle(&b) == b.angle(&a)
        }

        fn angle_is_commutative_3(a: Vector3<f64>, b: Vector3<f64>) -> bool {
            a.angle(&b) == b.angle(&a)
        }

        /*
         *
         * Rotation matrix between vectors.
         *
         */
        fn rotation_between_is_anticommutative_2(a: Vector2<f64>, b: Vector2<f64>) -> bool {
            let rab = Rotation2::rotation_between(&a, &b);
            let rba = Rotation2::rotation_between(&b, &a);

            relative_eq!(rab * rba, Rotation2::identity())
        }

        fn rotation_between_is_anticommutative_3(a: Vector3<f64>, b: Vector3<f64>) -> bool {
            let rots = (Rotation3::rotation_between(&a, &b), Rotation3::rotation_between(&b, &a));
            if let (Some(rab), Some(rba)) = rots {
                relative_eq!(rab * rba, Rotation3::identity(), epsilon = 1.0e-7)
            }
            else {
                true
            }
        }

        fn rotation_between_is_identity(v2: Vector2<f64>, v3: Vector3<f64>) -> bool {
            let vv2 = 3.42 * v2;
            let vv3 = 4.23 * v3;

            relative_eq!(v2.angle(&vv2), 0.0, epsilon = 1.0e-7) &&
            relative_eq!(v3.angle(&vv3), 0.0, epsilon = 1.0e-7) &&
            relative_eq!(Rotation2::rotation_between(&v2, &vv2), Rotation2::identity()) &&
            Rotation3::rotation_between(&v3, &vv3).unwrap() == Rotation3::identity()
        }

        fn rotation_between_2(a: Vector2<f64>, b: Vector2<f64>) -> bool {
            if !relative_eq!(a.angle(&b), 0.0, epsilon = 1.0e-7) {
                let r = Rotation2::rotation_between(&a, &b);
                relative_eq!((r * a).angle(&b), 0.0, epsilon = 1.0e-7)
            }
            else {
                true
            }
        }

        fn rotation_between_3(a: Vector3<f64>, b: Vector3<f64>) -> bool {
            if !relative_eq!(a.angle(&b), 0.0, epsilon = 1.0e-7) {
                let r = Rotation3::rotation_between(&a, &b).unwrap();
                relative_eq!((r * a).angle(&b), 0.0, epsilon = 1.0e-7)
            }
            else {
                true
            }
        }


        /*
         *
         * Rotation construction.
         *
         */
        fn new_rotation_2(angle: f64) -> bool {
            let r = Rotation2::new(angle);

            let angle = na::wrap(angle, -f64::pi(), f64::pi());
            relative_eq!(r.angle(), angle, epsilon = 1.0e-7)
        }

        fn new_rotation_3(axisangle: Vector3<f64>) -> bool {
            let r = Rotation3::new(axisangle);

            if let Some((axis, angle)) = Unit::try_new_and_get(axisangle, 0.0) {
                let angle = na::wrap(angle, -f64::pi(), f64::pi());
                (relative_eq!(r.angle(), angle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), axis, epsilon = 1.0e-7)) ||
                (relative_eq!(r.angle(), -angle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), -axis, epsilon = 1.0e-7))
            }
            else {
                r == Rotation3::identity()
            }
        }

        /*
         *
         * Rotation pow.
         *
         */
        fn powf_rotation_2(angle: f64, pow: f64) -> bool {
            let r = Rotation2::new(angle).powf(pow);

            let angle  = na::wrap(angle, -f64::pi(), f64::pi());
            let pangle = na::wrap(angle * pow, -f64::pi(), f64::pi());
            relative_eq!(r.angle(), pangle, epsilon = 1.0e-7)
        }

        fn powf_rotation_3(axisangle: Vector3<f64>, pow: f64) -> bool {
            let r = Rotation3::new(axisangle).powf(pow);

            if let Some((axis, angle)) = Unit::try_new_and_get(axisangle, 0.0) {
                let angle = na::wrap(angle, -f64::pi(), f64::pi());
                let pangle = na::wrap(angle * pow, -f64::pi(), f64::pi());

                (relative_eq!(r.angle(), pangle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), axis, epsilon = 1.0e-7)) ||
                (relative_eq!(r.angle(), -pangle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), -axis, epsilon = 1.0e-7))
            }
            else {
                r == Rotation3::identity()
            }
        }
    }
}
