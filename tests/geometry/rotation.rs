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

#[cfg(feature = "proptest-support")]
mod proptest_tests {
    use na::{self, Rotation2, Rotation3, Unit};
    use simba::scalar::RealField;
    use std::f64;

    use crate::proptest::*;
    use proptest::{prop_assert, prop_assert_eq, proptest};

    proptest! {
        /*
         *
         * Euler angles.
         *
         */
        #[test]
        fn from_euler_angles(r in PROPTEST_F64, p in PROPTEST_F64, y in PROPTEST_F64) {
            let roll  = Rotation3::from_euler_angles(r, 0.0, 0.0);
            let pitch = Rotation3::from_euler_angles(0.0, p, 0.0);
            let yaw   = Rotation3::from_euler_angles(0.0, 0.0, y);

            let rpy = Rotation3::from_euler_angles(r, p, y);

            prop_assert_eq!(roll[(0, 0)], 1.0); // rotation wrt. x axis.
            prop_assert_eq!(pitch[(1, 1)], 1.0); // rotation wrt. y axis.
            prop_assert_eq!(yaw[(2, 2)], 1.0); // rotation wrt. z axis.
            prop_assert_eq!(yaw * pitch * roll, rpy);
        }

        #[test]
        fn euler_angles(r in PROPTEST_F64, p in PROPTEST_F64, y in PROPTEST_F64) {
            let rpy = Rotation3::from_euler_angles(r, p, y);
            let (roll, pitch, yaw) = rpy.euler_angles();
            prop_assert!(relative_eq!(Rotation3::from_euler_angles(roll, pitch, yaw), rpy, epsilon = 1.0e-7));
        }

        #[test]
        fn euler_angles_gimble_lock(r in PROPTEST_F64, y in PROPTEST_F64) {
            let pos = Rotation3::from_euler_angles(r,  f64::frac_pi_2(), y);
            let neg = Rotation3::from_euler_angles(r, -f64::frac_pi_2(), y);
            let (pos_r, pos_p, pos_y) = pos.euler_angles();
            let (neg_r, neg_p, neg_y) = neg.euler_angles();
            prop_assert!(relative_eq!(Rotation3::from_euler_angles(pos_r, pos_p, pos_y), pos, epsilon = 1.0e-7));
            prop_assert!(relative_eq!(Rotation3::from_euler_angles(neg_r, neg_p, neg_y), neg, epsilon = 1.0e-7));
        }

        /*
         *
         * Inversion is transposition.
         *
         */
        #[test]
        fn rotation_inv_3(a in rotation3()) {
            let ta = a.transpose();
            let ia = a.inverse();

            prop_assert_eq!(ta, ia);
            prop_assert!(relative_eq!(&ta * &a,  Rotation3::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(&ia *  a,  Rotation3::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(  a * &ta, Rotation3::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(  a *  ia, Rotation3::identity(), epsilon = 1.0e-7));
        }

        #[test]
        fn rotation_inv_2(a in rotation2()) {
            let ta = a.transpose();
            let ia = a.inverse();

            prop_assert_eq!(ta, ia);
            prop_assert!(relative_eq!(&ta * &a,  Rotation2::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(&ia *  a,  Rotation2::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(  a * &ta, Rotation2::identity(), epsilon = 1.0e-7));
            prop_assert!(relative_eq!(  a *  ia, Rotation2::identity(), epsilon = 1.0e-7));
        }

        /*
         *
         * Angle between vectors.
         *
         */
        #[test]
        fn angle_is_commutative_2(a in vector2(), b in vector2()) {
            prop_assert_eq!(a.angle(&b), b.angle(&a))
        }

        #[test]
        fn angle_is_commutative_3(a in vector3(), b in vector3()) {
            prop_assert_eq!(a.angle(&b), b.angle(&a))
        }

        /*
         *
         * Rotation matrix between vectors.
         *
         */
        #[test]
        fn rotation_between_is_anticommutative_2(a in vector2(), b in vector2()) {
            let rab = Rotation2::rotation_between(&a, &b);
            let rba = Rotation2::rotation_between(&b, &a);

            prop_assert!(relative_eq!(rab * rba, Rotation2::identity()));
        }

        #[test]
        fn rotation_between_is_anticommutative_3(a in vector3(), b in vector3()) {
            let rots = (Rotation3::rotation_between(&a, &b), Rotation3::rotation_between(&b, &a));
            if let (Some(rab), Some(rba)) = rots {
                prop_assert!(relative_eq!(rab * rba, Rotation3::identity(), epsilon = 1.0e-7));
            }
        }

        #[test]
        fn rotation_between_is_identity(v2 in vector2(), v3 in vector3()) {
            let vv2 = 3.42 * v2;
            let vv3 = 4.23 * v3;

            prop_assert!(relative_eq!(v2.angle(&vv2), 0.0, epsilon = 1.0e-7));
            prop_assert!(relative_eq!(v3.angle(&vv3), 0.0, epsilon = 1.0e-7));
            prop_assert!(relative_eq!(Rotation2::rotation_between(&v2, &vv2), Rotation2::identity()));
            prop_assert_eq!(Rotation3::rotation_between(&v3, &vv3).unwrap(), Rotation3::identity());
        }

        #[test]
        fn rotation_between_2(a in vector2(), b in vector2()) {
            if !relative_eq!(a.angle(&b), 0.0, epsilon = 1.0e-7) {
                let r = Rotation2::rotation_between(&a, &b);
                prop_assert!(relative_eq!((r * a).angle(&b), 0.0, epsilon = 1.0e-7))
            }
        }

        #[test]
        fn rotation_between_3(a in vector3(), b in vector3()) {
            if !relative_eq!(a.angle(&b), 0.0, epsilon = 1.0e-7) {
                let r = Rotation3::rotation_between(&a, &b).unwrap();
                prop_assert!(relative_eq!((r * a).angle(&b), 0.0, epsilon = 1.0e-7))
            }
        }


        /*
         *
         * Rotation construction.
         *
         */
        #[test]
        fn new_rotation_2(angle in PROPTEST_F64) {
            let r = Rotation2::new(angle);

            let angle = na::wrap(angle, -f64::pi(), f64::pi());
            prop_assert!(relative_eq!(r.angle(), angle, epsilon = 1.0e-7))
        }

        #[test]
        fn new_rotation_3(axisangle in vector3()) {
            let r = Rotation3::new(axisangle);

            if let Some((axis, angle)) = Unit::try_new_and_get(axisangle, 0.0) {
                let angle = na::wrap(angle, -f64::pi(), f64::pi());
                prop_assert!((relative_eq!(r.angle(), angle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), axis, epsilon = 1.0e-7)) ||
                (relative_eq!(r.angle(), -angle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), -axis, epsilon = 1.0e-7)))
            }
            else {
                prop_assert_eq!(r, Rotation3::identity())
            }
        }

        /*
         *
         * Rotation pow.
         *
         */
        #[test]
        fn powf_rotation_2(angle in PROPTEST_F64, pow in PROPTEST_F64) {
            let r = Rotation2::new(angle).powf(pow);

            let angle  = na::wrap(angle, -f64::pi(), f64::pi());
            let pangle = na::wrap(angle * pow, -f64::pi(), f64::pi());
            prop_assert!(relative_eq!(r.angle(), pangle, epsilon = 1.0e-7));
        }

        #[test]
        fn powf_rotation_3(axisangle in vector3(), pow in PROPTEST_F64) {
            let r = Rotation3::new(axisangle).powf(pow);

            if let Some((axis, angle)) = Unit::try_new_and_get(axisangle, 0.0) {
                let angle = na::wrap(angle, -f64::pi(), f64::pi());
                let pangle = na::wrap(angle * pow, -f64::pi(), f64::pi());

                prop_assert!((relative_eq!(r.angle(), pangle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), axis, epsilon = 1.0e-7)) ||
                (relative_eq!(r.angle(), -pangle, epsilon = 1.0e-7) &&
                 relative_eq!(r.axis().unwrap(), -axis, epsilon = 1.0e-7)));
            }
            else {
                prop_assert_eq!(r, Rotation3::identity())
            }
        }
    }
}
