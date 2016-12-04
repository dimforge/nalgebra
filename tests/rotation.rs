#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;
#[macro_use]
extern crate approx;
extern crate num_traits as num;
extern crate alga;
extern crate nalgebra as na;

use alga::general::Real;
use na::{Vector2, Vector3, Rotation2, Rotation3, Unit};

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

quickcheck!(
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
     * RotationBase matrix between vectors.
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
     * RotationBase construction.
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
     * RotationBase pow.
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
);
