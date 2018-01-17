#![cfg(feature = "arbitrary")]
#![allow(non_snake_case)]

use na::{Unit, UnitComplex, Vector2, Point2, Rotation2};

quickcheck!(

    /*
     *
     * From/to rotation matrix.
     *
     */
    fn unit_complex_rotation_conversion(c: UnitComplex<f64>) -> bool {
        let r  = c.to_rotation_matrix();
        let cc = UnitComplex::from_rotation_matrix(&r);
        let rr = cc.to_rotation_matrix();

        relative_eq!(c, cc, epsilon = 1.0e-7) &&
        relative_eq!(r, rr, epsilon = 1.0e-7)
    }

    /*
     *
     * Point/Vector transformation.
     *
     */
    fn unit_complex_transformation(c: UnitComplex<f64>, v: Vector2<f64>, p: Point2<f64>) -> bool {
        let r  = c.to_rotation_matrix();
        let rv = r * v;
        let rp = r * p;

        relative_eq!( c *  v, rv, epsilon = 1.0e-7) &&
        relative_eq!( c * &v, rv, epsilon = 1.0e-7) &&
        relative_eq!(&c *  v, rv, epsilon = 1.0e-7) &&
        relative_eq!(&c * &v, rv, epsilon = 1.0e-7) &&

        relative_eq!( c *  p, rp, epsilon = 1.0e-7) &&
        relative_eq!( c * &p, rp, epsilon = 1.0e-7) &&
        relative_eq!(&c *  p, rp, epsilon = 1.0e-7) &&
        relative_eq!(&c * &p, rp, epsilon = 1.0e-7)
    }

    /*
     *
     * Inversion.
     *
     */
    fn unit_complex_inv(c: UnitComplex<f64>) -> bool {
        let iq = c.inverse();
        relative_eq!(&iq * &c, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!( iq * &c, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!(&iq *  c, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!( iq *  c, UnitComplex::identity(), epsilon = 1.0e-7) &&

        relative_eq!(&c * &iq, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!( c * &iq, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!(&c *  iq, UnitComplex::identity(), epsilon = 1.0e-7) &&
        relative_eq!( c *  iq, UnitComplex::identity(), epsilon = 1.0e-7)
    }

    /*
     *
     * Quaterion * Vector == Rotation * Vector
     *
     */
    fn unit_complex_mul_vector(c: UnitComplex<f64>, v: Vector2<f64>, p: Point2<f64>) -> bool {
        let r = c.to_rotation_matrix();

        relative_eq!(c * v, r * v, epsilon = 1.0e-7) &&
        relative_eq!(c * p, r * p, epsilon = 1.0e-7)
    }

    // Test that all operators (incl. all combinations of references) work.
    // See the top comment on `geometry/quaternion_ops.rs` for details on which operations are
    // supported.
    fn all_op_exist(uc: UnitComplex<f64>, v: Vector2<f64>, p: Point2<f64>, r: Rotation2<f64>) -> bool {
        let uv = Unit::new_normalize(v);

        let ucMuc = uc * uc;
        let ucMr  = uc * r;
        let rMuc  = r  * uc;
        let ucDuc = uc / uc;
        let ucDr  = uc / r;
        let rDuc  = r / uc;

        let ucMp  = uc * p;
        let ucMv  = uc * v;
        let ucMuv = uc * uv;

        let mut ucMuc1 = uc;
        let mut ucMuc2 = uc;

        let mut ucMr1 = uc;
        let mut ucMr2 = uc;

        let mut ucDuc1 = uc;
        let mut ucDuc2 = uc;

        let mut ucDr1 = uc;
        let mut ucDr2 = uc;

        ucMuc1 *= uc;
        ucMuc2 *= &uc;

        ucMr1 *= r;
        ucMr2 *= &r;

        ucDuc1 /= uc;
        ucDuc2 /= &uc;

        ucDr1 /= r;
        ucDr2 /= &r;

        ucMuc1 == ucMuc  &&
        ucMuc1 == ucMuc2 &&

        ucMr1  == ucMr   &&
        ucMr1  == ucMr2  &&

        ucDuc1 == ucDuc  &&
        ucDuc1 == ucDuc2 &&

        ucDr1  == ucDr   &&
        ucDr1  == ucDr2  &&

        ucMuc == &uc * &uc &&
        ucMuc ==  uc * &uc &&
        ucMuc == &uc *  uc &&

        ucMr == &uc * &r &&
        ucMr ==  uc * &r &&
        ucMr == &uc *  r &&

        rMuc == &r * &uc &&
        rMuc ==  r * &uc &&
        rMuc == &r *  uc &&

        ucDuc == &uc / &uc &&
        ucDuc ==  uc / &uc &&
        ucDuc == &uc /  uc &&

        ucDr == &uc / &r &&
        ucDr ==  uc / &r &&
        ucDr == &uc /  r &&

        rDuc == &r / &uc &&
        rDuc ==  r / &uc &&
        rDuc == &r /  uc &&

        ucMp == &uc * &p &&
        ucMp ==  uc * &p &&
        ucMp == &uc *  p &&

        ucMv == &uc * &v &&
        ucMv ==  uc * &v &&
        ucMv == &uc *  v &&

        ucMuv == &uc * &uv &&
        ucMuv ==  uc * &uv &&
        ucMuv == &uc *  uv
    }
);
