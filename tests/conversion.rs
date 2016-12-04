#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;
#[macro_use]
extern crate approx;
extern crate num_traits as num;
extern crate alga;
extern crate nalgebra as na;

use alga::linear::Transformation;
use na::{Vector3, Point3, Translation3, Isometry3, Similarity3, Affine3, Projective3, Transform3,
         Rotation3, UnitQuaternion};


#[cfg(feature = "arbitrary")]
quickcheck!{
    fn translation_conversion(t: Translation3<f64>, v: Vector3<f64>, p: Point3<f64>) -> bool {
        let iso: Isometry3<f64>   = na::convert(t);
        let sim: Similarity3<f64> = na::convert(t);
        let aff: Affine3<f64>     = na::convert(t);
        let prj: Projective3<f64> = na::convert(t);
        let tr:  Transform3<f64>  = na::convert(t);

        t == na::try_convert(iso).unwrap() &&
        t == na::try_convert(sim).unwrap() &&
        t == na::try_convert(aff).unwrap() &&
        t == na::try_convert(prj).unwrap() &&
        t == na::try_convert(tr).unwrap()  &&

        t.transform_vector(&v) == iso * v &&
        t.transform_vector(&v) == sim * v &&
        t.transform_vector(&v) == aff * v &&
        t.transform_vector(&v) == prj * v &&
        t.transform_vector(&v) == tr  * v &&

        t * p == iso * p &&
        t * p == sim * p &&
        t * p == aff * p &&
        t * p == prj * p &&
        t * p == tr  * p
    }

    fn rotation_conversion(r: Rotation3<f64>, v: Vector3<f64>, p: Point3<f64>) -> bool {
        let uq:  UnitQuaternion<f64> = na::convert(r);
        let iso: Isometry3<f64>      = na::convert(r);
        let sim: Similarity3<f64>    = na::convert(r);
        let aff: Affine3<f64>        = na::convert(r);
        let prj: Projective3<f64>    = na::convert(r);
        let tr:  Transform3<f64>     = na::convert(r);

        relative_eq!(r, na::try_convert(uq).unwrap(),  epsilon = 1.0e-7) &&
        relative_eq!(r, na::try_convert(iso).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(r, na::try_convert(sim).unwrap(), epsilon = 1.0e-7) &&
        r == na::try_convert(aff).unwrap() &&
        r == na::try_convert(prj).unwrap() &&
        r == na::try_convert(tr).unwrap()  &&

        // NOTE: we need relative_eq because IsometryBase and SimilarityBase use quaternions.
        relative_eq!(r * v, uq  * v, epsilon = 1.0e-7) &&
        relative_eq!(r * v, iso * v, epsilon = 1.0e-7) &&
        relative_eq!(r * v, sim * v, epsilon = 1.0e-7) &&
        r * v == aff * v &&
        r * v == prj * v &&
        r * v == tr  * v &&

        relative_eq!(r * p, uq  * p, epsilon = 1.0e-7) &&
        relative_eq!(r * p, iso * p, epsilon = 1.0e-7) &&
        relative_eq!(r * p, sim * p, epsilon = 1.0e-7) &&
        r * p == aff * p &&
        r * p == prj * p &&
        r * p == tr  * p
    }

    fn unit_quaternion_conversion(uq: UnitQuaternion<f64>, v: Vector3<f64>, p: Point3<f64>) -> bool {
        let rot: Rotation3<f64>   = na::convert(uq);
        let iso: Isometry3<f64>   = na::convert(uq);
        let sim: Similarity3<f64> = na::convert(uq);
        let aff: Affine3<f64>     = na::convert(uq);
        let prj: Projective3<f64> = na::convert(uq);
        let tr:  Transform3<f64>  = na::convert(uq);

        uq == na::try_convert(iso).unwrap() &&
        uq == na::try_convert(sim).unwrap() &&
        relative_eq!(uq, na::try_convert(rot).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(uq, na::try_convert(aff).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(uq, na::try_convert(prj).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(uq, na::try_convert(tr).unwrap(), epsilon = 1.0e-7)  &&

        // NOTE: iso and sim use unit quaternions for the rotation so conversions to them are exact.
        relative_eq!(uq * v, rot * v, epsilon = 1.0e-7) &&
        uq * v == iso * v &&
        uq * v == sim * v &&
        relative_eq!(uq * v, aff * v, epsilon = 1.0e-7) &&
        relative_eq!(uq * v, prj * v, epsilon = 1.0e-7) &&
        relative_eq!(uq * v, tr  * v, epsilon = 1.0e-7) &&

        relative_eq!(uq * p, rot * p, epsilon = 1.0e-7) &&
        uq * p == iso * p &&
        uq * p == sim * p &&
        relative_eq!(uq * p, aff * p, epsilon = 1.0e-7) &&
        relative_eq!(uq * p, prj * p, epsilon = 1.0e-7) &&
        relative_eq!(uq * p, tr  * p, epsilon = 1.0e-7)
    }

    fn isometry_conversion(iso: Isometry3<f64>, v: Vector3<f64>, p: Point3<f64>) -> bool {
        let sim: Similarity3<f64> = na::convert(iso);
        let aff: Affine3<f64>     = na::convert(iso);
        let prj: Projective3<f64> = na::convert(iso);
        let tr:  Transform3<f64>  = na::convert(iso);


        iso == na::try_convert(sim).unwrap() &&
        relative_eq!(iso, na::try_convert(aff).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(iso, na::try_convert(prj).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(iso, na::try_convert(tr).unwrap(), epsilon = 1.0e-7)  &&

        iso * v == sim * v &&
        relative_eq!(iso * v, aff * v, epsilon = 1.0e-7) &&
        relative_eq!(iso * v, prj * v, epsilon = 1.0e-7) &&
        relative_eq!(iso * v, tr  * v, epsilon = 1.0e-7) &&

        iso * p == sim * p &&
        relative_eq!(iso * p, aff * p, epsilon = 1.0e-7) &&
        relative_eq!(iso * p, prj * p, epsilon = 1.0e-7) &&
        relative_eq!(iso * p, tr  * p, epsilon = 1.0e-7)
    }

    fn similarity_conversion(sim: Similarity3<f64>, v: Vector3<f64>, p: Point3<f64>) -> bool {
        let aff: Affine3<f64>     = na::convert(sim);
        let prj: Projective3<f64> = na::convert(sim);
        let tr:  Transform3<f64>  = na::convert(sim);

        relative_eq!(sim, na::try_convert(aff).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(sim, na::try_convert(prj).unwrap(), epsilon = 1.0e-7) &&
        relative_eq!(sim, na::try_convert(tr).unwrap(),  epsilon = 1.0e-7) &&

        relative_eq!(sim * v, aff * v, epsilon = 1.0e-7) &&
        relative_eq!(sim * v, prj * v, epsilon = 1.0e-7) &&
        relative_eq!(sim * v, tr  * v, epsilon = 1.0e-7) &&

        relative_eq!(sim * p, aff * p, epsilon = 1.0e-7) &&
        relative_eq!(sim * p, prj * p, epsilon = 1.0e-7) &&
        relative_eq!(sim * p, tr  * p, epsilon = 1.0e-7)
    }

    // XXX test TransformBase
}
