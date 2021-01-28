#![cfg(feature = "arbitrary")]
#![allow(non_snake_case)]

use na::{Isometry3, Point3, Translation3, UnitDualQuaternion, UnitQuaternion, Vector3};

quickcheck!(
    fn isometry_equivalence(iso: Isometry3<f64>, p: Point3<f64>, v: Vector3<f64>) -> bool {
        let dq = UnitDualQuaternion::from_isometry(&iso);

        relative_eq!(iso * p, dq * p, epsilon = 1.0e-7)
            && relative_eq!(iso * v, dq * v, epsilon = 1.0e-7)
    }

    fn inverse_is_identity(i: UnitDualQuaternion<f64>, p: Point3<f64>, v: Vector3<f64>) -> bool {
        let ii = i.inverse();

        relative_eq!(i * ii, UnitDualQuaternion::identity(), epsilon = 1.0e-7)
            && relative_eq!(ii * i, UnitDualQuaternion::identity(), epsilon = 1.0e-7)
            && relative_eq!((i * ii) * p, p, epsilon = 1.0e-7)
            && relative_eq!((ii * i) * p, p, epsilon = 1.0e-7)
            && relative_eq!((i * ii) * v, v, epsilon = 1.0e-7)
            && relative_eq!((ii * i) * v, v, epsilon = 1.0e-7)
    }

    fn multiply_equals_alga_transform(
        dq: UnitDualQuaternion<f64>,
        v: Vector3<f64>,
        p: Point3<f64>
    ) -> bool {
        dq * v == dq.transform_vector(&v)
            && dq * p == dq.transform_point(&p)
            && relative_eq!(
                dq.inverse() * v,
                dq.inverse_transform_vector(&v),
                epsilon = 1.0e-7
            )
            && relative_eq!(
                dq.inverse() * p,
                dq.inverse_transform_point(&p),
                epsilon = 1.0e-7
            )
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn composition(
        dq: UnitDualQuaternion<f64>,
        uq: UnitQuaternion<f64>,
        t: Translation3<f64>,
        v: Vector3<f64>,
        p: Point3<f64>
    ) -> bool {
        // (rotation × dual quaternion) * point = rotation × (dual quaternion * point)
        relative_eq!((uq * dq) * v, uq * (dq * v), epsilon = 1.0e-7) &&
        relative_eq!((uq * dq) * p, uq * (dq * p), epsilon = 1.0e-7) &&

        // (dual quaternion × rotation) * point = dual quaternion × (rotation * point)
        relative_eq!((dq * uq) * v, dq * (uq * v), epsilon = 1.0e-7) &&
        relative_eq!((dq * uq) * p, dq * (uq * p), epsilon = 1.0e-7) &&

        // (translation × dual quaternion) * point = translation × (dual quaternion * point)
        relative_eq!((t * dq) * v,     (dq * v), epsilon = 1.0e-7) &&
        relative_eq!((t * dq) * p, t * (dq * p), epsilon = 1.0e-7) &&

        // (dual quaternion × translation) * point = dual quaternion × (translation * point)
        relative_eq!((dq * t) * v, dq * v,       epsilon = 1.0e-7) &&
        relative_eq!((dq * t) * p, dq * (t * p), epsilon = 1.0e-7)
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn all_op_exist(
        dq: UnitDualQuaternion<f64>,
        uq: UnitQuaternion<f64>,
        t: Translation3<f64>,
        v: Vector3<f64>,
        p: Point3<f64>
    ) -> bool {
        let iMi = dq * dq;
        let iMuq = dq * uq;
        let iDi = dq / dq;
        let iDuq = dq / uq;

        let iMp = dq * p;
        let iMv = dq * v;

        let iMt = dq * t;
        let tMi = t * dq;

        let tMuq = t * uq;

        let uqMi = uq * dq;
        let uqDi = uq / dq;

        let uqMt = uq * t;

        let mut iMt1 = dq;
        let mut iMt2 = dq;

        let mut iMi1 = dq;
        let mut iMi2 = dq;

        let mut iMuq1 = dq;
        let mut iMuq2 = dq;

        let mut iDi1 = dq;
        let mut iDi2 = dq;

        let mut iDuq1 = dq;
        let mut iDuq2 = dq;

        iMt1 *= t;
        iMt2 *= &t;

        iMi1 *= dq;
        iMi2 *= &dq;

        iMuq1 *= uq;
        iMuq2 *= &uq;

        iDi1 /= dq;
        iDi2 /= &dq;

        iDuq1 /= uq;
        iDuq2 /= &uq;

        iMt == iMt1
            && iMt == iMt2
            && iMi == iMi1
            && iMi == iMi2
            && iMuq == iMuq1
            && iMuq == iMuq2
            && iDi == iDi1
            && iDi == iDi2
            && iDuq == iDuq1
            && iDuq == iDuq2
            && iMi == &dq * &dq
            && iMi == dq * &dq
            && iMi == &dq * dq
            && iMuq == &dq * &uq
            && iMuq == dq * &uq
            && iMuq == &dq * uq
            && iDi == &dq / &dq
            && iDi == dq / &dq
            && iDi == &dq / dq
            && iDuq == &dq / &uq
            && iDuq == dq / &uq
            && iDuq == &dq / uq
            && iMp == &dq * &p
            && iMp == dq * &p
            && iMp == &dq * p
            && iMv == &dq * &v
            && iMv == dq * &v
            && iMv == &dq * v
            && iMt == &dq * &t
            && iMt == dq * &t
            && iMt == &dq * t
            && tMi == &t * &dq
            && tMi == t * &dq
            && tMi == &t * dq
            && tMuq == &t * &uq
            && tMuq == t * &uq
            && tMuq == &t * uq
            && uqMi == &uq * &dq
            && uqMi == uq * &dq
            && uqMi == &uq * dq
            && uqDi == &uq / &dq
            && uqDi == uq / &dq
            && uqDi == &uq / dq
            && uqMt == &uq * &t
            && uqMt == uq * &t
            && uqMt == &uq * t
    }
);
