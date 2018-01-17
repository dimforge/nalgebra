#![cfg(feature = "arbitrary")]
use alga::linear::Transformation;
use na::{
    self,
    Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
    RowVector1, RowVector2, RowVector3, RowVector4, RowVector5, RowVector6,
    Matrix2, Matrix3, Matrix4, Matrix5, Matrix6,
    Matrix2x3, Matrix2x4, Matrix2x5, Matrix2x6,
    Matrix3x2, Matrix3x4, Matrix3x5, Matrix3x6,
    Matrix4x2, Matrix4x3, Matrix4x5, Matrix4x6,
    Matrix5x2, Matrix5x3, Matrix5x4, Matrix5x6,
    Matrix6x2, Matrix6x3, Matrix6x4, Matrix6x5,
    Point3, Translation3, Isometry3, Similarity3, Affine3,
    Projective3, Transform3, Rotation3, UnitQuaternion};


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

        // NOTE: we need relative_eq because Isometry and Similarity use quaternions.
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

    // XXX test Transform
}

macro_rules! array_vector_conversion(
    ($($array_vector_conversion_i: ident, $Vector: ident, $SZ: expr);* $(;)*) => {$(
        #[test]
        fn $array_vector_conversion_i() {
            let v       = $Vector::from_fn(|i, _| i);
            let arr: [usize; $SZ] = v.into();
            let arr_ref: &[usize; $SZ] = v.as_ref();
            let v2      = $Vector::from(arr);

            for i in 0 .. $SZ {
                assert_eq!(arr[i], i);
                assert_eq!(arr_ref[i], i);
            }

            assert_eq!(v, v2);
        }
    )*}
);

array_vector_conversion!(
    array_vector_conversion_1,  Vector1,  1;
    array_vector_conversion_2,  Vector2,  2;
    array_vector_conversion_3,  Vector3,  3;
    array_vector_conversion_4,  Vector4,  4;
    array_vector_conversion_5,  Vector5,  5;
    array_vector_conversion_6,  Vector6,  6;
);

macro_rules! array_row_vector_conversion(
    ($($array_vector_conversion_i: ident, $Vector: ident, $SZ: expr);* $(;)*) => {$(
        #[test]
        fn $array_vector_conversion_i() {
            let v       = $Vector::from_fn(|_, i| i);
            let arr: [usize; $SZ] = v.into();
            let arr_ref = v.as_ref();
            let v2      = $Vector::from(arr);

            for i in 0 .. $SZ {
                assert_eq!(arr[i], i);
                assert_eq!(arr_ref[i], i);
            }

            assert_eq!(v, v2);
        }
    )*}
);

array_row_vector_conversion!(
    array_row_vector_conversion_1,  RowVector1,  1;
    array_row_vector_conversion_2,  RowVector2,  2;
    array_row_vector_conversion_3,  RowVector3,  3;
    array_row_vector_conversion_4,  RowVector4,  4;
    array_row_vector_conversion_5,  RowVector5,  5;
    array_row_vector_conversion_6,  RowVector6,  6;
);

macro_rules! array_matrix_conversion(
    ($($array_matrix_conversion_i_j: ident, $Matrix: ident, ($NRows: expr, $NCols: expr));* $(;)*) => {$(
        #[test]
        fn $array_matrix_conversion_i_j() {
            let m       = $Matrix::from_fn(|i, j| i * 10 + j);
            let arr: [[usize; $NRows]; $NCols] = m.into();
            let arr_ref = m.as_ref();
            let m2      = $Matrix::from(arr);

            for i in 0 .. $NRows {
                for j in 0 .. $NCols {
                    assert_eq!(arr[j][i], i * 10 + j);
                    assert_eq!(arr_ref[j][i], i * 10 + j);
                }
            }

            assert_eq!(m, m2);
        }
    )*}
);

array_matrix_conversion!(
    array_matrix_conversion_2_2, Matrix2,   (2, 2);
    array_matrix_conversion_2_3, Matrix2x3, (2, 3);
    array_matrix_conversion_2_4, Matrix2x4, (2, 4);
    array_matrix_conversion_2_5, Matrix2x5, (2, 5);
    array_matrix_conversion_2_6, Matrix2x6, (2, 6);

    array_matrix_conversion_3_2, Matrix3x2, (3, 2);
    array_matrix_conversion_3_3, Matrix3,   (3, 3);
    array_matrix_conversion_3_4, Matrix3x4, (3, 4);
    array_matrix_conversion_3_5, Matrix3x5, (3, 5);
    array_matrix_conversion_3_6, Matrix3x6, (3, 6);

    array_matrix_conversion_4_2, Matrix4x2, (4, 2);
    array_matrix_conversion_4_3, Matrix4x3, (4, 3);
    array_matrix_conversion_4_4, Matrix4,   (4, 4);
    array_matrix_conversion_4_5, Matrix4x5, (4, 5);
    array_matrix_conversion_4_6, Matrix4x6, (4, 6);

    array_matrix_conversion_5_2, Matrix5x2, (5, 2);
    array_matrix_conversion_5_3, Matrix5x3, (5, 3);
    array_matrix_conversion_5_4, Matrix5x4, (5, 4);
    array_matrix_conversion_5_5, Matrix5,   (5, 5);
    array_matrix_conversion_5_6, Matrix5x6, (5, 6);

    array_matrix_conversion_6_2, Matrix6x2, (6, 2);
    array_matrix_conversion_6_3, Matrix6x3, (6, 3);
    array_matrix_conversion_6_4, Matrix6x4, (6, 4);
    array_matrix_conversion_6_5, Matrix6x5, (6, 5);
    array_matrix_conversion_6_6, Matrix6,   (6, 6);
);
