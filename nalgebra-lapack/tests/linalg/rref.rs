use na::rref;

use std::cmp;

use na::{DMatrix, DVector, Matrix4x3, Vector4};
use nl::Cholesky;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    pub fn rref_test() {
        let mat: Mat4 = Mat4::Identity();
        let res = rref(&mat);

        assert_eq!(mat, res);

        let m = Matrix3x4::<f64>::new(
             1.0, 2.0, -1.0, -4.0,
             2.0, 3.0, -1.0, -11.0,
            -2.0, 0.0, -3.0,  22.0,
        );

        let expected = Matrix3x4::<f64>::new(
            1.0, 0.0, -1.0, -8.0,
            0.0, 1.0,  0.0, 1.0,
            0.0, 0.0,  1.0, -2.0,
       );

        let res = rref(&mat);
        prop_assert!(relative_eq!(res, expected, epsilon = 1.0e-5));
    }
}
