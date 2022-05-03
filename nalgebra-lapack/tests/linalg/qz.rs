use na::DMatrix;
use nl::QZ;

use crate::proptest::*;
use proptest::{prop_assert, prop_compose, proptest};

prop_compose! {
    fn f64_dynamic_dim_squares()
        (n in PROPTEST_MATRIX_DIM)
        (a in matrix(PROPTEST_F64,n,n), b in matrix(PROPTEST_F64,n,n)) -> (DMatrix<f64>, DMatrix<f64>){
    (a,b)
}}

proptest! {
    #[test]
    fn qz((a,b) in f64_dynamic_dim_squares()) {

        let qz = QZ::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.clone().unpack();

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7));

    }

    #[test]
    fn qz_static(a in matrix4(), b in matrix4()) {
        let qz = QZ::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.unpack();

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7));
    }
}
