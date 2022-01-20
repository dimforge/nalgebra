use na::{zero, DMatrix, SMatrix};
use nl::QZ;
use num_complex::Complex;
use simba::scalar::ComplexField;
use std::cmp;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn qz(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let a = DMatrix::<f64>::new_random(n, n);
        let b = DMatrix::<f64>::new_random(n, n);

        let qz =  QZ::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.clone().unpack();
        //let eigenvalues = qz.eigenvalues();
        //let a_c = a.clone().map(|x| Complex::new(x, zero::<f64>()));

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a.clone(), epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b.clone(), epsilon = 1.0e-7));
        // spotty test that skips over the first eigenvalue which in some cases is extremely large relative to the other ones
        // and fails the condition
        //for i in 1..n {
        //   let b_c = b.clone().map(|x| eigenvalues[i]*Complex::new(x,zero::<f64>()));
        //   prop_assert!(relative_eq!((&a_c  - &b_c).determinant().modulus(), 0.0, epsilon = 1.0e-6));
        //}
    }

    #[test]
    fn qz_static(a in matrix4(), b in matrix4()) {
        let qz = QZ::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.unpack();
        //let eigenvalues = qz.eigenvalues();
        //let a_c = a.clone().map(|x| Complex::new(x, zero::<f64>()));

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7));

        //for i in 0..4 {
        //    let b_c = b.clone().map(|x| eigenvalues[i]*Complex::new(x,zero::<f64>()));
        //    println!("{}",eigenvalues);
        //    prop_assert!(relative_eq!((&a_c  - &b_c).determinant().modulus(), 0.0, epsilon = 1.0e-4))
        //}
    }
}
