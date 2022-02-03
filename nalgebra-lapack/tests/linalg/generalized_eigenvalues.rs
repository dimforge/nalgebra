use na::dimension::{Const, Dynamic};
use na::{DMatrix, EuclideanNorm, Norm, OMatrix};
use nl::GE;
use num_complex::Complex;
use simba::scalar::ComplexField;
use std::cmp;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn ge(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let a = DMatrix::<f64>::new_random(n, n);
        let b = DMatrix::<f64>::new_random(n, n);
        let a_condition_no = a.clone().try_inverse().and_then(|x| Some(EuclideanNorm.norm(&x)* EuclideanNorm.norm(&a)));
        let b_condition_no = b.clone().try_inverse().and_then(|x| Some(EuclideanNorm.norm(&x)* EuclideanNorm.norm(&b)));

        if a_condition_no.unwrap_or(200000.0) < 5.0 && b_condition_no.unwrap_or(200000.0) < 5.0 {
            let a_c = a.clone().map(|x| Complex::new(x, 0.0));
            let b_c = b.clone().map(|x| Complex::new(x, 0.0));

            let ge = GE::new(a.clone(), b.clone());
            let (vsl,vsr) = ge.clone().eigenvectors();

            for (i,(alpha,beta)) in ge.raw_eigenvalues().iter().enumerate() {
                let l_a = a_c.clone() * Complex::new(*beta, 0.0);
                let l_b = b_c.clone() * *alpha;

                prop_assert!(
                    relative_eq!(
                        ((&l_a - &l_b)*vsr.column(i)).map(|x| x.modulus()),
                        OMatrix::zeros_generic(Dynamic::new(n), Const::<1>),
                        epsilon = 1.0e-7));

                prop_assert!(
                    relative_eq!(
                        (vsl.column(i).adjoint()*(&l_a - &l_b)).map(|x| x.modulus()),
                        OMatrix::zeros_generic(Const::<1>, Dynamic::new(n)),
                        epsilon = 1.0e-7))
            };
        };
    }

    #[test]
    fn ge_static(a in matrix4(), b in matrix4()) {
        let a_condition_no = a.clone().try_inverse().and_then(|x| Some(EuclideanNorm.norm(&x)* EuclideanNorm.norm(&a)));
        let b_condition_no = b.clone().try_inverse().and_then(|x| Some(EuclideanNorm.norm(&x)* EuclideanNorm.norm(&b)));

        if a_condition_no.unwrap_or(200000.0) < 5.0 && b_condition_no.unwrap_or(200000.0) < 5.0 {
            let ge = GE::new(a.clone(), b.clone());
            let a_c =a.clone().map(|x| Complex::new(x, 0.0));
            let b_c = b.clone().map(|x| Complex::new(x, 0.0));
            let (vsl,vsr) = ge.eigenvectors();
            let eigenvalues = ge.raw_eigenvalues();

            for (i,(alpha,beta)) in eigenvalues.iter().enumerate() {
                let l_a = a_c.clone() * Complex::new(*beta, 0.0);
                let l_b = b_c.clone() * *alpha;

                prop_assert!(
                    relative_eq!(
                        ((&l_a - &l_b)*vsr.column(i)).map(|x| x.modulus()),
                        OMatrix::zeros_generic(Const::<4>, Const::<1>),
                        epsilon = 1.0e-7));
                prop_assert!(
                    relative_eq!((vsl.column(i).adjoint()*(&l_a - &l_b)).map(|x| x.modulus()),
                                 OMatrix::zeros_generic(Const::<1>, Const::<4>),
                                 epsilon = 1.0e-7))
            }
        };
    }
}
