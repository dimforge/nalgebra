use na::dimension::Const;
use na::{DMatrix, OMatrix};
use nl::GeneralizedEigen;
use num_complex::Complex;
use simba::scalar::ComplexField;

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
    fn ge((a,b) in f64_dynamic_dim_squares()){

        let a_c = a.clone().map(|x| Complex::new(x, 0.0));
        let b_c = b.clone().map(|x| Complex::new(x, 0.0));
        let n = a.shape_generic().0;

        let ge = GeneralizedEigen::new(a.clone(), b.clone());
        let (vsl,vsr) = ge.clone().eigenvectors();


        for (i,(alpha,beta)) in ge.raw_eigenvalues().iter().enumerate() {
            let l_a = a_c.clone() * Complex::new(*beta, 0.0);
            let l_b = b_c.clone() * *alpha;

            prop_assert!(
                relative_eq!(
                    ((&l_a - &l_b)*vsr.column(i)).map(|x| x.modulus()),
                    OMatrix::zeros_generic(n, Const::<1>),
                    epsilon = 1.0e-5));

            prop_assert!(
                relative_eq!(
                    (vsl.column(i).adjoint()*(&l_a - &l_b)).map(|x| x.modulus()),
                    OMatrix::zeros_generic(Const::<1>, n),
                    epsilon = 1.0e-5))
        };
    }

    #[test]
    fn ge_static(a in matrix4(), b in matrix4()) {

        let ge = GeneralizedEigen::new(a.clone(), b.clone());
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
                    epsilon = 1.0e-5));
            prop_assert!(
                relative_eq!((vsl.column(i).adjoint()*(&l_a - &l_b)).map(|x| x.modulus()),
                             OMatrix::zeros_generic(Const::<1>, Const::<4>),
                             epsilon = 1.0e-5))
        }
    }

}
