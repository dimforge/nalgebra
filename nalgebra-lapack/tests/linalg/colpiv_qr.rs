use crate::proptest::*;
use na::{DMatrix, Dyn, Matrix3x2, OMatrix};
use nl::ColPivQR;
use proptest::prelude::*;

proptest! {
    #[test]
    fn colpiv_qr_basics(mut a  in square_or_overdetermined_dmatrix()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        // this tests Q^T Q = Id
        // note that Q*Q^T is typically not the identity matrix
        // since Q is the economy QR decomposition and
        // this calculates orthonormal Q in R^(m x n)
        let qtq = q.transpose()*&q;
        let eye = DMatrix::identity(qtq.nrows(),qtq.ncols());
        prop_assert!(relative_eq!(qtq, eye, epsilon = 1.0e-7));
        // prop_assert!
        // this tests A P = Q R
        qr.p().permute_cols_mut(&mut a).unwrap();
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn colpiv_qr_basics_static(mut a  in matrix5x3()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        // this tests Q^T Q = Id
        let qtq = q.transpose()*&q;
        let (nrows, ncols) = qtq.shape_generic();
        let eye = OMatrix::identity_generic(nrows,ncols);
        prop_assert!(relative_eq!(qtq, eye, epsilon = 1.0e-7));

        // this tests A P = Q R
        qr.p().permute_cols_mut(&mut a).unwrap();
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn colpiv_qr_solve_static(a in matrix5x3(), b in vector5()) {
        let rank = a.rank(f64::EPSILON.sqrt());
        let qr = ColPivQR::new(a.clone()).unwrap();
        let result = qr.solve(&b);
        match result {
            Ok(x) => {
                let req=relative_eq!(a.transpose()*a*x,a.transpose()*b,epsilon = 1e-5);
                if !req {
                    println!("a = {:?}",a);
                    println!("x = {:?}",x);
                    println!("b = {:?}",b);
                    println!("a*x = {:?}",a*x);
                }
                prop_assert!(req);
            },
            Err(err) => {
                prop_assert!(rank == 0);
                prop_assert!(err == nl::ColPivQrError::ZeroRank);
            },
        }
    }
}
