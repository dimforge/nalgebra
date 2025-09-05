use crate::proptest::*;
use na::{DMatrix, OMatrix};
use nl::ColPivQR;
use proptest::{prop_assert, proptest};

proptest! {

    #[test]
    fn colpiv_qr_decomposition(mut a  in square_or_overdetermined_dmatrix()) {
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
    fn colpiv_qr_decomposition_static(mut a  in matrix5x3()) {
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
}
