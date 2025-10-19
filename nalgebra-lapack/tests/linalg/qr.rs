use super::test_util::*;
use crate::proptest::*;
use na::{DMatrix, OMatrix};
use nl::{QR, qr::QrDecomposition};
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn qr_basics(a  in square_or_overdetermined_dmatrix()) {
        let qr = QR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        prop_assert!(is_upper_triangular(&r));

        // this tests Q^T Q = Id
        // @note(geo-ant) that Q*Q^T is typically not the identity matrix
        // since Q is the economy QR decomposition and
        // this calculates orthonormal Q in R^(m x n)
        let qtq = q.transpose()*&q;
        let eye = DMatrix::identity(qtq.nrows(),qtq.ncols());
        prop_assert!(relative_eq!(qtq, eye, epsilon = 1.0e-7));
        // this tests A P = Q R
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn qr_basics_static(a  in matrix5x3()) {
        let qr = QR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        prop_assert!(is_upper_triangular(&r));

        // this tests Q^T Q = Id
        let qtq = q.transpose()*&q;
        let (nrows, ncols) = qtq.shape_generic();
        let eye = OMatrix::identity_generic(nrows,ncols);
        prop_assert!(relative_eq!(qtq, eye, epsilon = 1.0e-7));

        // this tests A P = Q R
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn qr_solve_static(a in full_rank_matrix5x3(), b in matrix5x2()) {
        let qr = QR::new(a.clone()).unwrap();
        let x = qr.solve(b.clone()).unwrap();
        prop_assert!(relative_eq!(a.transpose()*a*x,a.transpose()*b,epsilon = 1e-5));
    }

    #[test]
    fn qr_solve((a,b) in full_rank_linear_system_dynamic()) {
        let qr = QR::new(a.clone()).unwrap();
        let x = qr.solve(b.clone()).unwrap();
        //@note(geo-ant) We can't just test A*x = b, since that is
        // not what the QR solution guarantees. It guarantees a minimum
        // residual solution, which means the normal equations must
        // hold. That means we can assert on A^T A * X = A^T *B
        prop_assert!(relative_eq!(a.transpose()*&a*x,a.transpose()*b,epsilon = 1e-5));
    }

    #[test]
    fn r_mul_mut((a, mut x) in square_or_overdetermined_mat_and_r_multipliable()) {
        let qr = QR::new(a).unwrap();
        let rx  = qr.r()*&x;
        qr.r_mul_mut(&mut x).unwrap();
        prop_assert!(relative_eq!(rx,x,epsilon = 1e-5));
    }
}
