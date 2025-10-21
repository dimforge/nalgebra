use super::test_util::*;
use crate::proptest::*;
use core::panic;
use na::{DMatrix, OMatrix};
use nl::{ColPivQR, qr::QrDecomposition};
use proptest::prelude::*;

proptest! {
    #[test]
    fn colpiv_qr_basics(mut a  in square_or_overdetermined_dmatrix()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
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
        // by checking A = Q R P^-1
        qr.p().inv_permute_cols_mut(&mut a).unwrap();
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn colpiv_qr_basics_static(mut a  in matrix5x3()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        prop_assert!(is_upper_triangular(&r));

        // this tests Q^T Q = Id
        let qtq = q.transpose()*&q;
        let (nrows, ncols) = qtq.shape_generic();
        let eye = OMatrix::identity_generic(nrows,ncols);
        prop_assert!(relative_eq!(qtq, eye, epsilon = 1.0e-7));

        // this tests A P = Q R
        // by checking A = Q R P^-1
        qr.p().inv_permute_cols_mut(&mut a).unwrap();
        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7));
    }

    #[test]
    fn colpiv_qr_solve_static(a in matrix5x3(), b in matrix5x2()) {
        let rank = a.rank(f64::EPSILON.sqrt());
        let qr = ColPivQR::new(a.clone()).unwrap();
        let result = qr.solve(b.clone());
        match result {
            Ok(x) => {
                //@note(geo-ant) We can't just test A*x = b, since that is
                // not what the QR solution guarantees. It guarantees a minimum
                // residual solution, which means the normal equations must
                // hold. That means we can assert on A^T A * X = A^T *B
                prop_assert!(relative_eq!(a.transpose()*a*x,a.transpose()*b,epsilon = 1e-5));
            },
            Err(err) => {
                prop_assert!(rank == 0);
                prop_assert!(err == nl::colpiv_qr::Error::ZeroRank);
            },
        }
    }

    #[test]
    fn colpiv_qr_solve((a,b) in linear_system_dynamic()) {
        let rank = a.rank(f64::EPSILON.sqrt());
        let qr = ColPivQR::new(a.clone()).unwrap();
        let result = qr.solve(b.clone());
        match result {
            Ok(x) => {
                //@note(geo-ant) We can't just test A*x = b, since that is
                // not what the QR solution guarantees. It guarantees a minimum
                // residual solution, which means the normal equations must
                // hold. That means we can assert on A^T A * X = A^T *B
                prop_assert!(relative_eq!(a.transpose()*&a*x,a.transpose()*b,epsilon = 1e-5));
            },
            Err(err) => {
                prop_assert!(rank == 0);
                prop_assert!(err == nl::colpiv_qr::Error::ZeroRank);
            },
        }
    }

    #[test]
    fn r_multiplication(a in square_or_overdetermined_dmatrix()) {
        // we use the identity matrix for multiplication to check if
        // this results in the correct qr.r(). We have already verified qr.r()
        // above.
        let qr = ColPivQR::new(a).unwrap();

        let mut r = DMatrix::identity(qr.ncols(),qr.ncols());
        qr.r_mul_mut(&mut r).unwrap();
        prop_assert!(relative_eq!(r,qr.r(),epsilon = 1e-5));

        let mut rt = DMatrix::identity(qr.ncols(),qr.ncols());
        qr.r_tr_mul_mut(&mut rt).unwrap();
        prop_assert!(relative_eq!(rt,qr.r().transpose(),epsilon = 1e-5));

        let mut r = DMatrix::identity(qr.ncols(),qr.ncols());
        qr.mul_r_mut(&mut r).unwrap();
        prop_assert!(relative_eq!(r,qr.r(),epsilon = 1e-5));

        let mut rt = DMatrix::identity(qr.ncols(),qr.ncols());
        qr.mul_r_tr_mut(&mut rt).unwrap();
        prop_assert!(relative_eq!(rt,qr.r().transpose(),epsilon = 1e-5));
    }

    #[test]
    fn q_multiplication(a in square_or_overdetermined_dmatrix()) {
        // same trick as in r-multiplication test. Use rectangular identity matrix
        // for multiplication and compare with qr.q(), which is computed
        // differently and verified above.
        let qr = ColPivQR::new(a).unwrap();

        let mut q = DMatrix::identity(qr.nrows(),qr.ncols());
        qr.q_mul_mut(&mut q).unwrap();
        prop_assert!(relative_eq!(q,qr.q(),epsilon = 1e-5));

        // this tests orthogonality Q^T Q = I
        qr.q_tr_mul_mut(&mut q).unwrap();
        prop_assert!(relative_eq!(q,DMatrix::<f64>::identity(q.nrows(),q.ncols()),epsilon = 1e-5));

        // now since we've verified Q itself, we now use it to verify
        // the other multiplications by generating rectangular matrices m x m

        // I*Q
        let mut q = DMatrix::identity(qr.nrows(),qr.nrows());
        qr.mul_q_mut(&mut q).unwrap();

        // I * Q^T
        let mut qt = DMatrix::identity(qr.nrows(),qr.nrows());
        qr.mul_q_tr_mut(&mut qt).unwrap();

        // Q * I
        let mut q2 = DMatrix::identity(qr.nrows(),qr.nrows());
        qr.q_mul_mut(&mut q2).unwrap();

        // Q^T * I
        let mut qt2 = DMatrix::identity(qr.nrows(),qr.nrows());
        qr.q_tr_mul_mut(&mut qt2).unwrap();

        let eye = DMatrix::identity(qr.nrows(),qr.nrows());

        prop_assert!(relative_eq!(q,q2, epsilon = 1e-5));
        prop_assert!(relative_eq!(qt,qt2, epsilon = 1e-5));
        prop_assert!(relative_eq!(&qt*&q, eye, epsilon = 1e-5));
        prop_assert!(relative_eq!(&q*&qt, eye, epsilon = 1e-5));
    }
}
