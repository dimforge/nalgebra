use crate::proptest::*;
use na::{DMatrix, Dim, Matrix, OMatrix, RawStorage};
use nl::ColPivQR;
use num_traits::Zero;
use proptest::prelude::*;

fn is_upper_triangular<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> bool
where
    T: Zero + PartialEq,
    C: Dim,
    R: Dim,
    S: RawStorage<T, R, C>,
{
    let ncols = mat.ncols();
    let nrows = mat.nrows();

    let zero = T::zero();
    for c in 0..ncols {
        for r in c + 1..nrows {
            if mat[(r, c)] != zero {
                return false;
            }
        }
    }
    return true;
}

fn square_or_overdetermined_dmatrix() -> impl Strategy<Value = DMatrix<f64>> {
    PROPTEST_MATRIX_DIM.prop_flat_map(|rows| {
        (1..=rows).prop_flat_map(move |cols| matrix(PROPTEST_F64, rows..=rows, cols..=cols))
    })
}

fn linear_system_dynamic() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    square_or_overdetermined_dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.nrows(), PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

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
        let result = qr.solve(&b);
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
                prop_assert!(err == nl::ColPivQrError::ZeroRank);
            },
        }
    }

    #[test]
    fn colpiv_qr_solve((a,b) in linear_system_dynamic()) {
        let rank = a.rank(f64::EPSILON.sqrt());
        let qr = ColPivQR::new(a.clone()).unwrap();
        let result = qr.solve(&b);
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
                prop_assert!(err == nl::ColPivQrError::ZeroRank);
            },
        }
    }
}
