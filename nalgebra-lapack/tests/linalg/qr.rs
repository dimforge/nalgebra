use crate::proptest::*;
use na::DMatrix;
use nl::{QR, qr::QrDecomposition};
use proptest::prelude::*;
use proptest::{prop_assert, proptest};

/// gives us a matrix A for QR decomposition and a matrix B where R*B can be calculated
fn dmatrix_and_r_multipliable() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.ncols(), PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

proptest! {
    #[test]
    fn qr(m in dmatrix()) {
        let qr = QR::new(m.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        prop_assert!(relative_eq!(m, q * r, epsilon = 1.0e-7))
    }

    #[test]
    fn qr_static(m in matrix5x3()) {
        let qr = QR::new(m).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        prop_assert!(relative_eq!(m, q * r, epsilon = 1.0e-7))
    }

    // #[test]
    // fn r_mul_mut((a, mut x) in dmatrix_and_r_multipliable()) {
    //     let qr = QR::new(a).unwrap();
    //     let rx  = qr.r()*&x;
    //     qr.r_mul_mut(&mut x).unwrap();
    //     prop_assert!(relative_eq!(rx,x,epsilon = 1e-5));
    // }
}
