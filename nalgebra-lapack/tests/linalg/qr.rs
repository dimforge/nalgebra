use nl::{QR, qr::QrDecomposition};

use crate::proptest::*;
use proptest::{prop_assert, proptest};

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
}
