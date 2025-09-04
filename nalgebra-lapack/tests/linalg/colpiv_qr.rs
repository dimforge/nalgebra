use crate::proptest::*;
use nl::ColPivQR;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn colpiv_qr_decomposition(mut a  in square_or_overdetermined_dmatrix()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        qr.p().permute_cols_mut(&mut a).unwrap();

        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7))
    }

    #[test]
    fn colpiv_qr_decomposition_static(mut a  in matrix5x3()) {
        let qr = ColPivQR::new(a.clone()).unwrap();
        let q  = qr.q();
        let r  = qr.r();

        qr.p().permute_cols_mut(&mut a).unwrap();

        prop_assert!(relative_eq!(a, q * r, epsilon = 1.0e-7))
        }
}
