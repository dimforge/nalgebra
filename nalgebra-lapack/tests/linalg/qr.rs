use na::{DMatrix, Matrix4x3};
use nl::QR;

quickcheck!{
    fn qr(m: DMatrix<f64>) -> bool {
        let qr = QR::new(m.clone());
        let q  = qr.q();
        let r  = qr.r();

        relative_eq!(m, q * r, epsilon = 1.0e-7)
    }

    fn qr_static(m: Matrix4x3<f64>) -> bool {
        let qr = QR::new(m);
        let q  = qr.q();
        let r  = qr.r();

        relative_eq!(m, q * r, epsilon = 1.0e-7)
    }
}
