use na::{DMatrix, Matrix3x4};
use nl::SVD;

quickcheck! {
    fn svd(m: DMatrix<f64>) -> bool {
        if m.nrows() != 0 && m.ncols() != 0 {
            let svd = SVD::new(m.clone()).unwrap();
            let sm  = DMatrix::from_partial_diagonal(m.nrows(), m.ncols(), svd.singular_values.as_slice());

            let reconstructed_m = &svd.u * sm * &svd.vt;
            let reconstructed_m2 = svd.recompose();

            relative_eq!(reconstructed_m, m, epsilon = 1.0e-7) &&
            relative_eq!(reconstructed_m2, reconstructed_m, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn svd_static(m: Matrix3x4<f64>) -> bool {
        let svd = SVD::new(m).unwrap();
        let sm  = Matrix3x4::from_partial_diagonal(svd.singular_values.as_slice());

        let reconstructed_m  = &svd.u * &sm * &svd.vt;
        let reconstructed_m2 = svd.recompose();

        relative_eq!(reconstructed_m, m, epsilon = 1.0e-7) &&
        relative_eq!(reconstructed_m2, m, epsilon = 1.0e-7)
    }

    fn pseudo_inverse(m: DMatrix<f64>) -> bool {
        if m.nrows() == 0 || m.ncols() == 0 {
            return true;
        }

        let svd = SVD::new(m.clone()).unwrap();
        let im  = svd.pseudo_inverse(1.0e-7);

        if m.nrows() <= m.ncols() {
            return (&m * &im).is_identity(1.0e-7)
        }

        if m.nrows() >= m.ncols() {
            return (im * m).is_identity(1.0e-7)
        }

        return true;
    }

    fn pseudo_inverse_static(m: Matrix3x4<f64>) -> bool {
        let svd = SVD::new(m).unwrap();
        let im  = svd.pseudo_inverse(1.0e-7);

        (m * im).is_identity(1.0e-7)
    }
}
