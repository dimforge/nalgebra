use crate::{Matrix3, SVD, U3};
use simba::scalar::RealField;

// For the 3x3 case, on the GPU, it is much more efficient to compute the SVD
// using an eigendecomposition followed by a QR decomposition.
pub fn svd3<T: RealField>(
    m: &Matrix3<T>,
    compute_u: bool,
    compute_v: bool,
    eps: T,
    niter: usize,
) -> Option<SVD<T, U3, U3>> {
    let s = m.tr_mul(&m);
    let v = s.try_symmetric_eigen(eps, niter)?.eigenvectors;
    let b = m * &v;

    let qr = b.qr();
    let singular_values = qr.diag_internal().map(|e| e.abs());

    Some(SVD {
        u: if compute_u { Some(qr.q()) } else { None },
        singular_values,
        v_t: if compute_v { Some(v.transpose()) } else { None },
    })
}
