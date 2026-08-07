use crate::{Matrix3, SVD, U3};
use simba::scalar::RealField;

// For the 3x3 case, on the GPU, it is much more efficient to compute the SVD
// using an eigendecomposition followed by a QR decomposition.
//
// This is based on the paper "Computing the Singular Value Decomposition of 3 x 3 matrices with
// minimal branching and elementary floating point operations" from McAdams, et al.
pub fn svd_ordered3<T: RealField>(
    m: &Matrix3<T>,
    compute_u: bool,
    compute_v: bool,
    eps: T,
    niter: usize,
) -> Option<SVD<T, U3, U3>> {
    let s = m.tr_mul(m);
    let mut v = s.try_symmetric_eigen(eps, niter)?.eigenvectors;
    let mut b = m * &v;

    // Sort singular values. This is a necessary step to ensure that
    // the QR decompositions R matrix ends up diagonal.
    let mut rho0 = b.column(0).norm_squared();
    let mut rho1 = b.column(1).norm_squared();
    let mut rho2 = b.column(2).norm_squared();

    if rho0 < rho1 {
        b.swap_columns(0, 1);
        b.column_mut(1).neg_mut();
        v.swap_columns(0, 1);
        v.column_mut(1).neg_mut();
        std::mem::swap(&mut rho0, &mut rho1);
    }
    if rho0 < rho2 {
        b.swap_columns(0, 2);
        b.column_mut(2).neg_mut();
        v.swap_columns(0, 2);
        v.column_mut(2).neg_mut();
        std::mem::swap(&mut rho0, &mut rho2);
    }
    if rho1 < rho2 {
        b.swap_columns(1, 2);
        b.column_mut(2).neg_mut();
        v.swap_columns(1, 2);
        v.column_mut(2).neg_mut();
        std::mem::swap(&mut rho0, &mut rho2);
    }

    let qr = b.qr();
    // The QR decomposition does not force the diagonal of `R` to be positive, so the singular
    // values are the absolute values of that diagonal, and the columns of `U` must get the
    // sign of the corresponding diagonal element.
    let r_diagonal = qr.qr_internal().diagonal();

    Some(SVD {
        u: if compute_u {
            let mut u = qr.q();

            for i in 0..3 {
                if r_diagonal[i] < T::zero() {
                    u.column_mut(i).neg_mut();
                }
            }

            Some(u)
        } else {
            None
        },
        singular_values: r_diagonal.map(|e| e.abs()),
        v_t: if compute_v { Some(v.transpose()) } else { None },
    })
}
