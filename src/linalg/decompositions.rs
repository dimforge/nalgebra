use std::num::{Zero, Float};
use na::DVec;
use na::DMat;
use traits::operations::{Transpose, ColSlice};
use traits::geometry::Norm;
use std::cmp::min;

/// QR decomposition using Householder reflections
/// # Arguments
/// * `m` matrix to decompose
pub fn decomp_qr<N: Clone + Num + Float>(m: &DMat<N>) -> (DMat<N>, DMat<N>) {
    let rows = m.nrows();
    let cols = m.ncols();
    assert!(rows >= cols);
    let mut q : DMat<N> = DMat::new_identity(rows);
    let mut r = m.clone();

    let subtract_reflection = |vec: DVec<N>| -> DMat<N> {
        // FIXME: we don't handle the complex case here
        let mut qk : DMat<N> = DMat::new_identity(rows);
        let start = rows - vec.at.len();
        for j in range(start, rows) {
            for i in range(start, rows) {
                unsafe {
                    let vv = vec.at_fast(i-start)*vec.at_fast(j-start);
                    let qkij = qk.at_fast(i,j);
                    qk.set_fast(i, j, qkij - vv - vv);
                }
            }
        }
        qk
    };

    let iterations = min(rows-1, cols);

    for ite in range(0u, iterations) {
        let mut v = r.col_slice(ite, ite, rows);
        //let mut v = r.col_slice<DVec<N>>(ite, rows-ite, rows);
        let alpha =
            if unsafe { v.at_fast(ite) } >= Zero::zero() {
                -Norm::norm(&v)
            }
            else {
                Norm::norm(&v)
            };
        unsafe {
            let x = v.at_fast(0);
            v.set_fast(0, x - alpha);
        }
        let _ = v.normalize();
        let qk = subtract_reflection(v);
        r = qk * r;
        q = q * Transpose::transpose_cpy(&qk);
    }

    (q, r)
}

