use std::num::{Zero, Float};
use traits::operations::Transpose;
use traits::structure::{ColSlice, Eye, Indexable};
use traits::geometry::Norm;
use std::cmp::min;

/// Get the householder matrix corresponding to a reflexion to the hyperplane
/// defined by `vec`. It can be a reflexion contained in a subspace.
///
/// # Arguments
/// * `dim` - the dimension of the space the resulting matrix operates in
/// * `start` - the starting dimension of the subspace of the reflexion
/// * `vec` - the vector defining the reflection.
pub fn householder_matrix<N: Float,
                          M: Eye + Indexable<(uint, uint), N>,
                          V: Indexable<uint, N>>
                          (dim: uint, start: uint, vec: V) -> M {
    let mut qk : M = Eye::new_identity(dim);
    let stop = start + vec.shape();
    assert!(stop <= dim);
    for j in range(start, stop) {
        for i in range(start, stop) {
            unsafe {
                let vv = vec.unsafe_at(i) * vec.unsafe_at(j);
                let qkij = qk.unsafe_at((i, j));
                qk.unsafe_set((i, j), qkij - vv - vv);
            }
        }
    }
    qk
}

/// QR decomposition using Householder reflections.
///
/// # Arguments
/// * `m` - matrix to decompose
pub fn qr<N: Float,
          V: Indexable<uint, N> + Norm<N>,
          M: Clone + Eye + ColSlice<V> + Transpose
              + Indexable<(uint, uint), N> + Mul<M, M>>
          (m: &M) 
          -> (M, M) {
    let (rows, cols) = m.shape();
    assert!(rows >= cols);
    let mut q : M = Eye::new_identity(rows);
    let mut r = m.clone();

    let iterations = min(rows - 1, cols);

    for ite in range(0u, iterations) {
        let mut v = r.col_slice(ite, ite, rows);
        let alpha =
            if unsafe { v.unsafe_at(ite) } >= Zero::zero() {
                -Norm::norm(&v)
            }
            else {
                Norm::norm(&v)
            };
        unsafe {
            let x = v.unsafe_at(0);
            v.unsafe_set(0, x - alpha);
        }
        let _ = v.normalize();
        let qk: M = householder_matrix(rows, 0, v);
        r = qk * r;
        q = q * Transpose::transpose_cpy(&qk);
    }

    (q, r)
}
