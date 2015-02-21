use traits::operations::{Transpose, ApproxEq};
use traits::structure::{ColSlice, Eye, Indexable, Diag, SquareMat, BaseFloat};
use traits::geometry::Norm;
use std::cmp::min;
use std::ops::{Mul, Add, Sub};

/// Get the householder matrix corresponding to a reflexion to the hyperplane
/// defined by `vec`. It can be a reflexion contained in a subspace.
///
/// # Arguments
/// * `dim` - the dimension of the space the resulting matrix operates in
/// * `start` - the starting dimension of the subspace of the reflexion
/// * `vec` - the vector defining the reflection.
pub fn householder_matrix<N, V, M>(dim: usize, start: usize, vec: V) -> M
    where N: BaseFloat,
          M: Eye + Indexable<(usize, usize), N>,
          V: Indexable<usize, N> {
    let mut qk : M = Eye::new_identity(dim);
    let subdim = vec.shape();

    let stop = subdim + start;

    assert!(dim >= stop);

    for j in (start .. stop) {
        for i in (start .. stop) {
            unsafe {
                let vv = vec.unsafe_at(i - start) * vec.unsafe_at(j - start);
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
pub fn qr<N, V, M>(m: &M) -> (M, M)
    where N: BaseFloat,
          V: Indexable<usize, N> + Norm<N>,
          M: Copy + Eye + ColSlice<V> + Transpose + Indexable<(usize, usize), N> +
             Mul<M, Output = M> {
    let (rows, cols) = m.shape();
    assert!(rows >= cols);
    let mut q : M = Eye::new_identity(rows);
    let mut r = *m;

    for ite in 0..min(rows - 1, cols) {
        let mut v = r.col_slice(ite, ite, rows);
        let alpha =
            if unsafe { v.unsafe_at(ite) } >= ::zero() {
                -Norm::norm(&v)
            }
            else {
                Norm::norm(&v)
            };
        unsafe {
            let x = v.unsafe_at(0);
            v.unsafe_set(0, x - alpha);
        }
        if !::is_zero(&v.normalize_mut()) {
            let qk: M = householder_matrix(rows, ite, v);
            r = qk * r;
            q = q * Transpose::transpose(&qk);
        }
    }

    (q, r)
}

/// Eigendecomposition of a square matrix using the qr algorithm.
pub fn eigen_qr<N, V, VS, M>(m: &M, eps: &N, niter: usize) -> (M, V)
    where N:  BaseFloat,
          VS: Indexable<usize, N> + Norm<N>,
          M:  Indexable<(usize, usize), N> + SquareMat<N, V> + Add<M, Output = M> +
              Sub<M, Output = M> + ColSlice<VS> +
              ApproxEq<N> + Copy {
    let mut eigenvectors: M = ::one::<M>();
    let mut eigenvalues = *m;
    // let mut shifter: M = Eye::new_identity(rows);

    let mut iter = 0;
    for _ in 0..niter {
        let mut stop = true;

        for j in 0..::dim::<M>() {
            for i in 0..j {
                if unsafe { eigenvalues.unsafe_at((i, j)) }.abs() >= *eps {
                    stop = false;
                    break;
                }
            }

            for i in j + 1..::dim::<M>() {
                if unsafe { eigenvalues.unsafe_at((i, j)) }.abs() >= *eps {
                    stop = false;
                    break;
                }
            }
        }

        if stop {
            break;
        }
        iter = iter + 1;

        let (q, r) = qr(&eigenvalues);;

        eigenvalues  = r * q;
        eigenvectors = eigenvectors * q;
    }

    (eigenvectors, eigenvalues.diag())
}
