use std::cmp;

use alga::general::Real;
use core::{SquareMatrix, OwnedSquareMatrix, ColumnVector, OwnedColumnVector};
use dimension::{Dim, Dynamic, U1};
use storage::{Storage, OwnedStorage};
use allocator::{Allocator, OwnedAllocator};



impl<N, D: Dim, S> SquareMatrix<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    /// Get the householder matrix corresponding to a reflexion to the hyperplane
    /// defined by `vector`. It can be a reflexion contained in a subspace.
    ///
    /// # Arguments
    /// * `dimension` - the dimension of the space the resulting matrix operates in
    /// * `start` - the starting dimension of the subspace of the reflexion
    /// * `vector` - the vector defining the reflection.
    pub fn new_householder_generic<SB, D2>(dimension: D, start: usize, vector: &ColumnVector<N, D2, SB>)
        -> OwnedSquareMatrix<N, D, S::Alloc>
        where D2: Dim,
              SB: Storage<N, D2, U1> {
        let mut qk = Self::identity_generic(dimension, dimension);
        let subdim = vector.shape().0;

        let stop = subdim + start;

        assert!(dimension.value() >= stop, "Householder matrix creation: subspace dimension index out of bounds.");

        for j in start .. stop {
            for i in start .. stop {
                unsafe {
                    let vv   = *vector.get_unchecked(i - start, 0) * *vector.get_unchecked(j - start, 0);
                    let qkij = *qk.get_unchecked(i, j);
                    *qk.get_unchecked_mut(i, j) = qkij - vv - vv;
                }
            }
        }

        qk
    }
}


impl<N: Real, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// QR decomposition using Householder reflections.
    pub fn qr(self) -> (OwnedSquareMatrix<N, D, S::Alloc>, OwnedSquareMatrix<N, D, S::Alloc>)
        where S::Alloc: Allocator<N, Dynamic, U1> +
                        Allocator<N, D, U1> {

        let (nrows, ncols) = self.data.shape();

        // XXX: too restrictive.
        assert!(nrows.value() >= ncols.value(), "");

        let mut q = OwnedSquareMatrix::<N, D, S::Alloc>::identity_generic(nrows, ncols);
        let mut r = self.into_owned();

        // Temporary buffer that contains a column.
        let mut col = unsafe {
            OwnedColumnVector::<N, D, S::Alloc>::new_uninitialized_generic(nrows, U1)
        };

        for ite in 0 .. cmp::min(nrows.value() - 1, ncols.value()) {
            let subdim = Dynamic::new(nrows.value() - ite);
            let mut v  = col.rows_mut(0, subdim.value());
            v.copy_from(&r.generic_slice((ite, ite), (subdim, U1)));

            let alpha =
                if unsafe { *v.get_unchecked(ite, 0) } >= ::zero() {
                    -v.norm()
                }
                else {
                    v.norm()
                };

            unsafe {
                let x = *v.get_unchecked(0, 0);
                *v.get_unchecked_mut(0, 0) = x - alpha;
            }

            if !v.normalize_mut().is_zero() {
                let mut qk = OwnedSquareMatrix::<N, D, S::Alloc>::new_householder_generic(nrows, ite, &v);
                r = &qk * r;

                // FIXME: add a method `q.mul_tr(qk) := q * qk.transpose` ?
                qk.transpose_mut();
                q = q * qk;
            }
        }

        (q, r)
    }

    /// Eigendecomposition of a square symmetric matrix.
    pub fn eig(&self, eps: N, niter: usize)
               -> (OwnedSquareMatrix<N, D, S::Alloc>, OwnedColumnVector<N, D, S::Alloc>)
        where S::Alloc: Allocator<N, D, U1> +
                        Allocator<N, Dynamic, U1> {

        assert!(self.is_square(),
                "Unable to compute the eigenvectors and eigenvalues of a non-square matrix.");

        let dim = self.data.shape().0;

        let (mut eigenvectors, mut eigenvalues) = self.hessenberg();

        if dim.value() == 1 {
            return (eigenvectors, eigenvalues.diagonal());
        }

        // Allocate arrays for Givens rotation components
        let mut c = unsafe { OwnedColumnVector::<N, D, S::Alloc>::new_uninitialized_generic(dim, U1) };
        let mut s = unsafe { OwnedColumnVector::<N, D, S::Alloc>::new_uninitialized_generic(dim, U1) };

        let mut iter = 0;
        let mut curdim = dim.value() - 1;

        for _ in 0 .. dim.value() {

            let mut stop = false;

            while !stop && iter < niter {

                let lambda;

                unsafe {
                    let a = *eigenvalues.get_unchecked(curdim - 1, curdim - 1);
                    let b = *eigenvalues.get_unchecked(curdim - 1, curdim);
                    let c = *eigenvalues.get_unchecked(curdim, curdim - 1);
                    let d = *eigenvalues.get_unchecked(curdim, curdim);

                    let trace = a + d;
                    let determinant = a * d - b * c;

                    let constquarter: N = ::convert(0.25f64);
                    let consthalf:    N = ::convert(0.5f64);

                    let e = (constquarter * trace * trace - determinant).sqrt();

                    let lambda1 = consthalf * trace + e;
                    let lambda2 = consthalf * trace - e;

                    if (lambda1 - d).abs() < (lambda2 - d).abs() {
                        lambda = lambda1;
                    }
                    else {
                        lambda = lambda2;
                    }

                }

                // Shift matrix
                for k in 0 .. curdim + 1 {
                    unsafe {
                        let a = *eigenvalues.get_unchecked(k, k);
                        *eigenvalues.get_unchecked_mut(k, k) = a - lambda;
                    }
                }


                // Givens rotation from left
                for k in 0 .. curdim {
                    let x_i = unsafe { *eigenvalues.get_unchecked(k, k) };
                    let x_j = unsafe { *eigenvalues.get_unchecked(k + 1, k) };

                    let ctmp;
                    let stmp;

                    if x_j.abs() < eps {
                        ctmp = N::one();
                        stmp = N::zero();
                    }
                    else if x_i.abs() < eps {
                        ctmp = N::zero();
                        stmp = -N::one();
                    }
                    else {
                        let r = x_i.hypot(x_j);
                        ctmp = x_i / r;
                        stmp = -x_j / r;
                    }

                    c[k] = ctmp;
                    s[k] = stmp;

                    for j in k .. (curdim + 1) {
                        unsafe {
                            let a = *eigenvalues.get_unchecked(k, j);
                            let b = *eigenvalues.get_unchecked(k + 1, j);

                            *eigenvalues.get_unchecked_mut(k, j)     = ctmp * a - stmp * b;
                            *eigenvalues.get_unchecked_mut(k + 1, j) = stmp * a + ctmp * b;
                        }

                    }
                }

                // Givens rotation from right applied to eigenvalues
                for k in 0 .. curdim {
                    for i in 0 .. (k + 2) {
                        unsafe {
                            let a = *eigenvalues.get_unchecked(i, k);
                            let b = *eigenvalues.get_unchecked(i, k + 1);

                            *eigenvalues.get_unchecked_mut(i, k)     = c[k] * a - s[k] * b;
                            *eigenvalues.get_unchecked_mut(i, k + 1) = s[k] * a + c[k] * b;
                        }
                    }
                }


                // Shift back
                for k in 0 .. curdim + 1 {
                    unsafe {
                        let a = *eigenvalues.get_unchecked(k, k);
                        *eigenvalues.get_unchecked_mut(k, k) = a + lambda;
                    }
                }


                // Givens rotation from right applied to eigenvectors
                for k in 0 .. curdim {
                    for i in 0 .. dim.value() {

                        unsafe {
                            let a = *eigenvectors.get_unchecked(i, k);
                            let b = *eigenvectors.get_unchecked(i, k + 1);

                            *eigenvectors.get_unchecked_mut(i, k)     = c[k] * a - s[k] * b;
                            *eigenvectors.get_unchecked_mut(i, k + 1) = s[k] * a + c[k] * b;
                        }
                    }
                }

                iter = iter + 1;
                stop = true;

                for j in 0 .. curdim {
                    // Check last row.
                    if unsafe { *eigenvalues.get_unchecked(curdim, j) }.abs() >= eps {
                        stop = false;
                        break;
                    }

                    // Check last column.
                    if unsafe { *eigenvalues.get_unchecked(j, curdim) }.abs() >= eps {
                        stop = false;
                        break;
                    }
                }
            }


            if stop {
                if curdim > 1 {
                    curdim = curdim - 1;
                }
                else {
                    break;
                }
            }
        }

        (eigenvectors, eigenvalues.diagonal())
    }

    /// Cholesky decomposition G of a square symmetric positive definite matrix A, such that A = G * G^T
    ///
    /// Matrix symmetricness is not checked. Returns `None` if `self` is not definite positive.
    #[inline]
    pub fn cholesky(&self) -> Option<OwnedSquareMatrix<N, D, S::Alloc>> {
        let out = self.transpose();
        self.do_cholesky(out).ok()
    }

    /// Cholesky decomposition G of a square symmetric positive definite matrix A, such that A = G * G^T
    #[inline]
    pub fn cholesky_unchecked(&self) -> Result<OwnedSquareMatrix<N, D, S::Alloc>, &'static str> {
        let out = self.transpose();

        if !out.relative_eq(self, N::default_epsilon(), N::default_max_relative()) {
            return Err("Cholesky: Input matrix is not symmetric");
        }

        self.do_cholesky(out)
    }

    #[inline(always)]
    fn do_cholesky(&self, mut out: OwnedSquareMatrix<N, D, S::Alloc>)
                   -> Result<OwnedSquareMatrix<N, D, S::Alloc>, &'static str> {
        assert!(self.is_square(), "The input matrix must be square.");

        for i in 0 .. out.nrows() {
            for j in 0 .. (i + 1) {

                let mut sum = out[(i, j)];

                for k in 0 .. j {
                    sum = sum - out[(i, k)] * out[(j, k)];
                }

                if i > j {
                    out[(i, j)] = sum / out[(j, j)];
                }
                else if sum > N::zero() {
                    out[(i, i)] = sum.sqrt();
                }
                else {
                    return Err("Cholesky: Input matrix is not positive definite to machine precision.");
                }
            }
        }

        for i in 0 .. out.nrows() {
            for j in i + 1 .. out.ncols() {
                out[(i, j)] = N::zero();
            }
        }

        Ok(out)
    }

    /// Hessenberg
    /// Returns the matrix `self` in Hessenberg form and the corresponding similarity transformation
    ///
    /// # Returns
    /// The tuple (`q`, `h`) that `q * h * q^T = self`
    pub fn hessenberg(&self) -> (OwnedSquareMatrix<N, D, S::Alloc>, OwnedSquareMatrix<N, D, S::Alloc>)
        where S::Alloc: Allocator<N, D, U1> + Allocator<N, Dynamic, U1> {

        let (nrows, ncols) = self.data.shape();
        let mut h = self.clone_owned();

        let mut q = OwnedSquareMatrix::<N, D, S::Alloc>::identity_generic(nrows, ncols);

        if ncols.value() <= 2 {
            return (q, h);
        }

        // Temporary buffer that contains a column.
        let mut col = unsafe {
            OwnedColumnVector::<N, D, S::Alloc>::new_uninitialized_generic(nrows, U1)
        };

        for ite in 0 .. (ncols.value() - 2) {
            let subdim = Dynamic::new(nrows.value() - (ite + 1));
            let mut v = col.rows_mut(0, subdim.value());
            v.copy_from(&h.generic_slice((ite + 1, ite), (subdim, U1)));

            let alpha = v.norm();

            unsafe {
                let x = *v.get_unchecked(0, 0);
                *v.get_unchecked_mut(0, 0) = x - alpha;
            }

            if !v.normalize_mut().is_zero() {
                // XXX: we output the householder matrix to a pre-allocated matrix instead of
                // return a value to `p`. This would avoid allocation at each iteration.
                let p = OwnedSquareMatrix::<N, D, S::Alloc>::new_householder_generic(nrows, ite + 1, &v);

                q = q * &p;
                h = &p * h * p;
            }
        }

        (q, h)
    }
}
