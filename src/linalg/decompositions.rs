use traits::operations::{Transpose, ApproxEq};
use traits::structure::{ColumnSlice, Eye, Indexable, Diagonal, SquareMatrix, BaseFloat, Cast};
use traits::geometry::Norm;
use std::cmp;
use std::ops::{Mul, Add, Sub};

/// Get the householder matrix corresponding to a reflexion to the hyperplane
/// defined by `vector`. It can be a reflexion contained in a subspace.
///
/// # Arguments
/// * `dimension` - the dimension of the space the resulting matrix operates in
/// * `start` - the starting dimension of the subspace of the reflexion
/// * `vector` - the vector defining the reflection.
pub fn householder_matrix<N, V, M>(dimension: usize, start: usize, vector: V) -> M
    where N: BaseFloat,
          M: Eye + Indexable<(usize, usize), N>,
          V: Indexable<usize, N> {
    let mut qk : M = Eye::new_identity(dimension);
    let subdim = vector.shape();

    let stop = subdim + start;

    assert!(dimension >= stop);

    for j in start .. stop {
        for i in start .. stop {
            unsafe {
                let vv = vector.unsafe_at(i - start) * vector.unsafe_at(j - start);
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
          M: Copy + Eye + ColumnSlice<V> + Transpose + Indexable<(usize, usize), N> +
             Mul<M, Output = M> {
    let (rows, cols) = m.shape();
    assert!(rows >= cols);
    let mut q : M = Eye::new_identity(rows);
    let mut r = *m;

    for ite in 0 .. cmp::min(rows - 1, cols) {
        let mut v = r.column_slice(ite, ite, rows);
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

/// Eigendecomposition of a square symmetric matrix using the qr algorithm
pub fn eigen_qr<N, V, VS, M>(m: &M, eps: &N, niter: usize) -> (M, V)
    where N:  BaseFloat,
          V:  Mul<M, Output = V>,
          VS: Indexable<usize, N> + Norm<N>,
          M:  Indexable<(usize, usize), N> + SquareMatrix<N, V> + Add<M, Output = M> +
              Sub<M, Output = M> + ColumnSlice<VS> +
              ApproxEq<N> + Copy {

    let (mut eigenvectors, mut eigenvalues) = hessenberg(m);

    // Allocate arrays for Givens rotation components
    let mut c = Vec::<N>::with_capacity(::dimension::<M>() - 1);
    let mut s = Vec::<N>::with_capacity(::dimension::<M>() - 1);

    if ::dimension::<M>() == 1 {
        return (eigenvectors, eigenvalues.diagonal());
    }

    unsafe { 
        c.set_len(::dimension::<M>() - 1);
        s.set_len(::dimension::<M>() - 1);
    }

    let mut iter = 0;
    let mut curdim = ::dimension::<M>() - 1;

    for _ in 0 .. ::dimension::<M>() {

        let mut stop = false;

        while !stop && iter < niter {

            let lambda;

            unsafe {
                let a = eigenvalues.unsafe_at((curdim - 1, curdim - 1));
                let b = eigenvalues.unsafe_at((curdim - 1, curdim));
                let c = eigenvalues.unsafe_at((curdim, curdim - 1));
                let d = eigenvalues.unsafe_at((curdim, curdim));

                let trace = a + d;
                let determinant = a * d - b * c;

                let constquarter: N = Cast::from(0.25f64);
                let consthalf: N = Cast::from(0.5f64);

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
                    let a = eigenvalues.unsafe_at((k, k));
                    eigenvalues.unsafe_set((k, k), a - lambda);
                }
            }
            

            // Givens rotation from left
            for k in 0 .. curdim {
                let x_i = unsafe { eigenvalues.unsafe_at((k, k)) };
                let x_j = unsafe { eigenvalues.unsafe_at((k + 1, k)) };

                let ctmp;
                let stmp;

                if x_j.abs() < *eps {
                    ctmp = N::one();
                    stmp = N::zero();
                }
                else if x_i.abs() < *eps {
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
                        let a = eigenvalues.unsafe_at((k, j));
                        let b = eigenvalues.unsafe_at((k + 1, j));

                        eigenvalues.unsafe_set((k, j), ctmp * a - stmp * b);
                        eigenvalues.unsafe_set((k + 1, j), stmp * a + ctmp * b);
                    }

                }
            }

            // Givens rotation from right applied to eigenvalues
            for k in 0 .. curdim {
                for i in 0 .. (k + 2) {

                    unsafe {
                        let a = eigenvalues.unsafe_at((i, k));
                        let b = eigenvalues.unsafe_at((i, k + 1));


                        eigenvalues.unsafe_set((i, k), c[k] * a - s[k] * b);
                        eigenvalues.unsafe_set((i, k + 1), s[k] * a + c[k] * b);
                    }
                }

            }

            
            // Shift back
            for k in 0 .. curdim + 1 {
                unsafe { 
                    let a = eigenvalues.unsafe_at((k, k));
                    eigenvalues.unsafe_set((k, k), a + lambda);
                }
            }
            

            // Givens rotation from right applied to eigenvectors
            for k in 0 .. curdim {
                for i in 0 .. ::dimension::<M>() {

                    unsafe {
                        let a = eigenvectors.unsafe_at((i, k));
                        let b = eigenvectors.unsafe_at((i, k + 1));


                        eigenvectors.unsafe_set((i, k), c[k] * a - s[k] * b);
                        eigenvectors.unsafe_set((i, k + 1), s[k] * a + c[k] * b);
                    }
                }
            }

            iter = iter + 1;

            stop = true;

            for j in 0 .. curdim {

                // Check last row
                if unsafe { eigenvalues.unsafe_at((curdim, j)) }.abs() >= *eps {
                    stop = false;
                    break;
                }

                // Check last column
                if unsafe { eigenvalues.unsafe_at((j, curdim)) }.abs() >= *eps {
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
/// # Arguments
/// * `m` - square symmetric positive definite matrix to decompose
pub fn cholesky<N, V, VS, M>(m: &M) -> Result<M, &'static str>
    where N:  BaseFloat,
          V:  Mul<M, Output = V>,
          VS: Indexable<usize, N> + Norm<N>,
          M:  Indexable<(usize, usize), N> + SquareMatrix<N, V> + Add<M, Output = M> +
              Sub<M, Output = M> + ColumnSlice<VS> +
              ApproxEq<N> + Copy {

    let mut out = m.transpose();

    if !ApproxEq::approx_eq(&out, &m) {
        return Err("Cholesky: Input matrix is not symmetric");
    }

    for i in 0 .. out.nrows() {
        for j in 0 .. (i + 1) {

            let mut sum: N = out[(i, j)];

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
                return Err("Cholesky: Input matrix is not positive definite to machine precision");
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
/// Returns the matrix m in Hessenberg form and the corresponding similarity transformation
///
/// # Arguments
/// * `m` - matrix to transform
///
/// # Returns
/// * First return value `q` - Similarity matrix p such that q * h * q^T = m
/// * Second return value `h` - Matrix m in Hessenberg form
pub fn hessenberg<N, V, M>(m: &M) -> (M, M)
    where N: BaseFloat,
          V: Indexable<usize, N> + Norm<N>,
          M: Copy + Eye + ColumnSlice<V> + Transpose + Indexable<(usize, usize), N> +
             Mul<M, Output = M> {
    
    let mut h = *m;
    let (rows, cols) = h.shape();

    let mut q : M = Eye::new_identity(cols);

    if cols <= 2 {
        return (q, h);
    }

    for ite in 0 .. (cols - 2) {
        let mut v = h.column_slice(ite, ite + 1, rows);

        let alpha = Norm::norm(&v);

        unsafe {
            let x = v.unsafe_at(0);
            v.unsafe_set(0, x - alpha);
        }

        if !::is_zero(&v.normalize_mut()) {
            let p: M = householder_matrix(rows, ite + 1, v);

            q = q * p;
            h = p * h * p;
        }
    }

    (q, h)
}
