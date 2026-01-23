//! Eigendecomposition of general (non-symmetric) square matrices.
//!
//! This module provides eigenvalue and eigenvector computation for square
//! real matrices that may be non-symmetric. Unlike [`SymmetricEigen`], this
//! handles matrices with complex eigenvalues.
//!
//! # Algorithm
//! 1. Compute the real Schur decomposition: A = Q * T * Qᵀ
//! 2. Extract eigenvalues from T (1×1 blocks = real, 2×2 blocks = complex pairs)
//! 3. Compute eigenvectors of T via back-substitution (handling 2×2 blocks)
//! 4. Transform eigenvectors back to original basis: v = Q * v_schur
//!
//! # Example
//! ```
//! use nalgebra::Matrix3;
//! use nalgebra::linalg::Eigen;
//!
//! let m = Matrix3::new(
//!     0.47, 0.53, 0.0,
//!     0.52, 0.0,  0.48,
//!     0.0,  0.0,  1.0,
//! );
//!
//! if let Some(eigen) = Eigen::new(m) {
//!     println!("Eigenvalues: {}", eigen.eigenvalues);
//! }
//! ```
//!
//! # See Also
//! - [`SymmetricEigen`](crate::linalg::SymmetricEigen) - More efficient for symmetric matrices
//! - [`Schur`](crate::linalg::Schur) - The underlying decomposition used here

use crate::RealField;
use crate::allocator::Allocator;
use crate::base::dimension::{Const, Dim, DimDiff, DimSub, U1};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, OMatrix, OVector, SquareMatrix};
use crate::linalg::Schur;
use num_complex::Complex;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

// =============================================================================
// EIGEN STRUCT
// =============================================================================

/// Eigendecomposition of a general square matrix.
///
/// This computes eigenvalues and eigenvectors for matrices that may be
/// non-symmetric. All results are complex-valued since non-symmetric
/// matrices can have complex eigenvalues even when all entries are real.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "OVector<Complex<T>, D>: Serialize,
                     OMatrix<Complex<T>, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "OVector<Complex<T>, D>: Deserialize<'de>,
                       OMatrix<Complex<T>, D, D>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct Eigen<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// The eigenvalues of the matrix (complex-valued).
    pub eigenvalues: OVector<Complex<T>, D>,
    /// The eigenvectors as columns of a matrix (complex-valued).
    pub eigenvectors: OMatrix<Complex<T>, D, D>,
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

impl<T: RealField, D: Dim + DimSub<Const<1>>> Eigen<T, D>
where
    DefaultAllocator:
        Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>> + Allocator<D, DimDiff<D, U1>>,
{
    /// Computes the eigendecomposition of a square matrix.
    ///
    /// Returns `None` if the Schur decomposition fails to converge.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to decompose
    ///
    /// # Example
    /// ```
    /// use nalgebra::Matrix2;
    /// use nalgebra::linalg::Eigen;
    ///
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      0.0, 3.0);
    /// let eigen = Eigen::new(m).unwrap();
    /// ```
    pub fn new(matrix: OMatrix<T, D, D>) -> Option<Self> {
        Self::try_new(matrix, T::default_epsilon(), 0)
    }

    /// Computes the eigendecomposition with custom convergence parameters.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to decompose
    /// * `eps` - Convergence tolerance for the Schur decomposition
    /// * `max_niter` - Maximum iterations (0 = use default)
    ///
    /// Returns `None` if the Schur decomposition fails to converge.
    pub fn try_new(matrix: OMatrix<T, D, D>, eps: T, max_niter: usize) -> Option<Self> {
        // Step 1: Compute Schur decomposition A = Q * T * Q^T
        let schur = Schur::try_new(matrix, eps.clone(), max_niter)?;
        let (q, t) = schur.unpack();

        let n = t.nrows();
        let dim = t.shape_generic().0;

        // Identify 2x2 block structure in the Schur form
        // block_size[i] = 1 for 1x1 block, 2 for start of 2x2 block, 0 for second row of 2x2
        let block_info = Self::identify_blocks(&t, n, eps.clone());

        // Step 2: Extract eigenvalues from the quasi-triangular matrix T
        let eigenvalues = Self::extract_eigenvalues(&t, &block_info, n, dim);

        // Step 3: Compute eigenvectors of T via back-substitution
        let eigenvectors_of_t =
            Self::compute_eigenvectors_of_schur(&t, &eigenvalues, &block_info, n, eps);

        // Step 4: Transform eigenvectors back to original basis: v = Q * v_schur
        let eigenvectors = Self::transform_eigenvectors(&q, &eigenvectors_of_t, n, dim);

        Some(Eigen {
            eigenvalues,
            eigenvectors,
        })
    }
}

// =============================================================================
// BLOCK IDENTIFICATION
// =============================================================================

impl<T: RealField, D: Dim> Eigen<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Identifies 2x2 blocks in the quasi-triangular Schur form.
    /// Returns a vector where:
    /// - 1 = 1x1 block
    /// - 2 = first row of 2x2 block
    /// - 0 = second row of 2x2 block (skip when iterating)
    fn identify_blocks(t: &OMatrix<T, D, D>, n: usize, eps: T) -> Vec<usize> {
        let mut block_info = vec![1usize; n];

        // Use a relative threshold for detecting 2x2 blocks
        let mut i = 0;
        while i < n {
            if i + 1 < n {
                let subdiag = t[(i + 1, i)].clone().abs();
                let diag_scale = t[(i, i)]
                    .clone()
                    .abs()
                    .max(t[(i + 1, i + 1)].clone().abs())
                    .max(T::one());

                // Relative threshold: subdiagonal element is significant if > eps * scale
                if subdiag > eps.clone() * diag_scale * (T::one() + T::one()) {
                    block_info[i] = 2; // Start of 2x2 block
                    block_info[i + 1] = 0; // Second row of 2x2 block
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }

        block_info
    }

    /// Extracts eigenvalues from a quasi-upper-triangular Schur matrix.
    fn extract_eigenvalues(
        t: &OMatrix<T, D, D>,
        block_info: &[usize],
        n: usize,
        dim: D,
    ) -> OVector<Complex<T>, D> {
        let zero_complex = Complex::new(T::zero(), T::zero());
        let mut eigenvalues =
            OVector::<Complex<T>, D>::from_element_generic(dim, Const::<1>, zero_complex);

        let mut i = 0;
        while i < n {
            if block_info[i] == 2 {
                // 2×2 block: complex conjugate pair
                let a = t[(i, i)].clone();
                let b = t[(i, i + 1)].clone();
                let c = t[(i + 1, i)].clone();
                let d = t[(i + 1, i + 1)].clone();

                let two = T::one() + T::one();
                let trace = a.clone() + d.clone();
                let half_trace = trace / two.clone();

                let diff = a - d;
                let quarter_diff_sq = (diff.clone() * diff) / (two.clone() * two.clone());
                let bc = b * c;
                let discriminant = quarter_diff_sq + bc;

                if discriminant < T::zero() {
                    let imag = (-discriminant).sqrt();
                    eigenvalues[i] = Complex::new(half_trace.clone(), imag.clone());
                    eigenvalues[i + 1] = Complex::new(half_trace, -imag);
                } else {
                    let sqrt_disc = discriminant.sqrt();
                    eigenvalues[i] =
                        Complex::new(half_trace.clone() + sqrt_disc.clone(), T::zero());
                    eigenvalues[i + 1] = Complex::new(half_trace - sqrt_disc, T::zero());
                }
                i += 2;
            } else if block_info[i] == 1 {
                // 1×1 block: real eigenvalue
                eigenvalues[i] = Complex::new(t[(i, i)].clone(), T::zero());
                i += 1;
            } else {
                // block_info[i] == 0: skip (second row of 2x2 block)
                i += 1;
            }
        }

        eigenvalues
    }

    /// Computes eigenvectors of the quasi-triangular Schur matrix T.
    fn compute_eigenvectors_of_schur(
        t: &OMatrix<T, D, D>,
        eigenvalues: &OVector<Complex<T>, D>,
        block_info: &[usize],
        n: usize,
        eps: T,
    ) -> Vec<Vec<Complex<T>>> {
        let mut eigenvectors: Vec<Vec<Complex<T>>> = Vec::with_capacity(n);

        let mut i = 0;
        while i < n {
            if block_info[i] == 2 {
                // 2×2 block: complex conjugate eigenvalue pair
                let lambda = eigenvalues[i].clone();
                let (v1, v2) =
                    Self::compute_eigenvectors_2x2_block(t, block_info, i, lambda, n, eps.clone());
                eigenvectors.push(v1);
                eigenvectors.push(v2);
                i += 2;
            } else if block_info[i] == 1 {
                // 1×1 block: real eigenvalue
                let lambda = eigenvalues[i].re.clone();
                let v =
                    Self::compute_eigenvector_1x1_block(t, block_info, i, lambda, n, eps.clone());
                eigenvectors.push(v);
                i += 1;
            } else {
                // block_info[i] == 0: skip
                i += 1;
            }
        }

        eigenvectors
    }

    /// Computes eigenvector for a 1x1 block (real eigenvalue) at position k.
    /// Handles 2x2 blocks during back-substitution by solving 2x2 systems.
    fn compute_eigenvector_1x1_block(
        t: &OMatrix<T, D, D>,
        block_info: &[usize],
        k: usize,
        lambda: T,
        n: usize,
        eps: T,
    ) -> Vec<Complex<T>> {
        let zero_complex = Complex::new(T::zero(), T::zero());
        let mut v = vec![zero_complex; n];

        // v[m] = 0 for m > k
        // v[k] = 1
        v[k] = Complex::new(T::one(), T::zero());

        // Back-substitute for j = k-1, k-2, ..., 0
        let mut j = k as isize - 1;
        while j >= 0 {
            let ju = j as usize;

            if ju > 0 && block_info[ju - 1] == 2 {
                // ju is second row of a 2x2 block at (ju-1, ju-1)
                // Solve the 2x2 system for v[ju-1] and v[ju] simultaneously
                let p = ju - 1;

                // Compute right-hand sides
                let mut b1_re = T::zero();
                let mut b2_re = T::zero();
                for m in (ju + 1)..=k {
                    b1_re = b1_re - t[(p, m)].clone() * v[m].re.clone();
                    b2_re = b2_re - t[(ju, m)].clone() * v[m].re.clone();
                }

                // 2x2 system matrix:
                // [t[p,p] - λ,     t[p,ju]    ] [v[p] ]   [b1]
                // [t[ju,p],        t[ju,ju] - λ] [v[ju]] = [b2]
                let a11 = t[(p, p)].clone() - lambda.clone();
                let a12 = t[(p, ju)].clone();
                let a21 = t[(ju, p)].clone();
                let a22 = t[(ju, ju)].clone() - lambda.clone();

                // Solve using Cramer's rule
                let det = a11.clone() * a22.clone() - a12.clone() * a21.clone();

                if det.clone().abs() > eps.clone() {
                    let v_p =
                        (b1_re.clone() * a22.clone() - b2_re.clone() * a12.clone()) / det.clone();
                    let v_ju = (a11.clone() * b2_re - a21.clone() * b1_re) / det;
                    v[p] = Complex::new(v_p, T::zero());
                    v[ju] = Complex::new(v_ju, T::zero());
                } else {
                    v[p] = Complex::new(T::zero(), T::zero());
                    v[ju] = Complex::new(T::zero(), T::zero());
                }

                j -= 2;
            } else if block_info[ju] == 2 {
                // ju is first row of a 2x2 block, but we're coming from above
                // Solve the 2x2 system for v[ju] and v[ju+1] simultaneously
                let p = ju;
                let q = ju + 1;

                // Compute right-hand sides
                let mut b1_re = T::zero();
                let mut b2_re = T::zero();
                for m in (q + 1)..=k {
                    b1_re = b1_re - t[(p, m)].clone() * v[m].re.clone();
                    b2_re = b2_re - t[(q, m)].clone() * v[m].re.clone();
                }

                let a11 = t[(p, p)].clone() - lambda.clone();
                let a12 = t[(p, q)].clone();
                let a21 = t[(q, p)].clone();
                let a22 = t[(q, q)].clone() - lambda.clone();

                let det = a11.clone() * a22.clone() - a12.clone() * a21.clone();

                if det.clone().abs() > eps.clone() {
                    let v_p =
                        (b1_re.clone() * a22.clone() - b2_re.clone() * a12.clone()) / det.clone();
                    let v_q = (a11.clone() * b2_re - a21.clone() * b1_re) / det;
                    v[p] = Complex::new(v_p, T::zero());
                    v[q] = Complex::new(v_q, T::zero());
                } else {
                    v[p] = Complex::new(T::zero(), T::zero());
                    v[q] = Complex::new(T::zero(), T::zero());
                }

                j -= 2;
            } else {
                // Standard 1x1 back-substitution
                let diag = t[(ju, ju)].clone() - lambda.clone();

                let mut sum_re = T::zero();
                for m in (ju + 1)..=k {
                    sum_re = sum_re + t[(ju, m)].clone() * v[m].re.clone();
                }

                if diag.clone().abs() > eps {
                    v[ju] = Complex::new(-sum_re / diag, T::zero());
                } else {
                    v[ju] = Complex::new(T::zero(), T::zero());
                }

                j -= 1;
            }
        }

        Self::normalize_vector(&mut v);
        v
    }

    /// Computes eigenvectors for a 2x2 block (complex conjugate eigenvalue pair).
    fn compute_eigenvectors_2x2_block(
        t: &OMatrix<T, D, D>,
        block_info: &[usize],
        k: usize,
        lambda: Complex<T>,
        n: usize,
        eps: T,
    ) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
        let zero_complex = Complex::new(T::zero(), T::zero());
        let mut v = vec![zero_complex.clone(); n];

        // For the 2x2 block at (k, k), solve the local 2x2 eigenproblem
        let a = t[(k, k)].clone();
        let b = t[(k, k + 1)].clone();

        // (a - λ)v[k] + b*v[k+1] = 0
        // Set v[k+1] = 1, then v[k] = -b / (a - λ)
        let a_minus_lambda_re = a - lambda.re.clone();
        let a_minus_lambda_im = -lambda.im.clone();

        let denom = a_minus_lambda_re.clone() * a_minus_lambda_re.clone()
            + a_minus_lambda_im.clone() * a_minus_lambda_im.clone();

        if denom.clone().sqrt() > eps.clone() {
            // -b / (a - λ) = -b * conj(a - λ) / |a - λ|²
            let neg_b = -b;
            v[k] = Complex::new(
                neg_b.clone() * a_minus_lambda_re / denom.clone(),
                neg_b * (-a_minus_lambda_im) / denom,
            );
        } else {
            v[k] = Complex::new(T::one(), T::zero());
        }

        v[k + 1] = Complex::new(T::one(), T::zero());

        // Back-substitute for j = k-1, k-2, ..., 0
        let mut j = k as isize - 1;
        while j >= 0 {
            let ju = j as usize;

            if ju > 0 && block_info[ju - 1] == 2 {
                // 2x2 block at (ju-1, ju)
                let p = ju - 1;

                // Compute complex RHS
                let mut b1 = Complex::new(T::zero(), T::zero());
                let mut b2 = Complex::new(T::zero(), T::zero());
                for m in (ju + 1)..=(k + 1) {
                    let t_pm = t[(p, m)].clone();
                    let t_jm = t[(ju, m)].clone();
                    b1 = Self::complex_add(b1, Self::complex_scale(-t_pm, v[m].clone()));
                    b2 = Self::complex_add(b2, Self::complex_scale(-t_jm, v[m].clone()));
                }

                // Complex 2x2 system
                let a11 = Complex::new(t[(p, p)].clone() - lambda.re.clone(), -lambda.im.clone());
                let a12 = Complex::new(t[(p, ju)].clone(), T::zero());
                let a21 = Complex::new(t[(ju, p)].clone(), T::zero());
                let a22 = Complex::new(t[(ju, ju)].clone() - lambda.re.clone(), -lambda.im.clone());

                // det = a11*a22 - a12*a21
                let det = Self::complex_sub(
                    Self::complex_mul(a11.clone(), a22.clone()),
                    Self::complex_mul(a12.clone(), a21.clone()),
                );

                let det_norm =
                    (det.re.clone() * det.re.clone() + det.im.clone() * det.im.clone()).sqrt();

                if det_norm > eps.clone() {
                    // v[p] = (b1*a22 - b2*a12) / det
                    // v[ju] = (a11*b2 - a21*b1) / det
                    let v_p = Self::complex_div(
                        Self::complex_sub(
                            Self::complex_mul(b1.clone(), a22.clone()),
                            Self::complex_mul(b2.clone(), a12.clone()),
                        ),
                        det.clone(),
                    );
                    let v_ju = Self::complex_div(
                        Self::complex_sub(Self::complex_mul(a11, b2), Self::complex_mul(a21, b1)),
                        det,
                    );
                    v[p] = v_p;
                    v[ju] = v_ju;
                }

                j -= 2;
            } else if block_info[ju] == 2 {
                // 2x2 block starting at ju
                let p = ju;
                let q = ju + 1;

                let mut b1 = Complex::new(T::zero(), T::zero());
                let mut b2 = Complex::new(T::zero(), T::zero());
                for m in (q + 1)..=(k + 1) {
                    let t_pm = t[(p, m)].clone();
                    let t_qm = t[(q, m)].clone();
                    b1 = Self::complex_add(b1, Self::complex_scale(-t_pm, v[m].clone()));
                    b2 = Self::complex_add(b2, Self::complex_scale(-t_qm, v[m].clone()));
                }

                let a11 = Complex::new(t[(p, p)].clone() - lambda.re.clone(), -lambda.im.clone());
                let a12 = Complex::new(t[(p, q)].clone(), T::zero());
                let a21 = Complex::new(t[(q, p)].clone(), T::zero());
                let a22 = Complex::new(t[(q, q)].clone() - lambda.re.clone(), -lambda.im.clone());

                let det = Self::complex_sub(
                    Self::complex_mul(a11.clone(), a22.clone()),
                    Self::complex_mul(a12.clone(), a21.clone()),
                );

                let det_norm =
                    (det.re.clone() * det.re.clone() + det.im.clone() * det.im.clone()).sqrt();

                if det_norm > eps.clone() {
                    let v_p = Self::complex_div(
                        Self::complex_sub(
                            Self::complex_mul(b1.clone(), a22.clone()),
                            Self::complex_mul(b2.clone(), a12.clone()),
                        ),
                        det.clone(),
                    );
                    let v_q = Self::complex_div(
                        Self::complex_sub(Self::complex_mul(a11, b2), Self::complex_mul(a21, b1)),
                        det,
                    );
                    v[p] = v_p;
                    v[q] = v_q;
                }

                j -= 2;
            } else {
                // Standard 1x1 complex back-substitution
                let diag =
                    Complex::new(t[(ju, ju)].clone() - lambda.re.clone(), -lambda.im.clone());

                let mut sum = Complex::new(T::zero(), T::zero());
                for m in (ju + 1)..=(k + 1) {
                    let t_jm = t[(ju, m)].clone();
                    sum = Self::complex_add(sum, Self::complex_scale(t_jm, v[m].clone()));
                }

                let diag_norm =
                    (diag.re.clone() * diag.re.clone() + diag.im.clone() * diag.im.clone()).sqrt();

                if diag_norm > eps {
                    v[ju] = Self::complex_div(Self::complex_neg(sum), diag);
                }

                j -= 1;
            }
        }

        Self::normalize_vector(&mut v);

        // Conjugate eigenvector
        let v_conj: Vec<Complex<T>> = v
            .iter()
            .map(|c| Complex::new(c.re.clone(), -c.im.clone()))
            .collect();

        (v, v_conj)
    }

    // Complex arithmetic helpers
    fn complex_add(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex::new(a.re + b.re, a.im + b.im)
    }

    fn complex_sub(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex::new(a.re - b.re, a.im - b.im)
    }

    fn complex_mul(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex::new(
            a.re.clone() * b.re.clone() - a.im.clone() * b.im.clone(),
            a.re * b.im + a.im * b.re,
        )
    }

    fn complex_div(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        let denom = b.re.clone() * b.re.clone() + b.im.clone() * b.im.clone();
        Complex::new(
            (a.re.clone() * b.re.clone() + a.im.clone() * b.im.clone()) / denom.clone(),
            (a.im * b.re - a.re * b.im) / denom,
        )
    }

    fn complex_neg(a: Complex<T>) -> Complex<T> {
        Complex::new(-a.re, -a.im)
    }

    fn complex_scale(s: T, a: Complex<T>) -> Complex<T> {
        Complex::new(s.clone() * a.re, s * a.im)
    }

    /// Normalizes a complex vector to unit length.
    fn normalize_vector(v: &mut [Complex<T>]) {
        let mut norm_sq = T::zero();
        for c in v.iter() {
            norm_sq = norm_sq + c.re.clone() * c.re.clone() + c.im.clone() * c.im.clone();
        }
        let norm = norm_sq.sqrt();

        if norm > T::default_epsilon() {
            for c in v.iter_mut() {
                c.re = c.re.clone() / norm.clone();
                c.im = c.im.clone() / norm.clone();
            }
        }
    }

    /// Transforms eigenvectors from Schur basis back to original basis.
    fn transform_eigenvectors(
        q: &OMatrix<T, D, D>,
        eigenvectors_of_t: &[Vec<Complex<T>>],
        n: usize,
        dim: D,
    ) -> OMatrix<Complex<T>, D, D> {
        let zero_complex = Complex::new(T::zero(), T::zero());
        let mut result = OMatrix::<Complex<T>, D, D>::from_element_generic(dim, dim, zero_complex);

        for (col, v_t) in eigenvectors_of_t.iter().enumerate() {
            for i in 0..n {
                let mut sum_re = T::zero();
                let mut sum_im = T::zero();
                for m in 0..n {
                    let q_im = q[(i, m)].clone();
                    sum_re = sum_re + q_im.clone() * v_t[m].re.clone();
                    sum_im = sum_im + q_im * v_t[m].im.clone();
                }
                result[(i, col)] = Complex::new(sum_re, sum_im);
            }
        }

        result
    }
}

// =============================================================================
// ACCESSORS AND VERIFICATION
// =============================================================================

impl<T: RealField, D: Dim> Eigen<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &OVector<Complex<T>, D> {
        &self.eigenvalues
    }

    /// Returns the eigenvectors as columns of a matrix.
    pub fn eigenvectors(&self) -> &OMatrix<Complex<T>, D, D> {
        &self.eigenvectors
    }

    /// Verifies the decomposition by computing ||AV - VD|| / ||A||.
    pub fn verify(&self, original: &OMatrix<T, D, D>) -> T {
        let n = original.nrows();
        let dim = original.shape_generic().0;

        let zero_complex = Complex::new(T::zero(), T::zero());

        let mut av =
            OMatrix::<Complex<T>, D, D>::from_element_generic(dim, dim, zero_complex.clone());
        for i in 0..n {
            for j in 0..n {
                let mut sum_re = T::zero();
                let mut sum_im = T::zero();
                for m in 0..n {
                    let a_im = original[(i, m)].clone();
                    let v_mj = &self.eigenvectors[(m, j)];
                    sum_re = sum_re + a_im.clone() * v_mj.re.clone();
                    sum_im = sum_im + a_im * v_mj.im.clone();
                }
                av[(i, j)] = Complex::new(sum_re, sum_im);
            }
        }

        let mut vd = OMatrix::<Complex<T>, D, D>::from_element_generic(dim, dim, zero_complex);
        for i in 0..n {
            for j in 0..n {
                let v_ij = &self.eigenvectors[(i, j)];
                let lambda_j = &self.eigenvalues[j];
                let re =
                    v_ij.re.clone() * lambda_j.re.clone() - v_ij.im.clone() * lambda_j.im.clone();
                let im =
                    v_ij.re.clone() * lambda_j.im.clone() + v_ij.im.clone() * lambda_j.re.clone();
                vd[(i, j)] = Complex::new(re, im);
            }
        }

        let mut diff_norm_sq = T::zero();
        for i in 0..n {
            for j in 0..n {
                let diff_re = av[(i, j)].re.clone() - vd[(i, j)].re.clone();
                let diff_im = av[(i, j)].im.clone() - vd[(i, j)].im.clone();
                diff_norm_sq = diff_norm_sq + diff_re.clone() * diff_re + diff_im.clone() * diff_im;
            }
        }
        let diff_norm = diff_norm_sq.sqrt();

        let mut a_norm_sq = T::zero();
        for i in 0..n {
            for j in 0..n {
                let val = original[(i, j)].clone();
                a_norm_sq = a_norm_sq + val.clone() * val;
            }
        }
        let a_norm = a_norm_sq.sqrt();

        if a_norm > T::default_epsilon() {
            diff_norm / a_norm
        } else {
            diff_norm
        }
    }
}

// =============================================================================
// CONVENIENCE METHODS FOR SQUAREMATRIX
// =============================================================================

impl<T: RealField, D: Dim + DimSub<Const<1>>, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    DefaultAllocator:
        Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>> + Allocator<D, DimDiff<D, U1>>,
{
    /// Computes the eigendecomposition of this matrix.
    pub fn eigen(self) -> Option<Eigen<T, D>>
    where
        Self: Sized,
        DefaultAllocator: Allocator<D, D>,
    {
        Eigen::new(self.into_owned())
    }
}
