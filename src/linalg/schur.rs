#![allow(clippy::suspicious_operation_groupings)]
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;
use num_complex::Complex as NumComplex;
use num_traits::identities::Zero;
use simba::scalar::{ComplexField, RealField};
use std::cmp;

use crate::allocator::Allocator;
use crate::base::dimension::{Const, Dim, DimDiff, DimSub, Dyn, U1, U2};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, OMatrix, OVector, SquareMatrix, Unit, Vector2, Vector3};

use crate::geometry::Reflection;
use crate::linalg::Hessenberg;
use crate::linalg::givens::GivensRotation;
use crate::linalg::householder;
use crate::{Matrix, UninitVector};
use std::mem::MaybeUninit;

/// Schur decomposition of a square matrix.
///
/// The Schur decomposition is a factorization of a square matrix A into the form:
/// **A = Q * T * Q<sup>H</sup>**
///
/// where:
/// - **Q** is a unitary (or orthogonal for real matrices) matrix
/// - **T** is an upper quasi-triangular matrix (upper triangular for complex matrices)
/// - **Q<sup>H</sup>** is the conjugate transpose of Q
///
/// # What is Schur Decomposition?
///
/// The Schur decomposition is a fundamental matrix factorization that transforms a matrix
/// into a form where its eigenvalues appear on the diagonal (or in 2×2 blocks). This makes
/// it particularly useful for:
///
/// - **Eigenvalue computation**: The diagonal entries (or 2×2 blocks) of T contain the eigenvalues
/// - **Matrix functions**: Computing functions like exp(A), log(A), or sqrt(A)
/// - **Stability analysis**: Used extensively in control theory and dynamical systems
/// - **Numerical algorithms**: Forms the basis for many advanced matrix algorithms
///
/// For real matrices, T is quasi-triangular, meaning it's block upper triangular with 1×1
/// and 2×2 blocks on the diagonal. The 2×2 blocks correspond to complex conjugate eigenvalue pairs.
///
/// If this is a real matrix, this will be a `RealField` Schur decomposition.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<D, D>,
         OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<D, D>,
         OMatrix<T, D, D>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct Schur<T: ComplexField, D: Dim>
where
    DefaultAllocator: Allocator<D, D>,
{
    q: OMatrix<T, D, D>,
    t: OMatrix<T, D, D>,
}

impl<T: ComplexField, D: Dim> Copy for Schur<T, D>
where
    DefaultAllocator: Allocator<D, D>,
    OMatrix<T, D, D>: Copy,
{
}

impl<T: ComplexField, D: Dim> Schur<T, D>
where
    D: DimSub<U1>, // For Hessenberg.
    DefaultAllocator:
        Allocator<D, DimDiff<D, U1>> + Allocator<DimDiff<D, U1>> + Allocator<D, D> + Allocator<D>,
{
    /// Computes the Schur decomposition of a square matrix.
    ///
    /// This method decomposes a square matrix **A** into **Q * T * Q<sup>H</sup>**, where
    /// **Q** is unitary and **T** is upper quasi-triangular.
    ///
    /// # What This Does
    ///
    /// The Schur decomposition transforms your matrix into a special form that makes it
    /// easy to extract eigenvalues and perform advanced matrix operations. The resulting
    /// **T** matrix has all eigenvalues on its diagonal (as 1×1 or 2×2 blocks), while
    /// preserving all the important properties of your original matrix.
    ///
    /// # When to Use This
    ///
    /// - When you need both eigenvalues and a stable numerical representation
    /// - For computing matrix functions (exponentials, logarithms, powers)
    /// - In control theory for analyzing system stability
    /// - When eigenvalue algorithms need to be robust to numerical errors
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or if the decomposition fails to converge.
    /// Use [`try_new`](#method.try_new) for a non-panicking version.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix4};
    ///
    /// // Basic 3x3 matrix decomposition
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0
    /// );
    ///
    /// let schur = m.clone().schur();
    /// let (q, t) = schur.unpack();
    ///
    /// // Verify the decomposition: A ≈ Q * T * Q^T
    /// let reconstructed = &q * &t * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // The eigenvalues are on the diagonal of T
    /// println!("Eigenvalue 1: {}", t[(0, 0)]);
    /// println!("Eigenvalue 2: {}", t[(1, 1)]);
    /// println!("Eigenvalue 3: {}", t[(2, 2)]);
    /// ```
    ///
    /// # Example: Symmetric Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Symmetric matrices have real eigenvalues
    /// let m = Matrix2::new(
    ///     4.0, 1.0,
    ///     1.0, 3.0
    /// );
    ///
    /// let schur = m.clone().schur();
    /// let (q, t) = schur.unpack();
    ///
    /// // Q is orthogonal: Q * Q^T = I
    /// let identity = &q * q.transpose();
    /// assert!(identity.is_identity(1e-10));
    ///
    /// // Verify decomposition
    /// let reconstructed = &q * &t * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_new`](#method.try_new): Non-panicking version with custom convergence parameters
    /// - [`eigenvalues`](#method.eigenvalues): Extract real eigenvalues from the decomposition
    /// - [`complex_eigenvalues`](#method.complex_eigenvalues): Extract complex eigenvalues
    /// - [`unpack`](#method.unpack): Get the Q and T matrices separately
    pub fn new(m: OMatrix<T, D, D>) -> Self {
        Self::try_new(m, T::RealField::default_epsilon(), 0).unwrap()
    }

    /// Attempts to compute the Schur decomposition of a square matrix.
    ///
    /// This is the safe, non-panicking version of [`new`](#method.new) that allows you to
    /// control convergence criteria and handle decomposition failures gracefully.
    ///
    /// # What This Does
    ///
    /// Like [`new`](#method.new), this computes **A = Q * T * Q<sup>H</sup>**, but returns
    /// `None` if the algorithm doesn't converge within the specified iterations, instead of
    /// panicking. This is useful when working with ill-conditioned matrices or when you need
    /// robust error handling.
    ///
    /// # When to Use This
    ///
    /// - When you need fine control over convergence tolerance
    /// - When working with potentially ill-conditioned matrices
    /// - When you need to limit computation time
    /// - In production code where panics are unacceptable
    ///
    /// If only eigenvalues are needed, it is more efficient to call the matrix method
    /// `.eigenvalues()` instead.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0. Smaller values
    ///   give more accurate results but may require more iterations. The default is typically
    ///   around 1e-10 for f64.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    ///   number of iteration is exceeded, `None` is returned. If `max_niter == 0`, then the
    ///   algorithm continues indefinitely until convergence.
    ///
    /// # Returns
    ///
    /// - `Some(Schur)` if the decomposition converges successfully
    /// - `None` if the algorithm fails to converge within `max_niter` iterations
    ///
    /// # Example: Basic Usage with Custom Tolerance
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, -1.0,  0.0,
    ///     -1.0,  2.0, -1.0,
    ///     0.0, -1.0,  2.0
    /// );
    ///
    /// // Try with custom tolerance and iteration limit
    /// let eps = 1e-10;
    /// let max_iterations = 100;
    ///
    /// match m.clone().try_schur(eps, max_iterations) {
    ///     Some(schur) => {
    ///         let (q, t) = schur.unpack();
    ///         println!("Decomposition successful!");
    ///         // Verify: A = Q * T * Q^T
    ///         let reconstructed = &q * &t * q.transpose();
    ///         assert!(reconstructed.relative_eq(&m, 1e-9, 1e-9));
    ///     },
    ///     None => {
    ///         println!("Decomposition failed to converge");
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Handling Non-Convergence
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0
    /// );
    ///
    /// // Very strict iteration limit (may not converge)
    /// let strict_limit = 1;
    /// if let Some(schur) = m.clone().try_schur(1e-10, strict_limit) {
    ///     println!("Converged quickly!");
    /// } else {
    ///     println!("Need more iterations");
    ///     // Retry with more reasonable limit
    ///     let schur = m.try_schur(1e-10, 100).expect("Should converge now");
    /// }
    /// ```
    ///
    /// # Example: Control Theory Application
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // State transition matrix from a dynamical system
    /// let a = Matrix2::new(
    ///     0.8, 0.1,
    ///     0.2, 0.9
    /// );
    ///
    /// if let Some(schur) = a.clone().try_schur(1e-12, 200) {
    ///     // Check system stability: all eigenvalues should have magnitude < 1
    ///     if let Some(eigenvalues) = schur.eigenvalues() {
    ///         let mut is_stable = true;
    ///         for i in 0..eigenvalues.len() {
    ///             let ev: f64 = eigenvalues[i];
    ///             if ev.abs() >= 1.0 {
    ///                 is_stable = false;
    ///                 break;
    ///             }
    ///         }
    ///         println!("System is stable: {}", is_stable);
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new`](#method.new): Panicking version with default parameters
    /// - [`eigenvalues`](#method.eigenvalues): Extract eigenvalues after decomposition
    /// - [`unpack`](#method.unpack): Retrieve the Q and T matrices
    pub fn try_new(m: OMatrix<T, D, D>, eps: T::RealField, max_niter: usize) -> Option<Self> {
        let mut work = Matrix::zeros_generic(m.shape_generic().0, Const::<1>);

        Self::do_decompose(m, &mut work, eps, max_niter, true)
            .map(|(q, t)| Schur { q: q.unwrap(), t })
    }

    fn do_decompose(
        mut m: OMatrix<T, D, D>,
        work: &mut OVector<T, D>,
        eps: T::RealField,
        max_niter: usize,
        compute_q: bool,
    ) -> Option<(Option<OMatrix<T, D, D>>, OMatrix<T, D, D>)> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvectors and eigenvalues of a non-square matrix."
        );

        let dim = m.shape_generic().0;

        // Specialization would make this easier.
        if dim.value() == 0 {
            let vecs = Some(OMatrix::from_element_generic(dim, dim, T::zero()));
            let vals = OMatrix::from_element_generic(dim, dim, T::zero());
            return Some((vecs, vals));
        } else if dim.value() == 1 {
            if compute_q {
                let q = OMatrix::from_element_generic(dim, dim, T::one());
                return Some((Some(q), m));
            } else {
                return Some((None, m));
            }
        } else if dim.value() == 2 {
            return decompose_2x2(m, compute_q);
        }

        let amax_m = m.camax();
        // if amax_m == 0 (i.e. the matrix is the zero matrix),
        // then the unscale_mut call will turn the entire matrix into NaNs
        // see https://github.com/dimforge/nalgebra/issues/1291
        if !amax_m.is_zero() {
            m.unscale_mut(amax_m.clone());
        }

        let hess = Hessenberg::new_with_workspace(m, work);
        let mut q;
        let mut t;

        if compute_q {
            // TODO: could we work without unpacking? Using only the internal representation of
            // hessenberg decomposition.
            let (vecs, vals) = hess.unpack();
            q = Some(vecs);
            t = vals;
        } else {
            q = None;
            t = hess.unpack_h()
        }

        // Implicit double-shift QR method.
        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&mut t, eps.clone(), dim.value() - 1);

        while end != start {
            let subdim = end - start + 1;

            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let h11 = t[(start, start)].clone();
                let h12 = t[(start, start + 1)].clone();
                let h21 = t[(start + 1, start)].clone();
                let h22 = t[(start + 1, start + 1)].clone();
                let h32 = t[(start + 2, start + 1)].clone();

                let hnn = t[(n, n)].clone();
                let hmm = t[(m, m)].clone();
                let hnm = t[(n, m)].clone();
                let hmn = t[(m, n)].clone();

                let tra = hnn.clone() + hmm.clone();
                let det = hnn * hmm - hnm * hmn;

                let mut axis = Vector3::new(
                    h11.clone() * h11.clone() + h12 * h21.clone() - tra.clone() * h11.clone() + det,
                    h21.clone() * (h11 + h22 - tra),
                    h21 * h32,
                );

                for k in start..n - 1 {
                    let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                    if not_zero {
                        if k > start {
                            t[(k, k - 1)] = norm;
                            t[(k + 1, k - 1)] = T::zero();
                            t[(k + 2, k - 1)] = T::zero();
                        }

                        let refl = Reflection::new(Unit::new_unchecked(axis.clone()), T::zero());

                        {
                            let krows = cmp::min(k + 4, end + 1);
                            let mut work = work.rows_mut(0, krows);
                            refl.reflect(
                                &mut t.generic_view_mut((k, k), (Const::<3>, Dyn(dim.value() - k))),
                            );
                            refl.reflect_rows(
                                &mut t.generic_view_mut((0, k), (Dyn(krows), Const::<3>)),
                                &mut work,
                            );
                        }

                        if let Some(ref mut q) = q {
                            refl.reflect_rows(
                                &mut q.generic_view_mut((0, k), (dim, Const::<3>)),
                                work,
                            );
                        }
                    }

                    axis.x = t[(k + 1, k)].clone();
                    axis.y = t[(k + 2, k)].clone();

                    if k < n - 2 {
                        axis.z = t[(k + 3, k)].clone();
                    }
                }

                let mut axis = Vector2::new(axis.x.clone(), axis.y.clone());
                let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                if not_zero {
                    let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

                    t[(m, m - 1)] = norm;
                    t[(n, m - 1)] = T::zero();

                    {
                        let mut work = work.rows_mut(0, end + 1);
                        refl.reflect(
                            &mut t.generic_view_mut((m, m), (Const::<2>, Dyn(dim.value() - m))),
                        );
                        refl.reflect_rows(
                            &mut t.generic_view_mut((0, m), (Dyn(end + 1), Const::<2>)),
                            &mut work,
                        );
                    }

                    if let Some(ref mut q) = q {
                        refl.reflect_rows(&mut q.generic_view_mut((0, m), (dim, Const::<2>)), work);
                    }
                }
            } else {
                // Decouple the 2x2 block if it has real eigenvalues.
                if let Some(rot) = compute_2x2_basis(&t.fixed_view::<2, 2>(start, start)) {
                    let inv_rot = rot.inverse();
                    inv_rot.rotate(
                        &mut t.generic_view_mut(
                            (start, start),
                            (Const::<2>, Dyn(dim.value() - start)),
                        ),
                    );
                    rot.rotate_rows(
                        &mut t.generic_view_mut((0, start), (Dyn(end + 1), Const::<2>)),
                    );
                    t[(end, start)] = T::zero();

                    if let Some(ref mut q) = q {
                        rot.rotate_rows(&mut q.generic_view_mut((0, start), (dim, Const::<2>)));
                    }
                }

                // Check if we reached the beginning of the matrix.
                if end > 2 {
                    end -= 2;
                } else {
                    break;
                }
            }

            let sub = Self::delimit_subproblem(&mut t, eps.clone(), end);

            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        t.scale_mut(amax_m);

        Some((q, t))
    }

    /// Computes the eigenvalues of the decomposed matrix.
    fn do_eigenvalues(t: &OMatrix<T, D, D>, out: &mut OVector<T, D>) -> bool {
        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = t[(m, m)].clone();
                m += 1;
            } else {
                // Complex eigenvalue.
                return false;
            }
        }

        if m == dim - 1 {
            out[m] = t[(m, m)].clone();
        }

        true
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    fn do_complex_eigenvalues(t: &OMatrix<T, D, D>, out: &mut UninitVector<NumComplex<T>, D>)
    where
        T: RealField,
        DefaultAllocator: Allocator<D>,
    {
        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = MaybeUninit::new(NumComplex::new(t[(m, m)].clone(), T::zero()));
                m += 1;
            } else {
                // Solve the 2x2 eigenvalue subproblem.
                let hmm = t[(m, m)].clone();
                let hnm = t[(n, m)].clone();
                let hmn = t[(m, n)].clone();
                let hnn = t[(n, n)].clone();

                // NOTE: use the same algorithm as in compute_2x2_eigvals.
                let val = (hmm.clone() - hnn.clone()) * crate::convert(0.5);
                let discr = hnm * hmn + val.clone() * val;

                // All 2x2 blocks have negative discriminant because we already decoupled those
                // with positive eigenvalues.
                let sqrt_discr = NumComplex::new(T::zero(), (-discr).sqrt());

                let half_tra = (hnn + hmm) * crate::convert(0.5);
                out[m] = MaybeUninit::new(
                    NumComplex::new(half_tra.clone(), T::zero()) + sqrt_discr.clone(),
                );
                out[m + 1] =
                    MaybeUninit::new(NumComplex::new(half_tra, T::zero()) - sqrt_discr.clone());

                m += 2;
            }
        }

        if m == dim - 1 {
            out[m] = MaybeUninit::new(NumComplex::new(t[(m, m)].clone(), T::zero()));
        }
    }

    fn delimit_subproblem(t: &mut OMatrix<T, D, D>, eps: T::RealField, end: usize) -> (usize, usize)
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>>,
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if t[(n, m)].clone().norm1()
                <= eps.clone() * (t[(n, n)].clone().norm1() + t[(m, m)].clone().norm1())
            {
                t[(n, m)] = T::zero();
            } else {
                break;
            }

            n -= 1;
        }

        if n == 0 {
            return (0, 0);
        }

        let mut new_start = n - 1;
        while new_start > 0 {
            let m = new_start - 1;

            let off_diag = t[(new_start, m)].clone();
            if off_diag.is_zero()
                || off_diag.norm1()
                    <= eps.clone()
                        * (t[(new_start, new_start)].clone().norm1() + t[(m, m)].clone().norm1())
            {
                t[(new_start, m)] = T::zero();
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    /// Retrieves the unitary matrix `Q` and the upper-quasitriangular matrix `T` such that the
    /// decomposed matrix equals `Q * T * Q.transpose()`.
    ///
    /// # What This Returns
    ///
    /// This method consumes the Schur decomposition and returns a tuple `(Q, T)` where:
    /// - **Q**: A unitary (orthogonal for real matrices) matrix. Satisfies Q * Q<sup>H</sup> = I
    /// - **T**: An upper quasi-triangular matrix containing eigenvalue information
    ///
    /// The original matrix can be reconstructed as: **A = Q * T * Q<sup>H</sup>**
    ///
    /// # Understanding the T Matrix
    ///
    /// For real matrices, T is "quasi-triangular":
    /// - 1×1 blocks on the diagonal represent real eigenvalues
    /// - 2×2 blocks on the diagonal represent complex conjugate eigenvalue pairs
    /// - All entries below these blocks are zero
    ///
    /// For complex matrices, T is fully upper triangular with eigenvalues on the diagonal.
    ///
    /// # Example: Basic Unpacking
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 0.0,
    ///     1.0, 3.0, 0.0,
    ///     0.0, 0.0, 5.0
    /// );
    ///
    /// let schur = m.clone().schur();
    /// let (q, t) = schur.unpack();
    ///
    /// // Q is orthogonal
    /// let identity = &q * q.transpose();
    /// assert!(identity.is_identity(1e-10));
    ///
    /// // Reconstruct the original matrix
    /// let reconstructed = &q * &t * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // T is upper triangular (or quasi-triangular)
    /// println!("T matrix:\n{}", t);
    /// ```
    ///
    /// # Example: Analyzing Eigenvalues from T
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, -1.0,
    ///     1.0,  1.0
    /// );
    ///
    /// let schur = m.schur();
    /// let (q, t) = schur.unpack();
    ///
    /// // For 2x2 matrices, check if eigenvalues are real or complex
    /// let off_diagonal: f64 = t[(1, 0)];
    /// if off_diagonal.abs() < 1e-10 {
    ///     // Diagonal: real eigenvalues
    ///     println!("Real eigenvalue 1: {}", t[(0, 0)]);
    ///     println!("Real eigenvalue 2: {}", t[(1, 1)]);
    /// } else {
    ///     // 2x2 block: complex eigenvalues
    ///     println!("Matrix has complex eigenvalues");
    /// }
    /// ```
    ///
    /// # Example: Using Q for Coordinate Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 4.0
    /// );
    ///
    /// let schur = m.schur();
    /// let (q, t) = schur.unpack();
    ///
    /// // Transform a vector to the Schur basis
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let v_transformed = q.transpose() * v;
    ///
    /// // Apply the matrix in Schur form (often simpler)
    /// let result_transformed = t * v_transformed;
    ///
    /// // Transform back to original basis
    /// let result = q * result_transformed;
    ///
    /// // This equals m * v
    /// assert!(result.relative_eq(&(m * v), 1e-10, 1e-10));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`eigenvalues`](#method.eigenvalues): Extract real eigenvalues directly
    /// - [`complex_eigenvalues`](#method.complex_eigenvalues): Extract all eigenvalues as complex numbers
    /// - [`new`](#method.new): Create a Schur decomposition
    pub fn unpack(self) -> (OMatrix<T, D, D>, OMatrix<T, D, D>) {
        (self.q, self.t)
    }

    /// Computes the real eigenvalues of the decomposed matrix.
    ///
    /// # What This Returns
    ///
    /// Returns a vector containing all eigenvalues if they are all real, or `None` if any
    /// eigenvalues are complex. For real matrices, complex eigenvalues come in conjugate pairs,
    /// so if this returns `None`, use [`complex_eigenvalues`](#method.complex_eigenvalues) instead.
    ///
    /// # Understanding Eigenvalues
    ///
    /// Eigenvalues are scalar values λ that satisfy **A·v = λ·v** for some non-zero vector v.
    /// They represent:
    /// - Growth/decay rates in dynamical systems
    /// - Frequencies in vibration analysis
    /// - Stability indicators in control theory
    /// - Principal components in data analysis
    ///
    /// # When Eigenvalues Are Real
    ///
    /// Eigenvalues are guaranteed to be real for:
    /// - Symmetric matrices (A = A<sup>T</sup>)
    /// - Hermitian matrices (A = A<sup>H</sup>)
    /// - Some special non-symmetric matrices
    ///
    /// # Example: Symmetric Matrix (Always Real Eigenvalues)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Symmetric matrix has real eigenvalues
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 0.0,
    ///     1.0, 3.0, 2.0,
    ///     0.0, 2.0, 5.0
    /// );
    ///
    /// let schur = m.schur();
    /// let eigenvalues = schur.eigenvalues().expect("Symmetric matrices have real eigenvalues");
    ///
    /// println!("Eigenvalues: {}", eigenvalues);
    /// // All eigenvalues are real numbers
    /// ```
    ///
    /// # Example: Checking for Complex Eigenvalues
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Rotation matrix has complex eigenvalues (except for 0° and 180°)
    /// let angle = std::f64::consts::PI / 4.0; // 45 degrees
    /// let m = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos()
    /// );
    ///
    /// let schur = m.schur();
    /// match schur.eigenvalues() {
    ///     Some(real_eigenvalues) => {
    ///         println!("All eigenvalues are real: {}", real_eigenvalues);
    ///     },
    ///     None => {
    ///         println!("Matrix has complex eigenvalues");
    ///         // Use complex_eigenvalues() instead
    ///         let complex_eigs = schur.complex_eigenvalues();
    ///         println!("Complex eigenvalues: {}", complex_eigs);
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Stability Analysis
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Linear system: dx/dt = A*x
    /// let a = Matrix2::new(
    ///     -1.0,  0.5,
    ///      0.0, -2.0
    /// );
    ///
    /// let schur = a.schur();
    /// if let Some(eigenvalues) = schur.eigenvalues() {
    ///     // System is stable if all eigenvalues have negative real parts
    ///     let is_stable = eigenvalues.iter().all(|&ev| ev < 0.0);
    ///     println!("System is stable: {}", is_stable);
    ///
    ///     if is_stable {
    ///         // The largest (least negative) eigenvalue determines decay rate
    ///         let decay_rate = eigenvalues.max();
    ///         println!("Decay rate: {}", decay_rate);
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Finding Dominant Eigenvalue
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 3.0
    /// );
    ///
    /// let schur = m.schur();
    /// if let Some(eigenvalues) = schur.eigenvalues() {
    ///     // Dominant eigenvalue (largest magnitude)
    ///     let mut max_abs = 0.0_f64;
    ///     for i in 0..eigenvalues.len() {
    ///         let ev: f64 = eigenvalues[i];
    ///         let abs_ev = ev.abs();
    ///         if abs_ev > max_abs {
    ///             max_abs = abs_ev;
    ///         }
    ///     }
    ///     println!("Dominant eigenvalue magnitude: {}", max_abs);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`complex_eigenvalues`](#method.complex_eigenvalues): Get all eigenvalues as complex numbers
    /// - [`unpack`](#method.unpack): Get the Q and T matrices to inspect eigenvalues directly
    /// - [`SquareMatrix::eigenvalues`](../struct.Matrix.html#method.eigenvalues): Compute eigenvalues without full decomposition
    ///
    /// Return `None` if some eigenvalues are complex.
    #[must_use]
    pub fn eigenvalues(&self) -> Option<OVector<T, D>> {
        let mut out = Matrix::zeros_generic(self.t.shape_generic().0, Const::<1>);
        if Self::do_eigenvalues(&self.t, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    ///
    /// # What This Returns
    ///
    /// Returns a vector containing all eigenvalues as complex numbers. This always succeeds,
    /// unlike [`eigenvalues`](#method.eigenvalues) which returns `None` for matrices with
    /// complex eigenvalues. Real eigenvalues are returned with zero imaginary part.
    ///
    /// # Understanding Complex Eigenvalues
    ///
    /// Complex eigenvalues appear in real matrices as conjugate pairs (a ± bi). They represent:
    /// - **Oscillatory behavior** in dynamical systems
    /// - **Rotation components** in transformations
    /// - **Periodic solutions** in differential equations
    /// - **Resonance frequencies** in mechanical systems
    ///
    /// The magnitude |λ| = √(a² + b²) determines the growth/decay rate, while the argument
    /// (angle) determines the frequency of oscillation.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1.0, -2.0,
    ///     2.0,  1.0
    /// );
    ///
    /// let schur = m.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// println!("Eigenvalues:");
    /// for (i, ev) in eigenvalues.iter().enumerate() {
    ///     println!("  λ_{} = {} + {}i", i, ev.re, ev.im);
    /// }
    ///
    /// // Complex eigenvalues come in conjugate pairs for real matrices
    /// let sum_im: f64 = eigenvalues[0].im + eigenvalues[1].im;
    /// assert!(sum_im.abs() < 1e-10);
    /// ```
    ///
    /// # Example: Rotation Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    /// use std::f64::consts::PI;
    ///
    /// // 90-degree rotation matrix
    /// let angle = PI / 2.0;
    /// let rotation = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos()
    /// );
    ///
    /// let schur = rotation.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// // Rotation matrices have eigenvalues on the unit circle
    /// for ev in eigenvalues.iter() {
    ///     let re: f64 = ev.re;
    ///     let im: f64 = ev.im;
    ///     let magnitude = (re * re + im * im).sqrt();
    ///     assert!((magnitude - 1.0).abs() < 1e-10);
    /// }
    /// ```
    ///
    /// # Example: Oscillatory System Stability
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Damped oscillator: dx/dt = A*x
    /// let m = Matrix2::new(
    ///     -0.1,  1.0,  // Damping = -0.1
    ///     -1.0, -0.1   // Spring constant = 1.0
    /// );
    ///
    /// let schur = m.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// for ev in eigenvalues.iter() {
    ///     println!("Eigenvalue: {} + {}i", ev.re, ev.im);
    ///
    ///     // Negative real part => stable (decaying oscillations)
    ///     if ev.re < 0.0 {
    ///         println!("  Stable with decay rate: {}", -ev.re);
    ///         let im_part: f64 = ev.im;
    ///         if im_part.abs() > 1e-10 {
    ///             let frequency = im_part.abs();
    ///             println!("  Oscillation frequency: {}", frequency);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Checking Stability (Control Theory)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // State space model matrix
    /// let a = Matrix3::new(
    ///     -1.0,  2.0,  0.0,
    ///      0.0, -0.5,  1.0,
    ///      0.0, -1.0, -0.5
    /// );
    ///
    /// let schur = a.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// // Continuous-time system: stable if all real parts < 0
    /// let is_stable = eigenvalues.iter().all(|ev| ev.re < 0.0);
    /// println!("Continuous system is stable: {}", is_stable);
    ///
    /// // Find eigenvalue closest to imaginary axis (most critical)
    /// let critical = eigenvalues.iter()
    ///     .max_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
    ///     .unwrap();
    /// println!("Critical eigenvalue: {} + {}i", critical.re, critical.im);
    /// ```
    ///
    /// # Example: Discrete System Stability
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Discrete-time system: x[k+1] = A * x[k]
    /// let a = Matrix2::new(
    ///     0.8,  0.3,
    ///     -0.2, 0.9
    /// );
    ///
    /// let schur = a.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// // Discrete-time system: stable if all magnitudes < 1
    /// let mut is_stable = true;
    /// for ev in eigenvalues.iter() {
    ///     let re: f64 = ev.re;
    ///     let im: f64 = ev.im;
    ///     let magnitude = (re * re + im * im).sqrt();
    ///     if magnitude >= 1.0 {
    ///         is_stable = false;
    ///         break;
    ///     }
    /// }
    ///
    /// println!("Discrete system is stable: {}", is_stable);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`eigenvalues`](#method.eigenvalues): Get real eigenvalues only (faster if applicable)
    /// - [`unpack`](#method.unpack): Access the Schur form matrices directly
    /// - [`SquareMatrix::complex_eigenvalues`](../struct.Matrix.html#method.complex_eigenvalues): Compute without creating full Schur decomposition
    #[must_use]
    pub fn complex_eigenvalues(&self) -> OVector<NumComplex<T>, D>
    where
        T: RealField,
        DefaultAllocator: Allocator<D>,
    {
        let mut out = Matrix::uninit(self.t.shape_generic().0, Const::<1>);
        Self::do_complex_eigenvalues(&self.t, &mut out);
        // Safety: out has been fully initialized by do_complex_eigenvalues.
        unsafe { out.assume_init() }
    }
}

fn decompose_2x2<T: ComplexField, D: Dim>(
    mut m: OMatrix<T, D, D>,
    compute_q: bool,
) -> Option<(Option<OMatrix<T, D, D>>, OMatrix<T, D, D>)>
where
    DefaultAllocator: Allocator<D, D>,
{
    let dim = m.shape_generic().0;
    let mut q = None;
    match compute_2x2_basis(&m.fixed_view::<2, 2>(0, 0)) {
        Some(rot) => {
            let mut m = m.fixed_view_mut::<2, 2>(0, 0);
            let inv_rot = rot.inverse();
            inv_rot.rotate(&mut m);
            rot.rotate_rows(&mut m);
            m[(1, 0)] = T::zero();

            if compute_q {
                // XXX: we have to build the matrix manually because
                // rot.to_rotation_matrix().unwrap() causes an ICE.
                let c = T::from_real(rot.c());
                q = Some(OMatrix::from_column_slice_generic(
                    dim,
                    dim,
                    &[c.clone(), rot.s(), -rot.s().conjugate(), c],
                ));
            }
        }
        None => {
            if compute_q {
                q = Some(OMatrix::identity_generic(dim, dim));
            }
        }
    };

    Some((q, m))
}

fn compute_2x2_eigvals<T: ComplexField, S: Storage<T, U2, U2>>(
    m: &SquareMatrix<T, U2, S>,
) -> Option<(T, T)> {
    // Solve the 2x2 eigenvalue subproblem.
    let h00 = m[(0, 0)].clone();
    let h10 = m[(1, 0)].clone();
    let h01 = m[(0, 1)].clone();
    let h11 = m[(1, 1)].clone();

    // NOTE: this discriminant computation is more stable than the
    // one based on the trace and determinant: 0.25 * tra * tra - det
    // because it ensures positiveness for symmetric matrices.
    let val = (h00.clone() - h11.clone()) * crate::convert(0.5);
    let discr = h10 * h01 + val.clone() * val;

    discr.try_sqrt().map(|sqrt_discr| {
        let half_tra = (h00 + h11) * crate::convert(0.5);
        (half_tra.clone() + sqrt_discr.clone(), half_tra - sqrt_discr)
    })
}

// Computes the 2x2 transformation that upper-triangulates a 2x2 matrix with real eigenvalues.
/// Computes the singular vectors for a 2x2 matrix.
///
/// Returns `None` if the matrix has complex eigenvalues, or is upper-triangular. In both case,
/// the basis is the identity.
fn compute_2x2_basis<T: ComplexField, S: Storage<T, U2, U2>>(
    m: &SquareMatrix<T, U2, S>,
) -> Option<GivensRotation<T>> {
    let h10 = m[(1, 0)].clone();

    if h10.is_zero() {
        return None;
    }

    let (eigval1, eigval2) = compute_2x2_eigvals(m)?;
    let x1 = eigval1 - m[(1, 1)].clone();
    let x2 = eigval2 - m[(1, 1)].clone();

    // NOTE: Choose the one that yields a larger x component.
    // This is necessary for numerical stability of the normalization of the complex
    // number.
    if x1.clone().norm1() > x2.clone().norm1() {
        Some(GivensRotation::new(x1, h10).0)
    } else {
        Some(GivensRotation::new(x2, h10).0)
    }
}

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    D: DimSub<U1>, // For Hessenberg.
    DefaultAllocator:
        Allocator<D, DimDiff<D, U1>> + Allocator<DimDiff<D, U1>> + Allocator<D, D> + Allocator<D>,
{
    /// Computes the eigenvalues of this matrix.
    ///
    /// This is a convenience method that computes eigenvalues directly without requiring you
    /// to first create a Schur decomposition. It returns real eigenvalues only.
    ///
    /// # What This Does
    ///
    /// Computes all eigenvalues λ that satisfy **A·v = λ·v**. Returns `Some(eigenvalues)` if
    /// all eigenvalues are real, or `None` if any are complex. For real symmetric matrices,
    /// all eigenvalues are guaranteed to be real.
    ///
    /// This method is more efficient than computing the full Schur decomposition if you only
    /// need the eigenvalues and not the transformation matrices.
    ///
    /// # Understanding Eigenvalues
    ///
    /// Eigenvalues reveal fundamental properties of linear transformations:
    /// - **Scaling factors** along principal directions
    /// - **Stability** of dynamical systems
    /// - **Resonance frequencies** in physical systems
    /// - **Data variance** in principal component analysis
    ///
    /// # Returns
    ///
    /// - `Some(vector)` containing all eigenvalues if they are all real
    /// - `None` if any eigenvalues are complex (use [`complex_eigenvalues`](#method.complex_eigenvalues) instead)
    ///
    /// # Example: Symmetric Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Symmetric matrices always have real eigenvalues
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 2.0,
    ///     1.0, 5.0, 3.0,
    ///     2.0, 3.0, 6.0
    /// );
    ///
    /// let eigenvalues = m.eigenvalues().expect("Symmetric matrix has real eigenvalues");
    /// println!("Eigenvalues: {}", eigenvalues);
    ///
    /// // The sum of eigenvalues equals the trace
    /// let trace = m[(0,0)] + m[(1,1)] + m[(2,2)];
    /// let sum_eigenvalues: f64 = eigenvalues.iter().sum();
    /// assert!((trace - sum_eigenvalues).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Checking for Complex Eigenvalues
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     0.0, -1.0,
    ///     1.0,  0.0  // 90-degree rotation
    /// );
    ///
    /// match m.eigenvalues() {
    ///     Some(real_eigs) => {
    ///         println!("All real: {}", real_eigs);
    ///     },
    ///     None => {
    ///         println!("Has complex eigenvalues");
    ///         // Use complex_eigenvalues() for this matrix
    ///         let complex_eigs = m.complex_eigenvalues();
    ///         println!("Complex eigenvalues: {}", complex_eigs);
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Stability Analysis of a Continuous System
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Linear dynamical system: dx/dt = A*x
    /// let a = Matrix2::new(
    ///     -2.0,  1.0,
    ///      0.0, -3.0
    /// );
    ///
    /// if let Some(eigenvalues) = a.eigenvalues() {
    ///     // System is stable if all eigenvalues are negative
    ///     let is_stable = eigenvalues.iter().all(|&lambda| lambda < 0.0);
    ///
    ///     if is_stable {
    ///         println!("System is stable!");
    ///         // The decay rate is determined by the largest (least negative) eigenvalue
    ///         let slowest_decay = eigenvalues.max();
    ///         println!("Slowest decay rate: {}", slowest_decay);
    ///     } else {
    ///         println!("System is unstable!");
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Discrete System Stability
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Discrete system: x[k+1] = A * x[k]
    /// let a = Matrix2::new(
    ///     0.9, 0.1,
    ///     0.0, 0.8
    /// );
    ///
    /// if let Some(eigenvalues) = a.eigenvalues() {
    ///     // Discrete system is stable if all |eigenvalues| < 1
    ///     let mut is_stable = true;
    ///     for i in 0..eigenvalues.len() {
    ///         let lambda: f64 = eigenvalues[i];
    ///         if lambda.abs() >= 1.0 {
    ///             is_stable = false;
    ///             break;
    ///         }
    ///     }
    ///     println!("Discrete system is stable: {}", is_stable);
    /// }
    /// ```
    ///
    /// # Example: Finding Spectral Radius
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     5.0, 1.0, 0.0,
    ///     1.0, 4.0, 1.0,
    ///     0.0, 1.0, 3.0
    /// );
    ///
    /// if let Some(eigenvalues) = m.eigenvalues() {
    ///     // Spectral radius: maximum absolute value of eigenvalues
    ///     let mut max_abs = 0.0_f64;
    ///     for i in 0..eigenvalues.len() {
    ///         let ev: f64 = eigenvalues[i];
    ///         let abs_ev = ev.abs();
    ///         if abs_ev > max_abs {
    ///             max_abs = abs_ev;
    ///         }
    ///     }
    ///
    ///     println!("Spectral radius: {}", max_abs);
    ///
    ///     // Spectral radius determines convergence of iterative methods
    ///     if max_abs < 1.0 {
    ///         println!("Iterative methods will converge");
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`complex_eigenvalues`](#method.complex_eigenvalues): Get all eigenvalues as complex numbers
    /// - [`schur`](struct.Schur.html#method.new): Compute full Schur decomposition
    /// - [`symmetric_eigen`](struct.SymmetricEigen.html): More efficient for symmetric matrices
    #[must_use]
    pub fn eigenvalues(&self) -> Option<OVector<T, D>> {
        assert!(
            self.is_square(),
            "Unable to compute eigenvalues of a non-square matrix."
        );

        let mut work = Matrix::zeros_generic(self.shape_generic().0, Const::<1>);

        // Special case for 2x2 matrices.
        if self.nrows() == 2 {
            // TODO: can we avoid this slicing
            // (which is needed here just to transform D to U2)?
            let me = self.fixed_view::<2, 2>(0, 0);
            return match compute_2x2_eigvals(&me) {
                Some((a, b)) => {
                    work[0] = a;
                    work[1] = b;
                    Some(work)
                }
                None => None,
            };
        }

        // TODO: add balancing?
        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            T::RealField::default_epsilon(),
            0,
            false,
        )
        .unwrap();

        if Schur::do_eigenvalues(&schur.1, &mut work) {
            Some(work)
        } else {
            None
        }
    }

    /// Computes the eigenvalues of this matrix.
    ///
    /// This is a convenience method that computes all eigenvalues as complex numbers directly,
    /// without requiring you to first create a Schur decomposition. Unlike [`eigenvalues`](#method.eigenvalues),
    /// this always succeeds and returns complex numbers (with real eigenvalues having zero imaginary part).
    ///
    /// # What This Does
    ///
    /// Computes all eigenvalues λ (complex numbers) that satisfy **A·v = λ·v**. This method
    /// always returns a result, representing real eigenvalues as complex numbers with zero
    /// imaginary part.
    ///
    /// This is more efficient than computing the full Schur decomposition if you only need
    /// eigenvalues and not the transformation matrices.
    ///
    /// # Understanding Complex Eigenvalues
    ///
    /// Complex eigenvalues in real matrices always appear as conjugate pairs (a ± bi):
    /// - The **real part (a)** indicates growth/decay rate
    /// - The **imaginary part (b)** indicates oscillation frequency
    /// - The **magnitude** |λ| = √(a² + b²) gives overall growth rate
    /// - The **argument** (angle) gives rotation rate
    ///
    /// # Example: General Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1.0,  2.0,
    ///     -1.0, 3.0
    /// );
    ///
    /// let eigenvalues = m.complex_eigenvalues();
    ///
    /// for (i, ev) in eigenvalues.iter().enumerate() {
    ///     println!("λ_{} = {} + {}i", i, ev.re, ev.im);
    ///     let re: f64 = ev.re;
    ///     let im: f64 = ev.im;
    ///     let magnitude = (re * re + im * im).sqrt();
    ///     println!("  Magnitude: {}", magnitude);
    /// }
    /// ```
    ///
    /// # Example: Rotation Matrix Analysis
    ///
    /// ```
    /// use nalgebra::Matrix2;
    /// use std::f64::consts::PI;
    ///
    /// // Rotation by 60 degrees
    /// let angle = PI / 3.0;
    /// let rotation = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos()
    /// );
    ///
    /// let eigenvalues = rotation.complex_eigenvalues();
    ///
    /// // Rotation matrices have complex eigenvalues on the unit circle
    /// for ev in eigenvalues.iter() {
    ///     let re: f64 = ev.re;
    ///     let im: f64 = ev.im;
    ///     let magnitude = (re * re + im * im).sqrt();
    ///     assert!((magnitude - 1.0).abs() < 1e-10);
    ///
    ///     // The angle can be recovered from the eigenvalue
    ///     let recovered_angle = im.atan2(re);
    ///     println!("Rotation angle from eigenvalue: {} radians", recovered_angle.abs());
    /// }
    /// ```
    ///
    /// # Example: Damped Oscillator
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Mass-spring-damper system: d²x/dt² + 2ζωₙ dx/dt + ωₙ²x = 0
    /// // Converted to state space form
    /// let damping_ratio = 0.2;  // Underdamped
    /// let natural_freq = 2.0;
    ///
    /// let a = Matrix2::new(
    ///     0.0,                          1.0,
    ///     -natural_freq * natural_freq, -2.0 * damping_ratio * natural_freq
    /// );
    ///
    /// let eigenvalues = a.complex_eigenvalues();
    ///
    /// for ev in eigenvalues.iter() {
    ///     let im_part: f64 = ev.im;
    ///     if im_part.abs() > 1e-10 {
    ///         println!("Underdamped oscillation:");
    ///         println!("  Decay rate: {}", -ev.re);
    ///         println!("  Oscillation frequency: {} rad/s", im_part.abs());
    ///         println!("  Period: {} s", 2.0 * std::f64::consts::PI / im_part.abs());
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Stability Analysis
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Linearized dynamics around an equilibrium point
    /// let a = Matrix3::new(
    ///     -0.5,  1.0,  0.0,
    ///      0.0, -1.0,  2.0,
    ///     -1.0,  0.0, -0.5
    /// );
    ///
    /// let eigenvalues = a.complex_eigenvalues();
    ///
    /// // Check stability: all real parts must be negative
    /// let is_stable = eigenvalues.iter().all(|ev| ev.re < 0.0);
    /// println!("System is stable: {}", is_stable);
    ///
    /// // Find the most critical eigenvalue (closest to instability)
    /// let critical = eigenvalues.iter()
    ///     .max_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
    ///     .unwrap();
    ///
    /// println!("Critical eigenvalue: {} + {}i", critical.re, critical.im);
    /// println!("Stability margin: {}", -critical.re);
    /// ```
    ///
    /// # Example: Discrete-Time System
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Discrete system: x[k+1] = A * x[k]
    /// let a = Matrix2::new(
    ///     0.7,  0.3,
    ///     -0.4, 0.8
    /// );
    ///
    /// let eigenvalues = a.complex_eigenvalues();
    ///
    /// // Discrete system is stable if all |λ| < 1
    /// let mut is_stable = true;
    /// for ev in eigenvalues.iter() {
    ///     let re: f64 = ev.re;
    ///     let im: f64 = ev.im;
    ///     let magnitude = (re * re + im * im).sqrt();
    ///     println!("Eigenvalue magnitude: {}", magnitude);
    ///
    ///     if magnitude < 1.0 {
    ///         println!("  Stable mode (decays)");
    ///     } else if magnitude > 1.0 {
    ///         println!("  Unstable mode (grows)");
    ///         is_stable = false;
    ///     } else {
    ///         println!("  Marginally stable (persists)");
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Comparing with Real Eigenvalues
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     0.0, 2.0  // Upper triangular - real eigenvalues
    /// );
    ///
    /// // Try real first
    /// match m.eigenvalues() {
    ///     Some(real_eigs) => {
    ///         println!("Real eigenvalues: {}", real_eigs);
    ///     },
    ///     None => {
    ///         // Fall back to complex
    ///         let complex_eigs = m.complex_eigenvalues();
    ///         println!("Complex eigenvalues: {}", complex_eigs);
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`eigenvalues`](#method.eigenvalues): Get real eigenvalues only (returns None if complex)
    /// - [`schur`](struct.Schur.html#method.new): Compute full Schur decomposition for more analysis
    /// - [`symmetric_eigen`](struct.SymmetricEigen.html): Specialized for symmetric matrices
    #[must_use]
    pub fn complex_eigenvalues(&self) -> OVector<NumComplex<T>, D>
    // TODO: add balancing?
    where
        T: RealField,
        DefaultAllocator: Allocator<D>,
    {
        let dim = self.shape_generic().0;
        let mut work = Matrix::zeros_generic(dim, Const::<1>);

        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            T::default_epsilon(),
            0,
            false,
        )
        .unwrap();
        let mut eig = Matrix::uninit(dim, Const::<1>);
        Schur::do_complex_eigenvalues(&schur.1, &mut eig);
        // Safety: eig has been fully initialized by do_complex_eigenvalues.
        unsafe { eig.assume_init() }
    }
}
