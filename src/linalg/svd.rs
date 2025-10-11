#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};
use std::any::TypeId;

use approx::AbsDiffEq;
use num::{One, Zero};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, Matrix2x3, OMatrix, OVector, Vector2};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimDiff, DimMin, DimMinimum, DimSub, U1};
use crate::storage::Storage;
use crate::{Matrix2, Matrix3, RawStorage, U2, U3};
use simba::scalar::{ComplexField, RealField};

use crate::linalg::Bidiagonal;
use crate::linalg::givens::GivensRotation;
use crate::linalg::symmetric_eigen;

/// Singular Value Decomposition (SVD) of a general matrix.
///
/// # What is SVD?
///
/// Singular Value Decomposition is a fundamental matrix factorization technique that decomposes
/// any matrix `A` (m × n) into three matrices:
///
/// ```text
/// A = U * Σ * V^T
/// ```
///
/// Where:
/// - `U` is an m × min(m,n) matrix with orthonormal columns (left-singular vectors)
/// - `Σ` is a min(m,n) × min(m,n) diagonal matrix with non-negative real numbers (singular values)
/// - `V^T` is the transpose of an n × min(m,n) matrix with orthonormal columns (right-singular vectors)
///
/// # Applications
///
/// SVD is incredibly useful for many applications:
///
/// - **Dimensionality Reduction**: Keep only the largest singular values to approximate data
///   with fewer dimensions (similar to PCA)
/// - **Pseudo-inverse**: Compute the Moore-Penrose pseudo-inverse for solving least-squares problems
/// - **Matrix Rank**: Determine the rank by counting non-zero singular values
/// - **Data Compression**: Approximate matrices with fewer components (e.g., image compression)
/// - **Noise Reduction**: Remove small singular values that often represent noise
/// - **Solving Linear Systems**: Solve overdetermined or underdetermined systems robustly
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix3, SVD};
///
/// let m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// // Compute the full SVD with both U and V^T
/// let svd = SVD::new(m, true, true);
///
/// // Access the singular values
/// println!("Singular values: {:?}", svd.singular_values);
///
/// // Reconstruct the original matrix
/// let reconstructed = svd.recompose().unwrap();
/// ```
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<DimMinimum<R, C>>    +
                           Allocator<DimMinimum<R, C>, C> +
                           Allocator<R, DimMinimum<R, C>>,
         OMatrix<T, R, DimMinimum<R, C>>: Serialize,
         OMatrix<T, DimMinimum<R, C>, C>: Serialize,
         OVector<T::RealField, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<DimMinimum<R, C>> +
                           Allocator<DimMinimum<R, C>, C> +
                           Allocator<R, DimMinimum<R, C>>,
         OMatrix<T, R, DimMinimum<R, C>>: Deserialize<'de>,
         OMatrix<T, DimMinimum<R, C>, C>: Deserialize<'de>,
         OVector<T::RealField, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct SVD<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>,
{
    /// The left-singular vectors `U` of this SVD.
    pub u: Option<OMatrix<T, R, DimMinimum<R, C>>>,
    /// The right-singular vectors `V^t` of this SVD.
    pub v_t: Option<OMatrix<T, DimMinimum<R, C>, C>>,
    /// The singular values of this SVD.
    pub singular_values: OVector<T::RealField, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for SVD<T, R, C>
where
    DefaultAllocator: Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, DimMinimum<R, C>>: Copy,
    OMatrix<T, DimMinimum<R, C>, C>: Copy,
    OVector<T::RealField, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> SVD<T, R, C>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
{
    fn use_special_always_ordered_svd2() -> bool {
        TypeId::of::<OMatrix<T, R, C>>() == TypeId::of::<Matrix2<T::RealField>>()
            && TypeId::of::<Self>() == TypeId::of::<SVD<T::RealField, U2, U2>>()
    }

    fn use_special_always_ordered_svd3() -> bool {
        TypeId::of::<OMatrix<T, R, C>>() == TypeId::of::<Matrix3<T::RealField>>()
            && TypeId::of::<Self>() == TypeId::of::<SVD<T::RealField, U3, U3>>()
    }

    /// Computes the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// This function computes the SVD decomposition where `A = U * Σ * V^T`, but the singular
    /// values in `Σ` are **not guaranteed to be sorted**. This can be faster than [`SVD::new`]
    /// when you don't need the singular values in any particular order.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The matrix to decompose
    /// * `compute_u` - Set to `true` to compute the left-singular vectors matrix `U`
    /// * `compute_v` - Set to `true` to compute the right-singular vectors matrix `V^T`
    ///
    /// # Performance
    ///
    /// If you don't need the singular vectors, set both `compute_u` and `compute_v` to `false`
    /// for better performance. If you need sorted singular values, use [`SVD::new`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// // Compute SVD without singular vectors (fastest)
    /// let svd = nalgebra::SVD::new_unordered(m.clone(), false, false);
    /// println!("Singular values: {}", svd.singular_values);
    ///
    /// // Compute full SVD with U and V^T
    /// let svd_full = nalgebra::SVD::new_unordered(m, true, true);
    /// let u = svd_full.u.unwrap();
    /// let v_t = svd_full.v_t.unwrap();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`SVD::new`] - Computes SVD with singular values sorted in descending order
    /// * [`SVD::try_new_unordered`] - Fallible version with configurable tolerance and iteration limit
    /// * [`Matrix::singular_values_unordered`] - Convenience method to compute only singular values
    pub fn new_unordered(matrix: OMatrix<T, R, C>, compute_u: bool, compute_v: bool) -> Self {
        Self::try_new_unordered(
            matrix,
            compute_u,
            compute_v,
            T::RealField::default_epsilon() * crate::convert(5.0),
            0,
        )
        .unwrap()
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// This is the fallible version of [`SVD::new_unordered`] that allows you to control the
    /// convergence tolerance and maximum number of iterations. The singular values are **not
    /// guaranteed to be sorted** in any particular order.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The matrix to decompose
    /// * `compute_u` - Set to `true` to compute the left-singular vectors matrix `U`
    /// * `compute_v` - Set to `true` to compute the right-singular vectors matrix `V^T`
    /// * `eps` - Convergence tolerance. Values smaller than this are considered to be zero.
    ///   A typical value is `T::default_epsilon() * 5.0`
    /// * `max_niter` - Maximum number of iterations. Returns `None` if exceeded.
    ///   If `max_niter == 0`, the algorithm runs until convergence without a limit.
    ///
    /// # Returns
    ///
    /// * `Some(SVD)` - If the algorithm converges successfully
    /// * `None` - If the maximum number of iterations is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 3.0,
    /// );
    ///
    /// // Try to compute SVD with custom tolerance and iteration limit
    /// let eps = 1e-10;
    /// let max_iter = 100;
    ///
    /// match nalgebra::SVD::try_new_unordered(m, true, true, eps, max_iter) {
    ///     Some(svd) => {
    ///         println!("SVD computed successfully!");
    ///         println!("Singular values: {}", svd.singular_values);
    ///     }
    ///     None => {
    ///         println!("SVD failed to converge within {} iterations", max_iter);
    ///     }
    /// }
    /// ```
    ///
    /// # When to Use
    ///
    /// Use this function when:
    /// - You need fine control over convergence criteria
    /// - You want to limit computation time with `max_niter`
    /// - You're working with matrices that may be ill-conditioned
    /// - You need to handle convergence failures gracefully
    ///
    /// # See Also
    ///
    /// * [`SVD::new_unordered`] - Infallible version with default parameters
    /// * [`SVD::try_new`] - Sorted version with configurable parameters
    /// * [`SVD::new`] - Standard SVD with sorted singular values
    pub fn try_new_unordered(
        mut matrix: OMatrix<T, R, C>,
        compute_u: bool,
        compute_v: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<Self> {
        assert!(
            !matrix.is_empty(),
            "Cannot compute the SVD of an empty matrix."
        );
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if Self::use_special_always_ordered_svd2() {
            // SAFETY: the reference transmutes are OK since we checked that the types match exactly.
            let matrix: &Matrix2<T::RealField> = unsafe { std::mem::transmute(&matrix) };
            let result = super::svd2::svd_ordered2(matrix, compute_u, compute_v);
            let typed_result: &Self = unsafe { std::mem::transmute(&result) };
            return Some(typed_result.clone());
        } else if Self::use_special_always_ordered_svd3() {
            // SAFETY: the reference transmutes are OK since we checked that the types match exactly.
            let matrix: &Matrix3<T::RealField> = unsafe { std::mem::transmute(&matrix) };
            let result = super::svd3::svd_ordered3(matrix, compute_u, compute_v, eps, max_niter);
            let typed_result: &Self = unsafe { std::mem::transmute(&result) };
            return Some(typed_result.clone());
        }

        let dim = min_nrows_ncols.value();

        let m_amax = matrix.camax();

        if !m_amax.is_zero() {
            matrix.unscale_mut(m_amax.clone());
        }

        let bi_matrix = Bidiagonal::new(matrix);
        let mut u = if compute_u { Some(bi_matrix.u()) } else { None };
        let mut v_t = if compute_v {
            Some(bi_matrix.v_t())
        } else {
            None
        };
        let mut diagonal = bi_matrix.diagonal();
        let mut off_diagonal = bi_matrix.off_diagonal();

        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(
            &mut diagonal,
            &mut off_diagonal,
            &mut u,
            &mut v_t,
            bi_matrix.is_upper_diagonal(),
            dim - 1,
            eps.clone(),
        );

        while end != start {
            let subdim = end - start + 1;

            // Solve the subproblem.
            #[allow(clippy::comparison_chain)]
            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut vec;
                {
                    let dm = diagonal[m].clone();
                    let dn = diagonal[n].clone();
                    let fm = off_diagonal[m].clone();

                    let tmm = dm.clone() * dm.clone()
                        + off_diagonal[m - 1].clone() * off_diagonal[m - 1].clone();
                    let tmn = dm * fm.clone();
                    let tnn = dn.clone() * dn + fm.clone() * fm;

                    let shift = symmetric_eigen::wilkinson_shift(tmm, tnn, tmn);

                    vec = Vector2::new(
                        diagonal[start].clone() * diagonal[start].clone() - shift,
                        diagonal[start].clone() * off_diagonal[start].clone(),
                    );
                }

                for k in start..n {
                    let m12 = if k == n - 1 {
                        T::RealField::zero()
                    } else {
                        off_diagonal[k + 1].clone()
                    };

                    let mut subm = Matrix2x3::new(
                        diagonal[k].clone(),
                        off_diagonal[k].clone(),
                        T::RealField::zero(),
                        T::RealField::zero(),
                        diagonal[k + 1].clone(),
                        m12,
                    );

                    match GivensRotation::cancel_y(&vec) {
                        Some((rot1, norm1)) => {
                            rot1.inverse()
                                .rotate_rows(&mut subm.fixed_columns_mut::<2>(0));
                            let rot1 =
                                GivensRotation::new_unchecked(rot1.c(), T::from_real(rot1.s()));

                            if k > start {
                                // This is not the first iteration.
                                off_diagonal[k - 1] = norm1;
                            }

                            let v = Vector2::new(subm[(0, 0)].clone(), subm[(1, 0)].clone());
                            // TODO: does the case `v.y == 0` ever happen?
                            let (rot2, norm2) = GivensRotation::cancel_y(&v)
                                .unwrap_or((GivensRotation::identity(), subm[(0, 0)].clone()));

                            rot2.rotate(&mut subm.fixed_columns_mut::<2>(1));
                            let rot2 =
                                GivensRotation::new_unchecked(rot2.c(), T::from_real(rot2.s()));

                            subm[(0, 0)] = norm2;

                            if let Some(ref mut v_t) = v_t {
                                if bi_matrix.is_upper_diagonal() {
                                    rot1.rotate(&mut v_t.fixed_rows_mut::<2>(k));
                                } else {
                                    rot2.rotate(&mut v_t.fixed_rows_mut::<2>(k));
                                }
                            }

                            if let Some(ref mut u) = u {
                                if bi_matrix.is_upper_diagonal() {
                                    rot2.inverse().rotate_rows(&mut u.fixed_columns_mut::<2>(k));
                                } else {
                                    rot1.inverse().rotate_rows(&mut u.fixed_columns_mut::<2>(k));
                                }
                            }

                            diagonal[k] = subm[(0, 0)].clone();
                            diagonal[k + 1] = subm[(1, 1)].clone();
                            off_diagonal[k] = subm[(0, 1)].clone();

                            if k != n - 1 {
                                off_diagonal[k + 1] = subm[(1, 2)].clone();
                            }

                            vec.x = subm[(0, 1)].clone();
                            vec.y = subm[(0, 2)].clone();
                        }
                        None => {
                            break;
                        }
                    }
                }
            } else if subdim == 2 {
                // Solve the remaining 2x2 subproblem.
                let (u2, s, v2) = compute_2x2_uptrig_svd(
                    diagonal[start].clone(),
                    off_diagonal[start].clone(),
                    diagonal[start + 1].clone(),
                    compute_u && bi_matrix.is_upper_diagonal()
                        || compute_v && !bi_matrix.is_upper_diagonal(),
                    compute_v && bi_matrix.is_upper_diagonal()
                        || compute_u && !bi_matrix.is_upper_diagonal(),
                );
                let u2 = u2.map(|u2| GivensRotation::new_unchecked(u2.c(), T::from_real(u2.s())));
                let v2 = v2.map(|v2| GivensRotation::new_unchecked(v2.c(), T::from_real(v2.s())));

                diagonal[start] = s[0].clone();
                diagonal[start + 1] = s[1].clone();
                off_diagonal[start] = T::RealField::zero();

                if let Some(ref mut u) = u {
                    let rot = if bi_matrix.is_upper_diagonal() {
                        u2.clone().unwrap()
                    } else {
                        v2.clone().unwrap()
                    };
                    rot.rotate_rows(&mut u.fixed_columns_mut::<2>(start));
                }

                if let Some(ref mut v_t) = v_t {
                    let rot = if bi_matrix.is_upper_diagonal() {
                        v2.unwrap()
                    } else {
                        u2.unwrap()
                    };
                    rot.inverse().rotate(&mut v_t.fixed_rows_mut::<2>(start));
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(
                &mut diagonal,
                &mut off_diagonal,
                &mut u,
                &mut v_t,
                bi_matrix.is_upper_diagonal(),
                end,
                eps.clone(),
            );
            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        diagonal *= m_amax;

        // Ensure all singular value are non-negative.
        for i in 0..dim {
            let sval = diagonal[i].clone();

            if sval < T::RealField::zero() {
                diagonal[i] = -sval;

                if let Some(ref mut u) = u {
                    u.column_mut(i).neg_mut();
                }
            }
        }

        Some(Self {
            u,
            v_t,
            singular_values: diagonal,
        })
    }

    /*
    fn display_bidiag(b: &Bidiagonal<T, R, C>, begin: usize, end: usize) {
        for i in begin .. end {
            for k in begin .. i {
                print!("    ");
            }
            println!("{}  {}", b.diagonal[i], b.off_diagonal[i]);
        }
        for k in begin .. end {
            print!("    ");
        }
        println!("{}", b.diagonal[end]);
    }
    */

    fn delimit_subproblem(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        end: usize,
        eps: T::RealField,
    ) -> (usize, usize) {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if off_diagonal[m].is_zero()
                || off_diagonal[m].clone().norm1()
                    <= eps.clone() * (diagonal[n].clone().norm1() + diagonal[m].clone().norm1())
            {
                off_diagonal[m] = T::RealField::zero();
            } else if diagonal[m].clone().norm1() <= eps {
                diagonal[m] = T::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                    m + 1,
                );

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(
                        diagonal,
                        off_diagonal,
                        u,
                        v_t,
                        is_upper_diagonal,
                        m - 1,
                    );
                }
            } else if diagonal[n].clone().norm1() <= eps {
                diagonal[n] = T::RealField::zero();
                Self::cancel_vertical_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                );
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

            if off_diagonal[m].clone().norm1()
                <= eps.clone() * (diagonal[new_start].clone().norm1() + diagonal[m].clone().norm1())
            {
                off_diagonal[m] = T::RealField::zero();
                break;
            }
            // TODO: write a test that enters this case.
            else if diagonal[m].clone().norm1() <= eps {
                diagonal[m] = T::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                    n,
                );

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(
                        diagonal,
                        off_diagonal,
                        u,
                        v_t,
                        is_upper_diagonal,
                        m - 1,
                    );
                }
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_horizontal_off_diagonal_elt(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
        end: usize,
    ) {
        let mut v = Vector2::new(off_diagonal[i].clone(), diagonal[i + 1].clone());
        off_diagonal[i] = T::RealField::zero();

        for k in i..end {
            match GivensRotation::cancel_x(&v) {
                Some((rot, norm)) => {
                    let rot = GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                    diagonal[k + 1] = norm;

                    if is_upper_diagonal {
                        if let Some(ref mut u) = *u {
                            rot.inverse()
                                .rotate_rows(&mut u.fixed_columns_with_step_mut::<2>(i, k - i));
                        }
                    } else if let Some(ref mut v_t) = *v_t {
                        rot.rotate(&mut v_t.fixed_rows_with_step_mut::<2>(i, k - i));
                    }

                    if k + 1 != end {
                        v.x = -rot.s().real() * off_diagonal[k + 1].clone();
                        v.y = diagonal[k + 2].clone();
                        off_diagonal[k + 1] *= rot.c();
                    }
                }
                None => {
                    break;
                }
            }
        }
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_vertical_off_diagonal_elt(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
    ) {
        let mut v = Vector2::new(diagonal[i].clone(), off_diagonal[i].clone());
        off_diagonal[i] = T::RealField::zero();

        for k in (0..i + 1).rev() {
            match GivensRotation::cancel_y(&v) {
                Some((rot, norm)) => {
                    let rot = GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                    diagonal[k] = norm;

                    if is_upper_diagonal {
                        if let Some(ref mut v_t) = *v_t {
                            rot.rotate(&mut v_t.fixed_rows_with_step_mut::<2>(k, i - k));
                        }
                    } else if let Some(ref mut u) = *u {
                        rot.inverse()
                            .rotate_rows(&mut u.fixed_columns_with_step_mut::<2>(k, i - k));
                    }

                    if k > 0 {
                        v.x = diagonal[k - 1].clone();
                        v.y = rot.s().real() * off_diagonal[k - 1].clone();
                        off_diagonal[k - 1] *= rot.c();
                    }
                }
                None => {
                    break;
                }
            }
        }
    }

    /// Computes the rank of the decomposed matrix.
    ///
    /// The rank is the number of linearly independent rows (or columns) in the matrix.
    /// In terms of SVD, it's the count of singular values that are greater than the
    /// specified tolerance `eps`.
    ///
    /// # Parameters
    ///
    /// * `eps` - Threshold value. Singular values smaller than or equal to this are
    ///   considered to be zero. Must be non-negative.
    ///
    /// # Panics
    ///
    /// Panics if `eps` is negative.
    ///
    /// # What is Matrix Rank?
    ///
    /// The rank tells you the dimension of the space spanned by the matrix's columns
    /// (or rows). For example:
    /// - A 3×3 identity matrix has rank 3 (full rank)
    /// - A matrix of all zeros has rank 0
    /// - A 3×3 matrix with one column being a linear combination of others has rank < 3
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(m, false, false);
    /// let rank = svd.rank(1e-10);
    /// println!("Matrix rank: {}", rank); // This matrix has rank 2 (not full rank)
    /// ```
    ///
    /// # Example: Checking Full Rank
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let identity = Matrix3::identity();
    /// let svd = nalgebra::SVD::new(identity, false, false);
    /// let rank = svd.rank(1e-10);
    ///
    /// assert_eq!(rank, 3); // Identity matrix has full rank
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Matrix::rank`] - Convenience method to compute rank directly from a matrix
    /// * [`SVD::singular_values`] - The singular values used for rank computation
    #[must_use]
    pub fn rank(&self, eps: T::RealField) -> usize {
        assert!(
            eps >= T::RealField::zero(),
            "SVD rank: the epsilon must be non-negative."
        );
        self.singular_values.iter().filter(|e| **e > eps).count()
    }

    /// Reconstructs the original matrix from the SVD decomposition.
    ///
    /// This method computes `A = U * Σ * V^T` to rebuild the original matrix.
    /// This is particularly useful when you've modified the singular values to create
    /// a low-rank approximation or to filter out noise.
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix)` - The reconstructed matrix
    /// * `Err` - If `U` or `V^T` were not computed during SVD construction
    ///
    /// # Example: Basic Reconstruction
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// // Compute SVD with both U and V^T
    /// let svd = nalgebra::SVD::new(m, true, true);
    ///
    /// // Reconstruct the original matrix
    /// let reconstructed = svd.recompose().unwrap();
    ///
    /// // The reconstructed matrix should be very close to the original
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Low-Rank Approximation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let mut svd = nalgebra::SVD::new(m, true, true);
    ///
    /// // Keep only the largest singular value (rank-1 approximation)
    /// for i in 1..svd.singular_values.len() {
    ///     svd.singular_values[i] = 0.0;
    /// }
    ///
    /// // Reconstruct to get a rank-1 approximation
    /// let approximation = svd.recompose().unwrap();
    /// println!("Rank-1 approximation:\n{}", approximation);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Verification**: Check that SVD was computed correctly
    /// - **Data Compression**: Zero out small singular values for compression
    /// - **Noise Reduction**: Remove singular values associated with noise
    /// - **Dimensionality Reduction**: Create low-rank approximations
    ///
    /// # See Also
    ///
    /// * [`SVD::new`] - Make sure to set `compute_u` and `compute_v` to `true`
    /// * [`SVD::pseudo_inverse`] - Related operation using modified singular values
    pub fn recompose(self) -> Result<OMatrix<T, R, C>, &'static str> {
        match (self.u, self.v_t) {
            (Some(mut u), Some(v_t)) => {
                for i in 0..self.singular_values.len() {
                    let val = self.singular_values[i].clone();
                    u.column_mut(i).scale_mut(val);
                }
                Ok(u * v_t)
            }
            (None, None) => Err("SVD recomposition: U and V^t have not been computed."),
            (None, _) => Err("SVD recomposition: U has not been computed."),
            (_, None) => Err("SVD recomposition: V^t has not been computed."),
        }
    }

    /// Computes the Moore-Penrose pseudo-inverse of the decomposed matrix.
    ///
    /// The pseudo-inverse is a generalization of the matrix inverse that exists for any matrix
    /// (even non-square or singular matrices). It's computed using the SVD as:
    /// `A^+ = V * Σ^+ * U^T`, where `Σ^+` has the reciprocals of non-zero singular values.
    ///
    /// # Parameters
    ///
    /// * `eps` - Threshold for treating singular values as zero. Singular values smaller than
    ///   this are set to zero in the inverse. Must be non-negative.
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix)` - The pseudo-inverse matrix
    /// * `Err` - If `U` or `V^T` were not computed, or if `eps` is negative
    ///
    /// # What is a Pseudo-Inverse?
    ///
    /// For a regular invertible square matrix, `A * A^(-1) = I`. The pseudo-inverse `A^+`
    /// generalizes this:
    /// - For invertible matrices: `A^+ = A^(-1)`
    /// - For rectangular matrices: Provides the "best fit" inverse
    /// - For singular matrices: Handles rank deficiency gracefully
    ///
    /// # Example: Basic Pseudo-Inverse
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(m, true, true);
    /// let pinv = svd.pseudo_inverse(1e-10).unwrap();
    ///
    /// // For an invertible matrix, pseudo-inverse equals the regular inverse
    /// let regular_inv = m.try_inverse().unwrap();
    /// assert!((pinv - regular_inv).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving Least Squares
    ///
    /// ```
    /// use nalgebra::{Matrix2x3, Vector2};
    ///
    /// // Overdetermined system: 3 equations, 2 unknowns (no exact solution)
    /// let a = Matrix2x3::new(
    ///     1.0, 2.0, 1.0,
    ///     2.0, 1.0, 3.0,
    /// ).transpose();
    ///
    /// let b = Vector2::new(1.0, 2.0);
    ///
    /// // Pseudo-inverse gives the least-squares solution
    /// let svd = nalgebra::SVD::new(a, true, true);
    /// let pinv = svd.pseudo_inverse(1e-10).unwrap();
    /// let x = pinv * b;
    ///
    /// // x minimizes ||Ax - b||
    /// println!("Least squares solution: {}", x);
    /// ```
    ///
    /// # Example: Handling Singular Matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // This matrix is singular (rank 2, not full rank)
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Regular inverse doesn't exist
    /// assert!(m.try_inverse().is_none());
    ///
    /// // But pseudo-inverse always exists!
    /// let svd = nalgebra::SVD::new(m, true, true);
    /// let pinv = svd.pseudo_inverse(1e-10).unwrap();
    /// println!("Pseudo-inverse: {}", pinv);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Least Squares Problems**: Find best-fit solutions to overdetermined systems
    /// - **Robust Solving**: Handle nearly singular matrices without numerical issues
    /// - **Data Fitting**: Regression and curve fitting applications
    /// - **Control Theory**: Computing optimal controllers
    ///
    /// # See Also
    ///
    /// * [`SVD::solve`] - Solve linear systems directly
    /// * [`Matrix::pseudo_inverse`] - Convenience method on matrices
    /// * [`SVD::recompose`] - Related matrix reconstruction
    pub fn pseudo_inverse(mut self, eps: T::RealField) -> Result<OMatrix<T, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<C, R>,
    {
        if eps < T::RealField::zero() {
            Err("SVD pseudo inverse: the epsilon must be non-negative.")
        } else {
            for i in 0..self.singular_values.len() {
                let val = self.singular_values[i].clone();

                if val > eps {
                    self.singular_values[i] = T::RealField::one() / val;
                } else {
                    self.singular_values[i] = T::RealField::zero();
                }
            }

            self.recompose().map(|m| m.adjoint())
        }
    }

    /// Solves the linear system `A * x = b` using the SVD decomposition.
    ///
    /// This method finds the solution `x` to the equation `A * x = b`, where `A` is the
    /// original decomposed matrix. It uses the pseudo-inverse approach, which works for:
    /// - Square systems (exact solution if matrix is invertible)
    /// - Overdetermined systems (least-squares solution)
    /// - Underdetermined systems (minimum-norm solution)
    /// - Singular systems (best approximate solution)
    ///
    /// # Parameters
    ///
    /// * `b` - The right-hand side vector or matrix. Must have the same number of rows as
    ///   the original matrix `A`.
    /// * `eps` - Threshold for treating singular values as zero. Must be non-negative.
    ///
    /// # Returns
    ///
    /// * `Ok(x)` - The solution vector/matrix
    /// * `Err` - If `U` or `V^T` were not computed, or if `eps` is negative
    ///
    /// # How It Works
    ///
    /// Given `A = U * Σ * V^T`, the solution is computed as:
    /// 1. Compute `U^T * b`
    /// 2. Divide by non-zero singular values
    /// 3. Multiply by `V^T^T`
    ///
    /// Singular values smaller than `eps` are treated as zero, providing numerical stability.
    ///
    /// # Example: Solving a Square System
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let svd = nalgebra::SVD::new(a, true, true);
    /// let x = svd.solve(&b, 1e-10).unwrap();
    ///
    /// // Verify: A * x ≈ b
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Least-Squares Solution
    ///
    /// ```
    /// use nalgebra::{Matrix2x3, Vector3, Vector2};
    ///
    /// // Overdetermined system: more equations than unknowns
    /// let a = Matrix2x3::new(
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    /// ).transpose();  // Now it's 3x2
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.5);
    ///
    /// let svd = nalgebra::SVD::new(a, true, true);
    /// let x = svd.solve(&b, 1e-10).unwrap();
    ///
    /// // x is the least-squares solution minimizing ||Ax - b||
    /// println!("Least-squares solution: {}", x);
    /// ```
    ///
    /// # Example: Multiple Right-Hand Sides
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix3x2};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// // Solve for two different b vectors at once
    /// let b = Matrix3x2::new(
    ///     1.0, 4.0,
    ///     2.0, 5.0,
    ///     3.0, 6.0,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(a, true, true);
    /// let x = svd.solve(&b, 1e-10).unwrap();
    ///
    /// // Each column of x solves for corresponding column of b
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Linear Systems**: Solve Ax = b for well-conditioned matrices
    /// - **Least Squares**: Fit models to data with more observations than parameters
    /// - **Regularization**: Handle ill-conditioned systems by adjusting `eps`
    /// - **Batch Solving**: Solve multiple systems with the same matrix using matrix `b`
    ///
    /// # See Also
    ///
    /// * [`SVD::pseudo_inverse`] - Compute the pseudo-inverse explicitly
    /// * [`Matrix::solve_lower_triangular`] - For triangular systems
    /// * [`LU::solve`] - Alternative for square systems
    // TODO: make this more generic wrt the storage types and the dimensions for `b`.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
        eps: T::RealField,
    ) -> Result<OMatrix<T, C, C2>, &'static str>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<C, C2> + Allocator<DimMinimum<R, C>, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        if eps < T::RealField::zero() {
            Err("SVD solve: the epsilon must be non-negative.")
        } else {
            match (&self.u, &self.v_t) {
                (Some(u), Some(v_t)) => {
                    let mut ut_b = u.ad_mul(b);

                    for j in 0..ut_b.ncols() {
                        let mut col = ut_b.column_mut(j);

                        for i in 0..self.singular_values.len() {
                            let val = self.singular_values[i].clone();
                            if val > eps {
                                col[i] = col[i].clone().unscale(val);
                            } else {
                                col[i] = T::zero();
                            }
                        }
                    }

                    Ok(v_t.ad_mul(&ut_b))
                }
                (None, None) => Err("SVD solve: U and V^t have not been computed."),
                (None, _) => Err("SVD solve: U has not been computed."),
                (_, None) => Err("SVD solve: V^t has not been computed."),
            }
        }
    }

    /// Converts the SVD into a Polar Decomposition.
    ///
    /// The polar decomposition expresses a matrix as `A = P * U`, where:
    /// - `P` is a positive semi-definite Hermitian matrix (the "stretch" component)
    /// - `U` is a unitary matrix (the "rotation" component)
    ///
    /// This is the **Left Polar Decomposition** (also called Reverse Polar Decomposition).
    /// From the SVD `A = U * Σ * V^T`, the polar decomposition is:
    /// - `P = U * Σ * U^H` (positive semi-definite)
    /// - `U_polar = U * V^T` (unitary)
    ///
    /// # Returns
    ///
    /// * `Some((P, U_polar))` - The polar decomposition matrices
    /// * `None` - If the singular vectors `U` and `V^T` were not computed
    ///
    /// # What is Polar Decomposition?
    ///
    /// Polar decomposition is analogous to writing a complex number in polar form `z = r * e^(iθ)`:
    /// - `P` (positive semi-definite) is like the magnitude `r`
    /// - `U` (unitary) is like the phase `e^(iθ)`
    ///
    /// This decomposition is unique for invertible matrices and separates scaling from rotation.
    ///
    /// # Example: Basic Polar Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(m, true, true);
    ///
    /// if let Some((p, u)) = svd.to_polar() {
    ///     // P should be positive semi-definite (all eigenvalues ≥ 0)
    ///     // U should be unitary (U * U^H = I)
    ///
    ///     // Verify: P * U ≈ original matrix
    ///     assert!((p * u - m).norm() < 1e-10);
    ///
    ///     // Verify U is unitary
    ///     let identity = Matrix2::identity();
    ///     assert!((u * u.adjoint() - identity).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// # Example: Extracting Rotation and Scale
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A matrix that combines rotation and scaling
    /// let m = Matrix2::new(
    ///     2.0, -2.0,
    ///     2.0,  2.0,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(m, true, true);
    ///
    /// if let Some((p, u)) = svd.to_polar() {
    ///     println!("Stretch component P:\n{}", p);
    ///     println!("Rotation component U:\n{}", u);
    ///
    ///     // P contains the scaling information
    ///     // U contains the rotation information
    /// }
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Computer Graphics**: Decompose transformations into rotation and scaling
    /// - **Mechanics**: Separate deformation into pure strain and rigid rotation
    /// - **Continuum Mechanics**: Analyze material deformation
    /// - **Numerical Analysis**: Condition number estimation
    ///
    /// # See Also
    ///
    /// * [`SVD::new`] - Make sure to set `compute_u` and `compute_v` to `true`
    /// * [`SVD::recompose`] - Reconstruct the original matrix from SVD
    pub fn to_polar(&self) -> Option<(OMatrix<T, R, R>, OMatrix<T, R, C>)>
    where
        DefaultAllocator: Allocator<R, C> //result
            + Allocator<DimMinimum<R, C>, R> // adjoint
            + Allocator<DimMinimum<R, C>> // mapped vals
            + Allocator<R, R> // result
            + Allocator<DimMinimum<R, C>, DimMinimum<R, C>>, // square matrix
    {
        match (&self.u, &self.v_t) {
            (Some(u), Some(v_t)) => Some((
                u * OMatrix::from_diagonal(&self.singular_values.map(|e| T::from_real(e)))
                    * u.adjoint(),
                u * v_t,
            )),
            _ => None,
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> SVD<T, R, C>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>, // for sorted singular values
{
    /// Computes the Singular Value Decomposition of `matrix` with sorted singular values.
    ///
    /// This is the standard SVD computation where `A = U * Σ * V^T`, with singular values
    /// in `Σ` guaranteed to be sorted in **descending order** (largest to smallest).
    ///
    /// This is the most commonly used SVD function as many applications rely on sorted
    /// singular values (e.g., dimensionality reduction, low-rank approximation).
    ///
    /// # Parameters
    ///
    /// * `matrix` - The matrix to decompose
    /// * `compute_u` - Set to `true` to compute the left-singular vectors matrix `U`
    /// * `compute_v` - Set to `true` to compute the right-singular vectors matrix `V^T`
    ///
    /// # Performance
    ///
    /// If you don't need sorted singular values, use [`SVD::new_unordered`] for slightly
    /// better performance. If you don't need the singular vectors, set both flags to `false`.
    ///
    /// # Example: Computing SVD
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     4.0, 1.0,
    ///     2.0, 3.0,
    /// );
    ///
    /// // Compute full SVD with sorted singular values
    /// let svd = nalgebra::SVD::new(m, true, true);
    ///
    /// // Singular values are in descending order
    /// println!("Singular values: {}", svd.singular_values);
    /// assert!(svd.singular_values[0] >= svd.singular_values[1]);
    /// ```
    ///
    /// # Example: Dimensionality Reduction
    ///
    /// ```
    /// use nalgebra::DMatrix;
    ///
    /// // Create a 100x50 data matrix (e.g., 100 samples, 50 features)
    /// let data = DMatrix::<f64>::new_random(100, 50);
    ///
    /// let svd = nalgebra::SVD::new(data.clone(), true, true);
    ///
    /// // Keep only top 10 components (90% dimensionality reduction)
    /// let mut reduced_svd = svd.clone();
    /// for i in 10..reduced_svd.singular_values.len() {
    ///     reduced_svd.singular_values[i] = 0.0;
    /// }
    ///
    /// // Reconstruct with reduced dimensions
    /// let approximation = reduced_svd.recompose().unwrap();
    ///
    /// println!("Original shape: {}x{}", data.nrows(), data.ncols());
    /// println!("Effective rank: 10");
    /// ```
    ///
    /// # Example: Image Compression
    ///
    /// ```
    /// use nalgebra::DMatrix;
    ///
    /// // Simulate a grayscale image as a matrix (e.g., 256x256)
    /// let image = DMatrix::<f64>::from_fn(256, 256, |i, j| {
    ///     ((i as f64 / 255.0) * (j as f64 / 255.0)).sin()
    /// });
    ///
    /// let svd = nalgebra::SVD::new(image.clone(), true, true);
    ///
    /// // Compress by keeping only 50 largest singular values
    /// let mut compressed = svd.clone();
    /// for i in 50..compressed.singular_values.len() {
    ///     compressed.singular_values[i] = 0.0;
    /// }
    ///
    /// let compressed_image = compressed.recompose().unwrap();
    ///
    /// // Calculate compression ratio and error
    /// let compression_ratio = 50.0 / 256.0;
    /// let error = (image - compressed_image).norm() / image.norm();
    ///
    /// println!("Compression: {:.1}% of original", compression_ratio * 100.0);
    /// println!("Relative error: {:.2}%", error * 100.0);
    /// ```
    ///
    /// # Example: Noise Filtering
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let noisy_data = Matrix3::new(
    ///     5.1, 2.9, 1.1,
    ///     3.0, 4.8, 2.2,
    ///     1.1, 2.0, 2.9,
    /// );
    ///
    /// let svd = nalgebra::SVD::new(noisy_data, true, true);
    ///
    /// // Filter out small singular values (likely noise)
    /// let mut filtered = svd.clone();
    /// let threshold = 0.5; // Singular values below this are considered noise
    /// for i in 0..filtered.singular_values.len() {
    ///     if filtered.singular_values[i] < threshold {
    ///         filtered.singular_values[i] = 0.0;
    ///     }
    /// }
    ///
    /// let denoised = filtered.recompose().unwrap();
    /// println!("Denoised data:\n{}", denoised);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`SVD::new_unordered`] - Faster version without sorting
    /// * [`SVD::try_new`] - Fallible version with custom parameters
    /// * [`Matrix::singular_values`] - Convenience method for computing only singular values
    pub fn new(matrix: OMatrix<T, R, C>, compute_u: bool, compute_v: bool) -> Self {
        let mut svd = Self::new_unordered(matrix, compute_u, compute_v);

        if !Self::use_special_always_ordered_svd3() && !Self::use_special_always_ordered_svd2() {
            svd.sort_by_singular_values();
        }

        svd
    }

    /// Attempts to compute the SVD with sorted singular values and custom parameters.
    ///
    /// This is the fallible version of [`SVD::new`] that allows fine control over convergence
    /// criteria and iteration limits. The singular values are guaranteed to be sorted in
    /// **descending order**.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The matrix to decompose
    /// * `compute_u` - Set to `true` to compute the left-singular vectors matrix `U`
    /// * `compute_v` - Set to `true` to compute the right-singular vectors matrix `V^T`
    /// * `eps` - Convergence tolerance. Values smaller than this are considered zero.
    ///   Typical value: `T::default_epsilon() * 5.0`
    /// * `max_niter` - Maximum iterations allowed. Returns `None` if exceeded.
    ///   If `max_niter == 0`, runs until convergence without limit.
    ///
    /// # Returns
    ///
    /// * `Some(SVD)` - Successfully computed SVD with sorted singular values
    /// * `None` - Failed to converge within `max_niter` iterations
    ///
    /// # Example: Custom Tolerance
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Use stricter tolerance for higher precision
    /// let eps = 1e-15;
    /// let max_iter = 200;
    ///
    /// match nalgebra::SVD::try_new(m, true, true, eps, max_iter) {
    ///     Some(svd) => {
    ///         println!("SVD computed successfully!");
    ///         // Singular values are sorted in descending order
    ///         for i in 1..svd.singular_values.len() {
    ///             assert!(svd.singular_values[i-1] >= svd.singular_values[i]);
    ///         }
    ///     }
    ///     None => {
    ///         println!("Failed to converge!");
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Limited Iterations for Real-Time
    ///
    /// ```
    /// use nalgebra::DMatrix;
    ///
    /// let large_matrix = DMatrix::<f64>::new_random(500, 500);
    ///
    /// // Limit iterations for real-time applications
    /// let max_iter = 50; // Stop early if needed
    /// let eps = 1e-6;    // Relaxed tolerance
    ///
    /// match nalgebra::SVD::try_new(large_matrix, false, false, eps, max_iter) {
    ///     Some(svd) => {
    ///         println!("Converged within {} iterations", max_iter);
    ///         println!("Largest singular value: {}", svd.singular_values[0]);
    ///     }
    ///     None => {
    ///         println!("Did not converge, consider increasing max_iter");
    ///     }
    /// }
    /// ```
    ///
    /// # When to Use
    ///
    /// Use this function when you need:
    /// - **Sorted singular values** (most common case)
    /// - **Custom convergence criteria** for specific numerical requirements
    /// - **Iteration limits** for real-time or bounded-time applications
    /// - **Graceful handling** of convergence failures
    ///
    /// # See Also
    ///
    /// * [`SVD::new`] - Infallible version with default parameters
    /// * [`SVD::try_new_unordered`] - Unsorted version with custom parameters
    /// * [`SVD::new_unordered`] - Faster unsorted version
    pub fn try_new(
        matrix: OMatrix<T, R, C>,
        compute_u: bool,
        compute_v: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<Self> {
        Self::try_new_unordered(matrix, compute_u, compute_v, eps, max_niter).map(|mut svd| {
            if !Self::use_special_always_ordered_svd3() && !Self::use_special_always_ordered_svd2()
            {
                svd.sort_by_singular_values();
            }

            svd
        })
    }

    /// Sorts the SVD components by singular values in descending order.
    ///
    /// This method reorders the singular values from largest to smallest and permutes the
    /// corresponding columns of `U` and rows of `V^T` to maintain the decomposition
    /// `A = U * Σ * V^T`.
    ///
    /// # When to Use
    ///
    /// This function is only needed if you used [`SVD::new_unordered`] or
    /// [`SVD::try_new_unordered`] and later decided you need sorted singular values.
    /// If you use [`SVD::new`] or [`SVD::try_new`], the values are already sorted.
    ///
    /// # Why Sort?
    ///
    /// Many applications rely on sorted singular values:
    /// - **Dimensionality Reduction**: Keep the top-k largest values
    /// - **Principal Component Analysis**: Components ordered by importance
    /// - **Low-Rank Approximation**: Truncate at a specific index
    /// - **Condition Number**: Ratio of largest to smallest value
    /// - **Spectral Analysis**: Identify dominant modes
    ///
    /// # Example: Sorting After Unordered Computation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Compute SVD without sorting (faster)
    /// let mut svd = nalgebra::SVD::new_unordered(m, true, true);
    ///
    /// // Singular values might not be sorted
    /// println!("Before sorting: {}", svd.singular_values);
    ///
    /// // Sort when needed
    /// svd.sort_by_singular_values();
    ///
    /// // Now guaranteed to be in descending order
    /// println!("After sorting: {}", svd.singular_values);
    /// for i in 1..svd.singular_values.len() {
    ///     assert!(svd.singular_values[i-1] >= svd.singular_values[i]);
    /// }
    /// ```
    ///
    /// # Example: Conditional Sorting
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let mut svd = nalgebra::SVD::new_unordered(m, true, true);
    ///
    /// // Only sort if we need to use the top singular values
    /// let need_sorted = true;
    /// if need_sorted {
    ///     svd.sort_by_singular_values();
    ///
    ///     // Now safe to use for dimensionality reduction
    ///     println!("Top singular value: {}", svd.singular_values[0]);
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// Sorting has O(n log n) complexity where n is the number of singular values.
    /// If you know you'll need sorted values, it's more efficient to use [`SVD::new`]
    /// directly rather than computing unordered and then sorting.
    ///
    /// # See Also
    ///
    /// * [`SVD::new`] - Computes SVD with automatic sorting
    /// * [`SVD::new_unordered`] - Computes SVD without sorting
    pub fn sort_by_singular_values(&mut self) {
        const VALUE_PROCESSED: usize = usize::MAX;

        // Collect the singular values with their original index, ...
        let mut singular_values = self.singular_values.map_with_location(|r, _, e| (e, r));
        assert_ne!(
            singular_values.data.shape().0.value(),
            VALUE_PROCESSED,
            "Too many singular values"
        );

        // ... sort the singular values, ...
        singular_values
            .as_mut_slice()
            .sort_unstable_by(|(a, _), (b, _)| b.partial_cmp(a).expect("Singular value was NaN"));

        // ... and store them.
        self.singular_values
            .zip_apply(&singular_values, |value, (new_value, _)| {
                value.clone_from(&new_value)
            });

        // Calculate required permutations given the sorted indices.
        // We need to identify all circles to calculate the required swaps.
        let mut permutations =
            crate::PermutationSequence::identity_generic(singular_values.data.shape().0);

        for i in 0..singular_values.len() {
            let mut index_1 = i;
            let mut index_2 = singular_values[i].1;

            // Check whether the value was already visited ...
            while index_2 != VALUE_PROCESSED // ... or a "double swap" must be avoided.
                && singular_values[index_2].1 != VALUE_PROCESSED
            {
                // Add the permutation ...
                permutations.append_permutation(index_1, index_2);
                // ... and mark the value as visited.
                singular_values[index_1].1 = VALUE_PROCESSED;

                index_1 = index_2;
                index_2 = singular_values[index_1].1;
            }
        }

        // Permute the optional components
        if let Some(u) = self.u.as_mut() {
            permutations.permute_columns(u);
        }

        if let Some(v_t) = self.v_t.as_mut() {
            permutations.permute_rows(v_t);
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the singular values of this matrix (unordered).
    ///
    /// This method extracts only the singular values from the SVD decomposition, without
    /// computing the singular vectors `U` and `V^T`. The values are **not guaranteed to be
    /// sorted** in any particular order.
    ///
    /// # Performance
    ///
    /// This is faster than computing the full SVD since it skips the singular vectors.
    /// If you need sorted values, use [`Matrix::singular_values`] instead.
    ///
    /// # What Are Singular Values?
    ///
    /// Singular values are the square roots of the eigenvalues of `A^T * A` (or `A * A^T`).
    /// They represent the "stretching factors" in the different directions when the matrix
    /// is viewed as a linear transformation. Geometrically, they're the lengths of the
    /// semi-axes of the hyperellipsoid that is the image of the unit sphere.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let sv = m.singular_values_unordered();
    /// println!("Singular values: {}", sv);
    ///
    /// // All singular values are non-negative
    /// for i in 0..sv.len() {
    ///     assert!(sv[i] >= 0.0);
    /// }
    /// ```
    ///
    /// # Example: Condition Number
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     10.0, 1.0, 0.0,
    ///     1.0, 5.0, 1.0,
    ///     0.0, 1.0, 0.1,
    /// );
    ///
    /// let sv = m.singular_values_unordered();
    ///
    /// // Find max and min singular values (need to search since unordered)
    /// let max_sv = sv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    /// let min_sv = sv.iter().cloned().fold(f64::INFINITY, f64::min);
    ///
    /// let condition_number = max_sv / min_sv;
    /// println!("Condition number: {:.2}", condition_number);
    ///
    /// // High condition number indicates ill-conditioned matrix
    /// if condition_number > 1000.0 {
    ///     println!("Warning: Matrix is ill-conditioned!");
    /// }
    /// ```
    ///
    /// # Example: Frobenius Norm
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 4.0,
    ///     0.0, 0.0,
    /// );
    ///
    /// let sv = m.singular_values_unordered();
    ///
    /// // Frobenius norm equals the square root of sum of squared singular values
    /// let frobenius_from_sv = sv.iter().map(|s| s * s).sum::<f64>().sqrt();
    /// let frobenius_direct = m.norm();
    ///
    /// assert!((frobenius_from_sv - frobenius_direct).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Matrix::singular_values`] - Sorted version
    /// * [`SVD::new_unordered`] - Full SVD decomposition
    /// * [`Matrix::rank`] - Compute matrix rank using singular values
    #[must_use]
    pub fn singular_values_unordered(&self) -> OVector<T::RealField, DimMinimum<R, C>> {
        SVD::new_unordered(self.clone_owned(), false, false).singular_values
    }

    /// Computes the rank of this matrix.
    ///
    /// The rank is the number of linearly independent rows (or columns) in the matrix.
    /// This is computed by counting how many singular values are greater than the
    /// specified tolerance `eps`.
    ///
    /// # Parameters
    ///
    /// * `eps` - Threshold for treating singular values as zero. Singular values less
    ///   than or equal to this value are considered zero and don't contribute to the rank.
    ///
    /// # What is Matrix Rank?
    ///
    /// The rank tells you the dimension of the space spanned by the matrix's columns
    /// (column rank) or rows (row rank). These are always equal. For example:
    /// - A 3×3 identity matrix has rank 3 (full rank)
    /// - A matrix where one row is a multiple of another has reduced rank
    /// - A zero matrix has rank 0
    ///
    /// # Example: Full Rank Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let identity = Matrix3::identity();
    /// let rank = identity.rank(1e-10);
    ///
    /// assert_eq!(rank, 3); // Full rank
    /// ```
    ///
    /// # Example: Rank-Deficient Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // This matrix has rank 2 (third row = first row + second row)
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0,
    /// );
    ///
    /// let rank = m.rank(1e-10);
    /// assert_eq!(rank, 2); // Rank-deficient
    /// ```
    ///
    /// # Example: Determining Invertibility
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     4.0, 2.0,  // Second row is 2x the first row
    /// );
    ///
    /// let rank = m.rank(1e-10);
    ///
    /// if rank < m.nrows() {
    ///     println!("Matrix is singular (not invertible)");
    ///     assert!(m.try_inverse().is_none());
    /// }
    /// ```
    ///
    /// # Example: Checking Linear Independence
    ///
    /// ```
    /// use nalgebra::Matrix3x4;
    ///
    /// // 3 vectors as columns, in 4D space
    /// let vectors = Matrix3x4::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    ///     0.0, 0.0, 0.0,
    /// ).transpose();
    ///
    /// let rank = vectors.rank(1e-10);
    ///
    /// // The 3 column vectors are linearly independent
    /// assert_eq!(rank, 3);
    /// ```
    ///
    /// # Choosing `eps`
    ///
    /// The choice of `eps` depends on your numerical precision requirements:
    /// - Too small: Numerical errors might be counted as non-zero singular values
    /// - Too large: Actually significant values might be discarded
    /// - Typical choice: `1e-10` for f64, or `matrix.norm() * f64::EPSILON * 10.0`
    ///
    /// # See Also
    ///
    /// * [`SVD::rank`] - Rank from an existing SVD decomposition
    /// * [`Matrix::singular_values`] - View the singular values directly
    /// * [`Matrix::determinant`] - Zero determinant indicates rank < n
    #[must_use]
    pub fn rank(&self, eps: T::RealField) -> usize {
        let svd = SVD::new_unordered(self.clone_owned(), false, false);
        svd.rank(eps)
    }

    /// Computes the Moore-Penrose pseudo-inverse of this matrix.
    ///
    /// The pseudo-inverse is a generalization of the matrix inverse that works for any matrix,
    /// including non-square and singular matrices. For an invertible square matrix, the
    /// pseudo-inverse equals the regular inverse.
    ///
    /// # Parameters
    ///
    /// * `eps` - Threshold for treating singular values as zero. Singular values smaller than
    ///   this are treated as zero when computing the inverse. Must be non-negative.
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix)` - The pseudo-inverse
    /// * `Err` - If `eps` is negative
    ///
    /// # What is a Pseudo-Inverse?
    ///
    /// For a matrix `A`, the pseudo-inverse `A^+` satisfies:
    /// - `A * A^+ * A = A` (the pseudo-inverse "inverts" as much as possible)
    /// - `A^+ * A * A^+ = A^+`
    /// - `(A * A^+)^T = A * A^+`
    /// - `(A^+ * A)^T = A^+ * A`
    ///
    /// # Example: Invertible Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// let pinv = m.pseudo_inverse(1e-10).unwrap();
    /// let inv = m.try_inverse().unwrap();
    ///
    /// // For invertible matrices, pseudo-inverse equals regular inverse
    /// assert!((pinv - inv).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Singular Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // This matrix is singular (rank 2)
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Regular inverse doesn't exist
    /// assert!(m.try_inverse().is_none());
    ///
    /// // But pseudo-inverse always exists!
    /// let pinv = m.pseudo_inverse(1e-10).unwrap();
    ///
    /// // Check the property: A * A^+ * A = A
    /// let reconstructed = &m * &pinv * &m;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving Least Squares
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Vector3};
    ///
    /// // Overdetermined system: 3 equations, 2 unknowns
    /// // x + y = 1
    /// // 2x + y = 2
    /// // x + 2y = 3
    /// let a = Matrix3x2::new(
    ///     1.0, 1.0,
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Least-squares solution: x = A^+ * b
    /// let pinv = a.pseudo_inverse(1e-10).unwrap();
    /// let x = pinv * b;
    ///
    /// println!("Least-squares solution: {}", x);
    /// // This minimizes ||Ax - b||^2
    /// ```
    ///
    /// # Example: Rectangular Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    /// );
    ///
    /// let pinv = m.pseudo_inverse(1e-10).unwrap();
    ///
    /// // For rectangular matrices, A^+ is (A^T * A)^-1 * A^T (right pseudo-inverse)
    /// // or A^T * (A * A^T)^-1 (left pseudo-inverse)
    /// println!("Pseudo-inverse of {}x{} matrix: {}",
    ///          m.nrows(), m.ncols(), pinv);
    /// assert_eq!(pinv.nrows(), 3);
    /// assert_eq!(pinv.ncols(), 2);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Least Squares**: Solve overdetermined systems (more equations than unknowns)
    /// - **Minimum Norm Solutions**: Solve underdetermined systems with smallest norm
    /// - **Robust Inversion**: Handle nearly singular matrices gracefully
    /// - **Data Fitting**: Linear regression and curve fitting
    /// - **Signal Processing**: Deconvolution and filtering
    ///
    /// # Performance Note
    ///
    /// This method computes the full SVD, which can be expensive for large matrices.
    /// If you need to solve multiple systems with the same matrix, compute the SVD once
    /// and reuse it with [`SVD::solve`].
    ///
    /// # See Also
    ///
    /// * [`SVD::pseudo_inverse`] - Compute from an existing SVD
    /// * [`Matrix::try_inverse`] - Regular matrix inverse (square matrices only)
    /// * [`SVD::solve`] - Solve linear systems directly
    pub fn pseudo_inverse(self, eps: T::RealField) -> Result<OMatrix<T, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<C, R>,
    {
        SVD::new_unordered(self.clone_owned(), true, true).pseudo_inverse(eps)
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>,
{
    /// Computes the singular values of this matrix (sorted in descending order).
    ///
    /// This method extracts only the singular values from the SVD decomposition, without
    /// computing the singular vectors `U` and `V^T`. The values are guaranteed to be sorted
    /// in **descending order** (largest to smallest).
    ///
    /// # What Are Singular Values?
    ///
    /// Singular values are non-negative real numbers that represent the "stretching factors"
    /// of a matrix when viewed as a linear transformation. They are the square roots of the
    /// eigenvalues of `A^T * A` (or equivalently, `A * A^T`).
    ///
    /// Geometrically, if you imagine a matrix transforming a unit sphere, the singular values
    /// are the lengths of the semi-axes of the resulting hyperellipsoid.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let sv = m.singular_values();
    /// println!("Singular values: {}", sv);
    ///
    /// // Values are sorted in descending order
    /// for i in 1..sv.len() {
    ///     assert!(sv[i-1] >= sv[i]);
    /// }
    /// ```
    ///
    /// # Example: Spectral Norm (Largest Singular Value)
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let sv = m.singular_values();
    ///
    /// // The spectral norm (2-norm) is the largest singular value
    /// let spectral_norm = sv[0];
    /// println!("Spectral norm: {}", spectral_norm);
    ///
    /// // This is also available as:
    /// assert!((spectral_norm - m.norm()).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Condition Number
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     100.0, 1.0, 0.0,
    ///     1.0, 10.0, 1.0,
    ///     0.0, 1.0, 0.01,
    /// );
    ///
    /// let sv = m.singular_values();
    ///
    /// // Condition number is the ratio of largest to smallest singular value
    /// let condition_number = sv[0] / sv[sv.len() - 1];
    /// println!("Condition number: {:.2e}", condition_number);
    ///
    /// // High condition number means the matrix is ill-conditioned
    /// // (sensitive to numerical errors)
    /// if condition_number > 1000.0 {
    ///     println!("Warning: Matrix is ill-conditioned!");
    /// }
    /// ```
    ///
    /// # Example: Effective Rank
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     2.0, 4.0, 6.0, 8.0,
    ///     3.0, 6.0, 9.0, 12.0,
    ///     0.0, 0.0, 0.0, 1e-10,
    /// );
    ///
    /// let sv = m.singular_values();
    ///
    /// // Count "significant" singular values
    /// let threshold = 1e-8;
    /// let effective_rank = sv.iter().filter(|&&s| s > threshold).count();
    ///
    /// println!("Effective rank: {}", effective_rank);
    /// ```
    ///
    /// # Example: Energy/Variance Analysis
    ///
    /// ```
    /// use nalgebra::DMatrix;
    ///
    /// // Create a data matrix (e.g., 100 samples, 20 features)
    /// let data = DMatrix::<f64>::new_random(100, 20);
    ///
    /// let sv = data.singular_values();
    ///
    /// // Total "energy" is sum of squared singular values
    /// let total_energy: f64 = sv.iter().map(|s| s * s).sum();
    ///
    /// // Find how many components needed for 90% of energy
    /// let mut cumulative = 0.0;
    /// let mut components_90 = 0;
    ///
    /// for (i, &s) in sv.iter().enumerate() {
    ///     cumulative += s * s;
    ///     if cumulative / total_energy >= 0.90 {
    ///         components_90 = i + 1;
    ///         break;
    ///     }
    /// }
    ///
    /// println!("Need {} components for 90% energy", components_90);
    /// println!("Dimensionality reduction: {}%",
    ///          (1.0 - components_90 as f64 / sv.len() as f64) * 100.0);
    /// ```
    ///
    /// # Performance
    ///
    /// This method is faster than computing the full SVD since it doesn't compute the
    /// singular vectors. However, it's slightly slower than [`Matrix::singular_values_unordered`]
    /// due to the sorting step.
    ///
    /// # See Also
    ///
    /// * [`Matrix::singular_values_unordered`] - Faster unsorted version
    /// * [`SVD::new`] - Full SVD decomposition with sorted values
    /// * [`Matrix::norm`] - Matrix 2-norm (largest singular value)
    /// * [`Matrix::rank`] - Number of significant singular values
    #[must_use]
    pub fn singular_values(&self) -> OVector<T::RealField, DimMinimum<R, C>> {
        SVD::new(self.clone_owned(), false, false).singular_values
    }
}

// Explicit formulae inspired from the paper "Computing the Singular Values of 2-by-2 Complex
// Matrices", Sanzheng Qiao and Xiaohong Wang.
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
fn compute_2x2_uptrig_svd<T: RealField>(
    m11: T,
    m12: T,
    m22: T,
    compute_u: bool,
    compute_v: bool,
) -> (
    Option<GivensRotation<T>>,
    Vector2<T>,
    Option<GivensRotation<T>>,
) {
    let two: T::RealField = crate::convert(2.0f64);
    let half: T::RealField = crate::convert(0.5f64);

    let denom = (m11.clone() + m22.clone()).hypot(m12.clone())
        + (m11.clone() - m22.clone()).hypot(m12.clone());

    // NOTE: v1 is the singular value that is the closest to m22.
    // This prevents cancellation issues when constructing the vector `csv` below. If we chose
    // otherwise, we would have v1 ~= m11 when m12 is small. This would cause catastrophic
    // cancellation on `v1 * v1 - m11 * m11` below.
    let mut v1 = m11.clone() * m22.clone() * two / denom.clone();
    let mut v2 = half * denom;

    let mut u = None;
    let mut v_t = None;

    if compute_u || compute_v {
        let (csv, sgn_v) = GivensRotation::new(
            m11.clone() * m12.clone(),
            v1.clone() * v1.clone() - m11.clone() * m11.clone(),
        );
        v1 *= sgn_v.clone();
        v2 *= sgn_v;

        if compute_v {
            v_t = Some(csv.clone());
        }

        let cu = (m11.scale(csv.c()) + m12 * csv.s()) / v1.clone();
        let su = (m22 * csv.s()) / v1.clone();
        let (csu, sgn_u) = GivensRotation::new(cu, su);
        v1 *= sgn_u.clone();
        v2 *= sgn_u;

        if compute_u {
            u = Some(csu);
        }
    }

    (u, Vector2::new(v1, v2), v_t)
}
