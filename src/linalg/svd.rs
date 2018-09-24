#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

use num_complex::Complex;
use std::ops::MulAssign;

use alga::general::Real;
use allocator::Allocator;
use base::{DefaultAllocator, Matrix, Matrix2x3, MatrixMN, Vector2, VectorN};
use constraint::{SameNumberOfRows, ShapeConstraint};
use dimension::{Dim, DimDiff, DimMin, DimMinimum, DimSub, U1, U2};
use storage::Storage;

use geometry::UnitComplex;
use linalg::givens;
use linalg::symmetric_eigen;
use linalg::Bidiagonal;

/// Singular Value Decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            serialize = "DefaultAllocator: Allocator<N, R, C>                +
                           Allocator<N, DimMinimum<R, C>>    +
                           Allocator<N, DimMinimum<R, C>, C> +
                           Allocator<N, R, DimMinimum<R, C>>,
         MatrixMN<N, R, DimMinimum<R, C>>: Serialize,
         MatrixMN<N, DimMinimum<R, C>, C>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize"
        )
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            deserialize = "DefaultAllocator: Allocator<N, R, C>                +
                           Allocator<N, DimMinimum<R, C>>    +
                           Allocator<N, DimMinimum<R, C>, C> +
                           Allocator<N, R, DimMinimum<R, C>>,
         MatrixMN<N, R, DimMinimum<R, C>>: Deserialize<'de>,
         MatrixMN<N, DimMinimum<R, C>, C>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>"
        )
    )
)]
#[derive(Clone, Debug)]
pub struct SVD<N: Real, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>,
{
    /// The left-singular vectors `U` of this SVD.
    pub u: Option<MatrixMN<N, R, DimMinimum<R, C>>>,
    /// The right-singular vectors `V^t` of this SVD.
    pub v_t: Option<MatrixMN<N, DimMinimum<R, C>, C>>,
    /// The singular values of this SVD.
    pub singular_values: VectorN<N, DimMinimum<R, C>>,
}

impl<N: Real, R: DimMin<C>, C: Dim> Copy for SVD<N, R, C>
where
    DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>,
    MatrixMN<N, R, DimMinimum<R, C>>: Copy,
    MatrixMN<N, DimMinimum<R, C>, C>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
{
}

impl<N: Real, R: DimMin<C>, C: Dim> SVD<N, R, C>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>,
{
    /// Computes the Singular Value Decomposition of `matrix` using implicit shift.
    pub fn new(matrix: MatrixMN<N, R, C>, compute_u: bool, compute_v: bool) -> Self {
        Self::try_new(matrix, compute_u, compute_v, N::default_epsilon(), 0).unwrap()
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// # Arguments
    ///
    /// * `compute_u` − set this to `true` to enable the computation of left-singular vectors.
    /// * `compute_v` − set this to `true` to enable the computation of left-singular vectors.
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(
        mut matrix: MatrixMN<N, R, C>,
        compute_u: bool,
        compute_v: bool,
        eps: N,
        max_niter: usize,
    ) -> Option<Self> {
        assert!(
            matrix.len() != 0,
            "Cannot compute the SVD of an empty matrix."
        );
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let dim = min_nrows_ncols.value();

        let m_amax = matrix.amax();

        if !m_amax.is_zero() {
            matrix /= m_amax;
        }

        let mut b = Bidiagonal::new(matrix);
        let mut u = if compute_u { Some(b.u()) } else { None };
        let mut v_t = if compute_v { Some(b.v_t()) } else { None };

        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&mut b, &mut u, &mut v_t, dim - 1, eps);

        while end != start {
            let subdim = end - start + 1;

            // Solve the subproblem.
            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut vec;
                {
                    let dm = b.diagonal[m];
                    let dn = b.diagonal[n];
                    let fm = b.off_diagonal[m];

                    let tmm = dm * dm + b.off_diagonal[m - 1] * b.off_diagonal[m - 1];
                    let tmn = dm * fm;
                    let tnn = dn * dn + fm * fm;

                    let shift = symmetric_eigen::wilkinson_shift(tmm, tnn, tmn);

                    vec = Vector2::new(
                        b.diagonal[start] * b.diagonal[start] - shift,
                        b.diagonal[start] * b.off_diagonal[start],
                    );
                }

                for k in start..n {
                    let m12 = if k == n - 1 {
                        N::zero()
                    } else {
                        b.off_diagonal[k + 1]
                    };

                    let mut subm = Matrix2x3::new(
                        b.diagonal[k],
                        b.off_diagonal[k],
                        N::zero(),
                        N::zero(),
                        b.diagonal[k + 1],
                        m12,
                    );

                    if let Some((rot1, norm1)) = givens::cancel_y(&vec) {
                        rot1.conjugate()
                            .rotate_rows(&mut subm.fixed_columns_mut::<U2>(0));

                        if k > start {
                            // This is not the first iteration.
                            b.off_diagonal[k - 1] = norm1;
                        }

                        let v = Vector2::new(subm[(0, 0)], subm[(1, 0)]);
                        // FIXME: does the case `v.y == 0` ever happen?
                        let (rot2, norm2) =
                            givens::cancel_y(&v).unwrap_or((UnitComplex::identity(), subm[(0, 0)]));
                        rot2.rotate(&mut subm.fixed_columns_mut::<U2>(1));
                        subm[(0, 0)] = norm2;

                        if let Some(ref mut v_t) = v_t {
                            if b.is_upper_diagonal() {
                                rot1.rotate(&mut v_t.fixed_rows_mut::<U2>(k));
                            } else {
                                rot2.rotate(&mut v_t.fixed_rows_mut::<U2>(k));
                            }
                        }

                        if let Some(ref mut u) = u {
                            if b.is_upper_diagonal() {
                                rot2.inverse()
                                    .rotate_rows(&mut u.fixed_columns_mut::<U2>(k));
                            } else {
                                rot1.inverse()
                                    .rotate_rows(&mut u.fixed_columns_mut::<U2>(k));
                            }
                        }

                        b.diagonal[k + 0] = subm[(0, 0)];
                        b.diagonal[k + 1] = subm[(1, 1)];
                        b.off_diagonal[k + 0] = subm[(0, 1)];

                        if k != n - 1 {
                            b.off_diagonal[k + 1] = subm[(1, 2)];
                        }

                        vec.x = subm[(0, 1)];
                        vec.y = subm[(0, 2)];
                    } else {
                        break;
                    }
                }
            } else if subdim == 2 {
                // Solve the remaining 2x2 subproblem.
                let (u2, s, v2) = Self::compute_2x2_uptrig_svd(
                    b.diagonal[start],
                    b.off_diagonal[start],
                    b.diagonal[start + 1],
                    compute_u && b.is_upper_diagonal() || compute_v && !b.is_upper_diagonal(),
                    compute_v && b.is_upper_diagonal() || compute_u && !b.is_upper_diagonal(),
                );

                b.diagonal[start + 0] = s[0];
                b.diagonal[start + 1] = s[1];
                b.off_diagonal[start] = N::zero();

                if let Some(ref mut u) = u {
                    let rot = if b.is_upper_diagonal() {
                        u2.unwrap()
                    } else {
                        v2.unwrap()
                    };
                    rot.rotate_rows(&mut u.fixed_columns_mut::<U2>(start));
                }

                if let Some(ref mut v_t) = v_t {
                    let rot = if b.is_upper_diagonal() {
                        v2.unwrap()
                    } else {
                        u2.unwrap()
                    };
                    rot.inverse().rotate(&mut v_t.fixed_rows_mut::<U2>(start));
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(&mut b, &mut u, &mut v_t, end, eps);
            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        b.diagonal *= m_amax;

        // Ensure all singular value are non-negative.
        for i in 0..dim {
            let sval = b.diagonal[i];
            if sval < N::zero() {
                b.diagonal[i] = -sval;

                if let Some(ref mut u) = u {
                    u.column_mut(i).neg_mut();
                }
            }
        }

        Some(SVD {
            u: u,
            v_t: v_t,
            singular_values: b.diagonal,
        })
    }

    // Explicit formulaes inspired from the paper "Computing the Singular Values of 2-by-2 Complex
    // Matrices", Sanzheng Qiao and Xiaohong Wang.
    // http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
    fn compute_2x2_uptrig_svd(
        m11: N,
        m12: N,
        m22: N,
        compute_u: bool,
        compute_v: bool,
    ) -> (Option<UnitComplex<N>>, Vector2<N>, Option<UnitComplex<N>>) {
        let two: N = ::convert(2.0f64);
        let half: N = ::convert(0.5f64);

        let denom = (m11 + m22).hypot(m12) + (m11 - m22).hypot(m12);

        // NOTE: v1 is the singular value that is the closest to m22.
        // This prevents cancellation issues when constructing the vector `csv` below. If we chose
        // otherwise, we would have v1 ~= m11 when m12 is small. This would cause catastrophic
        // cancellation on `v1 * v1 - m11 * m11` below.
        let v1 = two * m11 * m22 / denom;
        let v2 = half * denom;

        let mut u = None;
        let mut v_t = None;

        if compute_u || compute_v {
            let csv = Vector2::new(m11 * m12, v1 * v1 - m11 * m11).normalize();

            if compute_v {
                v_t = Some(UnitComplex::new_unchecked(Complex::new(csv.x, csv.y)));
            }

            if compute_u {
                let cu = (m11 * csv.x + m12 * csv.y) / v1;
                let su = (m22 * csv.y) / v1;

                u = Some(UnitComplex::new_unchecked(Complex::new(cu, su)));
            }
        }

        (u, Vector2::new(v1, v2), v_t)
    }

    /*
    fn display_bidiag(b: &Bidiagonal<N, R, C>, begin: usize, end: usize) {
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
        b: &mut Bidiagonal<N, R, C>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        end: usize,
        eps: N,
    ) -> (usize, usize) {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if b.off_diagonal[m].is_zero()
                || b.off_diagonal[m].abs() <= eps * (b.diagonal[n].abs() + b.diagonal[m].abs())
            {
                b.off_diagonal[m] = N::zero();
            } else if b.diagonal[m].abs() <= eps {
                b.diagonal[m] = N::zero();
                Self::cancel_horizontal_off_diagonal_elt(b, u, v_t, m, m + 1);

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(b, u, v_t, m - 1);
                }
            } else if b.diagonal[n].abs() <= eps {
                b.diagonal[n] = N::zero();
                Self::cancel_vertical_off_diagonal_elt(b, u, v_t, m);
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

            if b.off_diagonal[m].abs() <= eps * (b.diagonal[new_start].abs() + b.diagonal[m].abs())
            {
                b.off_diagonal[m] = N::zero();
                break;
            }
            // FIXME: write a test that enters this case.
            else if b.diagonal[m].abs() <= eps {
                b.diagonal[m] = N::zero();
                Self::cancel_horizontal_off_diagonal_elt(b, u, v_t, m, n);

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(b, u, v_t, m - 1);
                }
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_horizontal_off_diagonal_elt(
        b: &mut Bidiagonal<N, R, C>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        i: usize,
        end: usize,
    ) {
        let mut v = Vector2::new(b.off_diagonal[i], b.diagonal[i + 1]);
        b.off_diagonal[i] = N::zero();

        for k in i..end {
            if let Some((rot, norm)) = givens::cancel_x(&v) {
                b.diagonal[k + 1] = norm;

                if b.is_upper_diagonal() {
                    if let Some(ref mut u) = *u {
                        rot.inverse()
                            .rotate_rows(&mut u.fixed_columns_with_step_mut::<U2>(i, k - i));
                    }
                } else if let Some(ref mut v_t) = *v_t {
                    rot.rotate(&mut v_t.fixed_rows_with_step_mut::<U2>(i, k - i));
                }

                if k + 1 != end {
                    v.x = -rot.sin_angle() * b.off_diagonal[k + 1];
                    v.y = b.diagonal[k + 2];
                    b.off_diagonal[k + 1] *= rot.cos_angle();
                }
            } else {
                break;
            }
        }
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_vertical_off_diagonal_elt(
        b: &mut Bidiagonal<N, R, C>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        i: usize,
    ) {
        let mut v = Vector2::new(b.diagonal[i], b.off_diagonal[i]);
        b.off_diagonal[i] = N::zero();

        for k in (0..i + 1).rev() {
            if let Some((rot, norm)) = givens::cancel_y(&v) {
                b.diagonal[k] = norm;

                if b.is_upper_diagonal() {
                    if let Some(ref mut v_t) = *v_t {
                        rot.rotate(&mut v_t.fixed_rows_with_step_mut::<U2>(k, i - k));
                    }
                } else if let Some(ref mut u) = *u {
                    rot.inverse()
                        .rotate_rows(&mut u.fixed_columns_with_step_mut::<U2>(k, i - k));
                }

                if k > 0 {
                    v.x = b.diagonal[k - 1];
                    v.y = rot.sin_angle() * b.off_diagonal[k - 1];
                    b.off_diagonal[k - 1] *= rot.cos_angle();
                }
            } else {
                break;
            }
        }
    }

    /// Computes the rank of the decomposed matrix, i.e., the number of singular values greater
    /// than `eps`.
    pub fn rank(&self, eps: N) -> usize {
        assert!(
            eps >= N::zero(),
            "SVD rank: the epsilon must be non-negative."
        );
        self.singular_values.iter().filter(|e| **e > eps).count()
    }

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the singular values have been manually modified.  Panics if the
    /// right- and left- singular vectors have not been computed at construction-time.
    pub fn recompose(self) -> MatrixMN<N, R, C> {
        let mut u = self.u.expect("SVD recomposition: U has not been computed.");
        let v_t = self.v_t
            .expect("SVD recomposition: V^t has not been computed.");

        for i in 0..self.singular_values.len() {
            let val = self.singular_values[i];
            u.column_mut(i).mul_assign(val);
        }

        u * v_t
    }

    /// Computes the pseudo-inverse of the decomposed matrix.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Panics if the right- and left- singular vectors have not been computed at
    /// construction-time.
    pub fn pseudo_inverse(mut self, eps: N) -> MatrixMN<N, C, R>
    where
        DefaultAllocator: Allocator<N, C, R>,
    {
        assert!(
            eps >= N::zero(),
            "SVD pseudo inverse: the epsilon must be non-negative."
        );
        for i in 0..self.singular_values.len() {
            let val = self.singular_values[i];

            if val > eps {
                self.singular_values[i] = N::one() / val;
            } else {
                self.singular_values[i] = N::zero();
            }
        }

        self.recompose().transpose()
    }

    /// Solves the system `self * x = b` where `self` is the decomposed matrix and `x` the unknown.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Returns `None` if the singular vectors `U` and `V` have not been computed.
    // FIXME: make this more generic wrt the storage types and the dimensions for `b`.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
        eps: N,
    ) -> MatrixMN<N, C, C2>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, C, C2> + Allocator<N, DimMinimum<R, C>, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        assert!(
            eps >= N::zero(),
            "SVD solve: the epsilon must be non-negative."
        );
        let u = self.u
            .as_ref()
            .expect("SVD solve: U has not been computed.");
        let v_t = self.v_t
            .as_ref()
            .expect("SVD solve: V^t has not been computed.");

        let mut ut_b = u.tr_mul(b);

        for j in 0..ut_b.ncols() {
            let mut col = ut_b.column_mut(j);

            for i in 0..self.singular_values.len() {
                let val = self.singular_values[i];
                if val > eps {
                    col[i] /= val;
                } else {
                    col[i] = N::zero();
                }
            }
        }

        v_t.tr_mul(&ut_b)
    }
}

impl<N: Real, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>,
{
    /// Computes the Singular Value Decomposition using implicit shift.
    pub fn svd(self, compute_u: bool, compute_v: bool) -> SVD<N, R, C> {
        SVD::new(self.into_owned(), compute_u, compute_v)
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// # Arguments
    ///
    /// * `compute_u` − set this to `true` to enable the computation of left-singular vectors.
    /// * `compute_v` − set this to `true` to enable the computation of left-singular vectors.
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_svd(
        self,
        compute_u: bool,
        compute_v: bool,
        eps: N,
        max_niter: usize,
    ) -> Option<SVD<N, R, C>> {
        SVD::try_new(self.into_owned(), compute_u, compute_v, eps, max_niter)
    }

    /// Computes the singular values of this matrix.
    pub fn singular_values(&self) -> VectorN<N, DimMinimum<R, C>> {
        SVD::new(self.clone_owned(), false, false).singular_values
    }

    /// Computes the rank of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    pub fn rank(&self, eps: N) -> usize {
        let svd = SVD::new(self.clone_owned(), false, false);
        svd.rank(eps)
    }

    /// Computes the pseudo-inverse of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    pub fn pseudo_inverse(self, eps: N) -> MatrixMN<N, C, R>
    where
        DefaultAllocator: Allocator<N, C, R>,
    {
        SVD::new(self.clone_owned(), true, true).pseudo_inverse(eps)
    }
}
