use num::Zero;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, OMatrix, OVector};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Const, Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

use std::mem::MaybeUninit;

/// The QR decomposition of a general matrix.
///
/// The decomposition is stored like in LAPACK's `?GEQR2`: the upper trapezoidal part of `qr` is
/// `R`, and the columns below its diagonal hold the Householder vectors `v` (their first component
/// is `1` and is not stored). `Q` is the product `H(0) * H(1) * ... ` of the reflections
/// `H(i) = I - tau(i) * v * v.adjoint()`.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct QR<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    qr: OMatrix<T, R, C>,
    tau: OVector<T, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<C> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the QR decomposition using householder reflections.
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return QR {
                qr: matrix,
                tau: Matrix::zeros_generic(min_nrows_ncols, Const::<1>),
            };
        }

        let mut tau = Matrix::uninit(min_nrows_ncols, Const::<1>);
        let mut work = Matrix::zeros_generic(ncols, Const::<1>);

        for i in 0..min_nrows_ncols.value() {
            let (mut left, mut right) = matrix.columns_range_pair_mut(i, i + 1..);
            let mut axis = left.rows_range_mut(i..);

            // Compute the scaled Householder vector, cf. LAPACK's `?LARFG`.
            let (beta, tau_i) = {
                let alpha = unsafe { axis.vget_unchecked(0).clone() };
                let xnorm = axis.rows_range(1..).norm();

                if xnorm.is_zero() && alpha.clone().imaginary().is_zero() {
                    // The column is already in the wanted form.
                    (alpha, T::zero())
                } else {
                    let a_r = alpha.clone().real();
                    let a_i = alpha.clone().imaginary();
                    // TODO: use LAPACK's `?LAPY3` once `RealField` has a `max` method.
                    let reflection_norm =
                        (a_r.clone() * a_r.clone() + a_i.clone() * a_i + xnorm.clone() * xnorm)
                            .sqrt();
                    // TODO: use `reflection_norm.copysign(a_r)`.
                    let beta = -reflection_norm.abs() * a_r.signum();
                    // TODO: rescale if `beta` is close to underflow, cf. LAPACK's `?LARFG`.
                    let tau_i = (T::from_real(beta.clone()) - alpha.clone()).unscale(beta.clone());
                    // Scale the Householder vector such that its first component is `1`.
                    let tmp = alpha - T::from_real(beta.clone());
                    axis.rows_range_mut(1..).apply(|x| *x /= tmp.clone());

                    (T::from_real(beta), tau_i)
                }
            };

            tau[i] = MaybeUninit::new(tau_i.clone());

            if !tau_i.is_zero() {
                // Apply the Householder reflection to the remaining columns.
                unsafe {
                    *axis.vget_unchecked_mut(0) = T::one();
                }

                let mut work = work.rows_range_mut(i + 1..);
                work.gemv_ad(T::one(), &right.rows_range(i..), &axis, T::zero());
                right
                    .rows_range_mut(i..)
                    .gerc(-tau_i.conjugate(), &axis, &work, T::one());
            }

            unsafe {
                *axis.vget_unchecked_mut(0) = beta;
            }
        }

        // Safety: tau is now fully initialized.
        let tau = unsafe { tau.assume_init() };
        QR { qr: matrix, tau }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let mut res = self.qr.resize_generic(nrows.min(ncols), ncols, T::zero());
        res.fill_lower_triangle(T::zero(), 1);
        res
    }

    /// Computes the first `ncols` columns of the orthogonal matrix `Q` of this decomposition.
    ///
    /// Use this to get the full `Q` of a tall matrix: `q_columns` accepts any `ncols` up to the
    /// number of rows of the decomposed matrix, while [`QR::q`] returns the first
    /// `min(nrows, ncols)` columns only.
    ///
    /// # Panics
    /// Panics if `ncols` is bigger than the number of rows of the decomposed matrix.
    #[must_use]
    pub fn q_columns<K: Dim>(&self, ncols: K) -> OMatrix<T, R, K>
    where
        DefaultAllocator: Allocator<R, K> + Allocator<K>,
    {
        // This is LAPACK's `?ORG2R`.
        let (q_nrows, q_ncols) = self.qr.shape_generic();
        assert!(
            ncols.value() <= q_nrows.value(),
            "The number of columns of Q cannot be bigger than the number of rows of the decomposed matrix."
        );

        let mut a = OMatrix::<T, R, K>::identity_generic(q_nrows, ncols);
        let mut work = Matrix::zeros_generic(ncols, Const::<1>);
        // The reflections after the first `k` ones do not change the first `ncols` columns.
        let k = q_nrows.value().min(q_ncols.value()).min(ncols.value());

        a.view_range_mut(.., ..k)
            .copy_from(&self.qr.view_range(.., ..k));

        for i in (0..k).rev() {
            let tau_i = unsafe { self.tau.vget_unchecked(i).clone() };

            if i + 1 < ncols.value() {
                // Apply the reflection to the columns computed so far.
                unsafe {
                    *a.get_unchecked_mut((i, i)) = T::one();
                }

                let (left, mut right) = a.columns_range_pair_mut(i, i + 1..);
                let axis = left.rows_range(i..);
                let mut work = work.rows_range_mut(i + 1..);
                work.gemv_ad(T::one(), &right.rows_range(i..), &axis, T::zero());
                right
                    .rows_range_mut(i..)
                    .gerc(-tau_i.clone(), &axis, &work, T::one());
            }

            if i + 1 < q_nrows.value() {
                a.view_range_mut(i + 1.., i).apply(|x| *x *= -tau_i.clone());
            }

            unsafe {
                *a.get_unchecked_mut((i, i)) = T::one() - tau_i;
            }
            a.view_range_mut(..i, i).fill(T::zero());
        }

        a
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>> + Allocator<DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        self.q_columns(nrows.min(ncols))
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        (self.q(), self.unpack_r())
    }

    #[doc(hidden)]
    pub const fn qr_internal(&self) -> &OMatrix<T, R, C> {
        &self.qr
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, R>,
    {
        for i in 0..self.tau.len() {
            let tau_i = unsafe { self.tau.vget_unchecked(i).clone() };

            if tau_i.is_zero() {
                continue;
            }

            // The first component of the Householder vector is `1` and is not stored.
            let axis = self.qr.view_range(i + 1.., i);

            for j in 0..rhs.ncols() {
                let mut col = rhs.column_mut(j);
                let dot =
                    unsafe { col.vget_unchecked(i).clone() } + axis.dotc(&col.rows_range(i + 1..));
                let factor = -(tau_i.clone().conjugate() * dot);

                unsafe {
                    *col.vget_unchecked_mut(i) += factor.clone();
                }
                col.rows_range_mut(i + 1..).axpy(factor, &axis, T::one());
            }
        }
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> QR<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<R2, C2>,
    {
        let mut res = b.clone_owned();

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` is
    /// overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.qr.nrows(),
            b.nrows(),
            "QR solve matrix dimension mismatch."
        );
        assert!(
            self.qr.is_square(),
            "QR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        self.qr.solve_upper_triangular_mut(b)
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.qr.is_square(),
            "QR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.qr.shape_generic();
        let mut res = OMatrix::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.qr.is_square(),
            "QR: unable to test the invertibility of a non-square matrix."
        );
        (0..self.qr.ncols()).all(|i| unsafe { !self.qr.get_unchecked((i, i)).is_zero() })
    }

    // /// Computes the determinant of the decomposed matrix.
    // pub fn determinant(&self) -> T {
    //     let dim = self.qr.nrows();
    //     assert!(self.qr.is_square(), "QR determinant: unable to compute the determinant of a non-square matrix.");

    //     let mut res = T::one();
    //     for i in 0 .. dim {
    //         res *= unsafe { *self.diag.vget_unchecked(i) };
    //     }

    //     res self.q_determinant()
    // }
}
