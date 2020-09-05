use num::Zero;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, VectorN};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum, U1};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

/// The QR decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct QR<N: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>,
{
    qr: MatrixMN<N, R, C>,
    tau: VectorN<N, DimMinimum<R, C>>,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for QR<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
{
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> QR<N, R, C>
where
    DefaultAllocator:
        Allocator<N, R, C> + Allocator<N, R> + Allocator<N, C> + Allocator<N, DimMinimum<R, C>>,
{
    /// Computes the QR decomposition using Householder reflections.
    pub fn new(mut matrix: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut tau = unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols, U1) };

        if min_nrows_ncols.value() == 0 {
            return QR { qr: matrix, tau };
        }

        let mut work = unsafe { MatrixMN::new_uninitialized_generic(ncols, U1) };
        for icol in 0..min_nrows_ncols.value() {
            let (mut left, mut right) = matrix.columns_range_pair_mut(icol, icol + 1..);
            let mut axis = left.rows_range_mut(icol..);
            // Compute the scaled Housholder vector
            let (beta, tau_i) = {
                let alpha = unsafe { *axis.vget_unchecked(0) };
                let xnorm = axis.rows_range(1..).norm();
                if xnorm.is_zero() && alpha.imaginary().is_zero() {
                    (alpha, N::zero())
                } else {
                    let a_r = alpha.real();
                    let a_i = alpha.imaginary();
                    // FIXME: Use LAPACK's ?LAPY3 once we have F::max
                    let reflection_norm = (a_r * a_r + a_i * a_i + xnorm * xnorm).sqrt();
                    // FIXME: Use reflection_norm.copysign(a_r)
                    let beta = -reflection_norm.abs() * a_r.signum();
                    // FIXME: Check for tiny beta and recompute, cf. LAPACK's ?LARFG
                    let tau = (N::from_real(beta) - alpha).unscale(beta);
                    let tmp = alpha - N::from_real(beta);
                    axis.rows_range_mut(1..).apply(|x| x / tmp);
                    (N::from_real(beta), tau)
                }
            };
            unsafe {
                *tau.vget_unchecked_mut(icol) = tau_i;
            }
            if !tau_i.is_zero() {
                // apply the Householder reflection to the remaining columns
                unsafe {
                    *axis.vget_unchecked_mut(0) = N::one();
                }
                let mut work = work.rows_range_mut(icol + 1..);
                work.gemv_ad(N::one(), &right.rows_range(icol..), &axis, N::zero());
                right
                    .rows_range_mut(icol..)
                    .gerc(-tau_i.conjugate(), &axis, &work, N::one());
            }
            unsafe {
                *axis.vget_unchecked_mut(0) = beta;
            }
        }

        QR { qr: matrix, tau }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    pub fn r(&self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.data.shape();
        self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<N, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.data.shape();
        let mut res = self.qr.resize_generic(nrows.min(ncols), ncols, N::zero());
        res.fill_lower_triangle(N::zero(), 1);
        res
    }

    /// Compute the first `ncols` columns of `Q`.
    ///
    /// Panics if `ncols` is bigger than the number of rows of the original matrix.
    pub fn q_columns<K: Dim>(&self, ncols: K) -> MatrixMN<N, R, K>
    where
        DefaultAllocator: Allocator<N, R, K> + Allocator<N, K>,
    {
        let (q_nrows, q_ncols) = self.qr.data.shape();
        assert!(
            ncols.value() <= q_nrows.value(),
            "k must be less than the number of columns"
        );
        let mut a = MatrixMN::<N, R, K>::identity_generic(q_nrows, ncols);
        let mut work = unsafe { VectorN::<N, K>::new_uninitialized_generic(ncols, U1) };
        let k = q_nrows.min(q_ncols);
        let ncols_to_copy = k.value().min(q_ncols.value());
        a.slice_range_mut(.., ..ncols_to_copy)
            .copy_from(&self.qr.slice_range(.., ..ncols_to_copy));
        for i in (0..k.value()).rev() {
            let tau_i = unsafe { *self.tau.vget_unchecked(i) };
            if i < q_ncols.value() {
                unsafe {
                    *a.get_unchecked_mut((i, i)) = N::one();
                }
                let (left, mut right) = a.columns_range_pair_mut(i, i + 1..);
                let axis = left.rows_range(i..);
                let mut work = work.rows_range_mut(i + 1..);
                work.gemv_ad(N::one(), &right.rows_range(i..), &axis, N::zero());
                right
                    .rows_range_mut(i..)
                    .gerc(-tau_i, &axis, &work, N::one());
            }
            if i < q_nrows.value() {
                a.slice_range_mut(i + 1.., i).apply(|x| x * (-tau_i));
            }
            unsafe {
                *a.get_unchecked_mut((i, i)) = N::one() - tau_i;
            }
            a.slice_range_mut(0..i, i).fill(N::zero());
        }
        a
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixMN<N, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<N, R, DimMinimum<R, C>> + Allocator<N, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.qr.data.shape();
        self.q_columns::<DimMinimum<R, C>>(nrows.min(ncols))
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<N, R, DimMinimum<R, C>>
            + Allocator<N, DimMinimum<R, C>>
            + Reallocator<N, R, C, DimMinimum<R, C>, C>,
    {
        (self.q(), self.unpack_r())
    }

    #[doc(hidden)]
    pub fn qr_internal(&self) -> &MatrixMN<N, R, C> {
        &self.qr
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&mut self, rhs: &mut Matrix<N, R2, C2, S2>)
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, R>,
        DefaultAllocator: Allocator<N, C2>,
    {
        let ncols = rhs.data.shape().1;
        let mut work = unsafe { VectorN::<N, C2>::new_uninitialized_generic(ncols, U1) };
        for i in 0..self.tau.len() {
            let mut axis = self.qr.slice_range_mut(i.., i);
            let temp = unsafe { *axis.vget_unchecked(0) };
            unsafe {
                *axis.vget_unchecked_mut(0) = N::one();
            }
            let tau_i = unsafe { *self.tau.vget_unchecked(i) };
            work.gemv_ad(N::one(), &rhs.rows_range(i..), &axis, N::zero());
            rhs.rows_range_mut(i..)
                .gerc(-tau_i.conjugate(), &axis, &work, N::one());
            unsafe {
                *axis.vget_unchecked_mut(0) = temp;
            }
        }
    }
}

impl<N: ComplexField, D: DimMin<D, Output = D>> QR<N, D, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &mut self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<N, C2>,
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
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&mut self, b: &mut Matrix<N, R2, C2, S2>) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        DefaultAllocator: Allocator<N, C2>,
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
    pub fn try_inverse(&mut self) -> Option<MatrixN<N, D>> {
        assert!(
            self.qr.is_square(),
            "QR inverse: unable to compute the inverse of a non-square matrix."
        );

        // FIXME: is there a less naive method ?
        let (nrows, ncols) = self.qr.data.shape();
        let mut res = MatrixN::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.qr.is_square(),
            "QR: unable to test the invertibility of a non-square matrix."
        );
        (0..self.qr.ncols()).all(|i| unsafe { !self.qr.get_unchecked((i, i)).is_zero() })
    }
}

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    DefaultAllocator:
        Allocator<N, R, C> + Allocator<N, R> + Allocator<N, C> + Allocator<N, DimMinimum<R, C>>,
{
    /// Computes the QR decomposition of this matrix.
    pub fn qr(self) -> QR<N, R, C> {
        QR::new(self.into_owned())
    }
}
