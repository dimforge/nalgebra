use num::Zero;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Unit, VectorN};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum, U1};
use crate::storage::{Storage, StorageMut};
use crate::ComplexField;

use crate::geometry::Reflection;
use crate::linalg::{householder, PermutationSequence};

/// The QR decomposition (with column pivoting) of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct ColPivQR<N: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    col_piv_qr: MatrixMN<N, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    diag: VectorN<N, DimMinimum<R, C>>,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for ColPivQR<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
{
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> ColPivQR<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, R>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    /// Computes the ColPivQR decomposition using householder reflections.
    pub fn new(mut matrix: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        let mut diag = unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols, U1) };

        if min_nrows_ncols.value() == 0 {
            return ColPivQR {
                col_piv_qr: matrix,
                p,
                diag,
            };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.slice_range(i.., i..).icamax_full();
            let col_piv = piv.1 + i;
            matrix.swap_columns(i, col_piv);
            p.append_permutation(i, col_piv);

            householder::clear_column_unchecked(&mut matrix, &mut diag[i], i, 0, None);
        }

        ColPivQR {
            col_piv_qr: matrix,
            p,
            diag,
        }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    pub fn r(&self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();
        let mut res = self
            .col_piv_qr
            .rows_generic(0, nrows.min(ncols))
            .upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<N, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();
        let mut res = self
            .col_piv_qr
            .resize_generic(nrows.min(ncols), ncols, N::zero());
        res.fill_lower_triangle(N::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixMN<N, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<N, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        for i in (0..dim).rev() {
            let axis = self.col_piv_qr.slice_range(i.., i);
            // TODO: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

            let mut res_rows = res.slice_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].signum());
        }

        res
    }
    /// Retrieves the column permutation of this decomposition.
    #[inline]
    pub fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<N, R, DimMinimum<R, C>>
            + Reallocator<N, R, C, DimMinimum<R, C>, C>
            + Allocator<(usize, usize), DimMinimum<R, C>>,
    {
        (self.q(), self.r(), self.p)
    }

    #[doc(hidden)]
    pub fn col_piv_qr_internal(&self) -> &MatrixMN<N, R, C> {
        &self.col_piv_qr
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
    where
        S2: StorageMut<N, R2, C2>,
    {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.col_piv_qr.slice_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].signum().conjugate());
        }
    }
}

impl<N: ComplexField, D: DimMin<D, Output = D>> ColPivQR<N, D, D>
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, D> + Allocator<(usize, usize), DimMinimum<D, D>>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<N, R2, C2>,
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
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.col_piv_qr.nrows(),
            b.nrows(),
            "ColPivQR solve matrix dimension mismatch."
        );
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        let solved = self.solve_upper_triangular_mut(b);
        self.p.inv_permute_rows(b);

        solved
    }

    // TODO: duplicate code from the `solve` module.
    fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.col_piv_qr.nrows();

        for k in 0..b.ncols() {
            let mut b = b.column_mut(k);
            for i in (0..dim).rev() {
                let coeff;

                unsafe {
                    let diag = self.diag.vget_unchecked(i).modulus();

                    if diag.is_zero() {
                        return false;
                    }

                    coeff = b.vget_unchecked(i).unscale(diag);
                    *b.vget_unchecked_mut(i) = coeff;
                }

                b.rows_range_mut(..i)
                    .axpy(-coeff, &self.col_piv_qr.slice_range(..i, i), N::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    pub fn try_inverse(&self) -> Option<MatrixN<N, D>> {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.col_piv_qr.data.shape();
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
            self.col_piv_qr.is_square(),
            "ColPivQR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    /// Computes the determinant of the decomposed matrix.
    pub fn determinant(&self) -> N {
        let dim = self.col_piv_qr.nrows();
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = N::one();
        for i in 0..dim {
            res *= unsafe { *self.diag.vget_unchecked(i) };
        }

        res * self.p.determinant()
    }
}

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, R>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    /// Computes the QR decomposition (with column pivoting) of this matrix.
    pub fn col_piv_qr(self) -> ColPivQR<N, R, C> {
        ColPivQR::new(self.into_owned())
    }
}
