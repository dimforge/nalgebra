#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ComplexField;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, SquareMatrix};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimSub, Dynamic, U1};
use crate::storage::{Storage, StorageMut};

/// The Cholesky decomposition of a symmetric-definite-positive matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<N, D>,
         MatrixN<N, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<N, D>,
         MatrixN<N, D>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct Cholesky<N: ComplexField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    chol: MatrixN<N, D>,
}

impl<N: ComplexField, D: Dim> Copy for Cholesky<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    MatrixN<N, D>: Copy,
{
}

impl<N: ComplexField, D: DimSub<Dynamic>> Cholesky<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Attempts to compute the Cholesky decomposition of `matrix`.
    ///
    /// Returns `None` if the input matrix is not definite-positive. The input matrix is assumed
    /// to be symmetric and only the lower-triangular part is read.
    pub fn new(mut matrix: MatrixN<N, D>) -> Option<Self> {
        assert!(matrix.is_square(), "The input matrix must be square.");

        let n = matrix.nrows();

        for j in 0..n {
            for k in 0..j {
                let factor = unsafe { -*matrix.get_unchecked((j, k)) };

                let (mut col_j, col_k) = matrix.columns_range_pair_mut(j, k);
                let mut col_j = col_j.rows_range_mut(j..);
                let col_k = col_k.rows_range(j..);

                col_j.axpy(factor.conjugate(), &col_k, N::one());
            }

            let diag = unsafe { *matrix.get_unchecked((j, j)) };
            if !diag.is_zero() {
                if let Some(denom) = diag.try_sqrt() {
                    unsafe {
                        *matrix.get_unchecked_mut((j, j)) = denom;
                    }

                    let mut col = matrix.slice_range_mut(j + 1.., j);
                    col /= denom;
                    continue;
                }
            }

            // The diagonal element is either zero or its square root could not
            // be taken (e.g. for negative real numbers).
            return None;
        }

        Some(Cholesky { chol: matrix })
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition with its strictly
    /// upper-triangular part filled with zeros.
    pub fn unpack(mut self) -> MatrixN<N, D> {
        self.chol.fill_upper_triangle(N::zero(), 1);
        self.chol
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// The values of the strict upper-triangular part are garbage and should be ignored by further
    /// computations.
    pub fn unpack_dirty(self) -> MatrixN<N, D> {
        self.chol
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition with its strictly
    /// uppen-triangular part filled with zeros.
    pub fn l(&self) -> MatrixN<N, D> {
        self.chol.lower_triangle()
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// This is an allocation-less version of `self.l()`. The values of the strict upper-triangular
    /// part are garbage and should be ignored by further computations.
    pub fn l_dirty(&self) -> &MatrixN<N, D> {
        &self.chol
    }

    /// Solves the system `self * x = b` where `self` is the decomposed matrix and `x` the unknown.
    ///
    /// The result is stored on `b`.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>)
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let _ = self.chol.solve_lower_triangular_mut(b);
        let _ = self.chol.ad_solve_lower_triangular_mut(b);
    }

    /// Returns the solution of the system `self * x = b` where `self` is the decomposed matrix and
    /// `x` the unknown.
    pub fn solve<R2: Dim, C2: Dim, S2>(&self, b: &Matrix<N, R2, C2, S2>) -> MatrixMN<N, R2, C2>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.solve_mut(&mut res);
        res
    }

    /// Computes the inverse of the decomposed matrix.
    pub fn inverse(&self) -> MatrixN<N, D> {
        let shape = self.chol.data.shape();
        let mut res = MatrixN::identity_generic(shape.0, shape.1);

        self.solve_mut(&mut res);
        res
    }

    /// Given the Cholesky decomposition of a matrix `M`, a scalar `sigma` and a vector `v`,
    /// performs a rank one update such that we end up with the decomposition of `M + sigma * v*v.adjoint()`.
    pub fn rank_one_update<R2: Dim, S2>(&mut self, x: &Matrix<N, R2, U1, S2>, sigma: N::RealField)
    where
        S2: Storage<N, R2, U1>,
        DefaultAllocator: Allocator<N, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        // heavily inspired by Eigen's `llt_rank_update_lower` implementation https://eigen.tuxfamily.org/dox/LLT_8h_source.html
        let n = x.nrows();
        let mut x = x.clone_owned();
        let mut beta = crate::one::<N::RealField>();
        for j in 0..n {
            let diag = N::real(unsafe { *self.chol.get_unchecked((j, j)) });
            let diag2 = diag * diag;
            let xj = unsafe { *x.get_unchecked(j) };
            let sigma_xj2 = sigma * N::modulus_squared(xj);
            let gamma = diag2 * beta + sigma_xj2;
            let new_diag = (diag2 + sigma_xj2 / beta).sqrt();
            unsafe { *self.chol.get_unchecked_mut((j, j)) = N::from_real(new_diag) };
            beta += sigma_xj2 / diag2;
            // Update the terms of L
            if j < n {
                let mut xjplus = x.rows_range_mut(j + 1..);
                let mut col_j = self.chol.slice_range_mut(j + 1.., j);
                // temp_jplus -= (wj / N::from_real(diag)) * col_j;
                xjplus.axpy(-xj / N::from_real(diag), &col_j, N::one());
                if gamma != crate::zero::<N::RealField>() {
                    // col_j = N::from_real(nljj / diag) * col_j  + (N::from_real(nljj * sigma / gamma) * N::conjugate(wj)) * temp_jplus;
                    col_j.axpy(
                        N::from_real(new_diag * sigma / gamma) * N::conjugate(xj),
                        &xjplus,
                        N::from_real(new_diag / diag),
                    );
                }
            }
        }
    }
}

impl<N: ComplexField, D: DimSub<Dynamic>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Attempts to compute the Cholesky decomposition of this matrix.
    ///
    /// Returns `None` if the input matrix is not definite-positive. The input matrix is assumed
    /// to be symmetric and only the lower-triangular part is read.
    pub fn cholesky(self) -> Option<Cholesky<N, D>> {
        Cholesky::new(self.into_owned())
    }
}
