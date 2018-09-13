#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

use alga::general::Real;
use allocator::Allocator;
use base::{DefaultAllocator, MatrixMN, MatrixN, SquareMatrix, VectorN};
use dimension::{DimDiff, DimSub, U1};
use storage::Storage;

use linalg::householder;

/// Tridiagonalization of a symmetric matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            serialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Serialize,
         VectorN<N, DimDiff<D, U1>>: Serialize"
        )
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            deserialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Deserialize<'de>,
         VectorN<N, DimDiff<D, U1>>: Deserialize<'de>"
        )
    )
)]
#[derive(Clone, Debug)]
pub struct SymmetricTridiagonal<N: Real, D: DimSub<U1>>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
{
    tri: MatrixN<N, D>,
    off_diagonal: VectorN<N, DimDiff<D, U1>>,
}

impl<N: Real, D: DimSub<U1>> Copy for SymmetricTridiagonal<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
    MatrixN<N, D>: Copy,
    VectorN<N, DimDiff<D, U1>>: Copy,
{
}

impl<N: Real, D: DimSub<U1>> SymmetricTridiagonal<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
{
    /// Computes the tridiagonalization of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    pub fn new(mut m: MatrixN<N, D>) -> Self {
        let dim = m.data.shape().0;

        assert!(
            m.is_square(),
            "Unable to compute the symmetric tridiagonal decomposition of a non-square matrix."
        );
        assert!(
            dim.value() != 0,
            "Unable to compute the symmetric tridiagonal decomposition of an empty matrix."
        );

        let mut off_diagonal = unsafe { MatrixMN::new_uninitialized_generic(dim.sub(U1), U1) };
        let mut p = unsafe { MatrixMN::new_uninitialized_generic(dim.sub(U1), U1) };

        for i in 0..dim.value() - 1 {
            let mut m = m.rows_range_mut(i + 1..);
            let (mut axis, mut m) = m.columns_range_pair_mut(i, i + 1..);

            let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);
            off_diagonal[i] = norm;

            if not_zero {
                let mut p = p.rows_range_mut(i..);

                p.gemv_symm(::convert(2.0), &m, &axis, N::zero());
                let dot = axis.dot(&p);
                p.axpy(-dot, &axis, N::one());
                m.ger_symm(-N::one(), &p, &axis, N::one());
                m.ger_symm(-N::one(), &axis, &p, N::one());
            }
        }

        SymmetricTridiagonal {
            tri: m,
            off_diagonal: off_diagonal,
        }
    }

    #[doc(hidden)]
    // For debugging.
    pub fn internal_tri(&self) -> &MatrixN<N, D> {
        &self.tri
    }

    /// Retrieve the orthogonal transformation, diagonal, and off diagonal elements of this
    /// decomposition.
    pub fn unpack(self) -> (MatrixN<N, D>, VectorN<N, D>, VectorN<N, DimDiff<D, U1>>)
    where
        DefaultAllocator: Allocator<N, D>,
    {
        let diag = self.diagonal();
        let q = self.q();

        (q, diag, self.off_diagonal)
    }

    /// Retrieve the diagonal, and off diagonal elements of this decomposition.
    pub fn unpack_tridiagonal(self) -> (VectorN<N, D>, VectorN<N, DimDiff<D, U1>>)
    where
        DefaultAllocator: Allocator<N, D>,
    {
        let diag = self.diagonal();

        (diag, self.off_diagonal)
    }

    /// The diagonal components of this decomposition.
    pub fn diagonal(&self) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        self.tri.diagonal()
    }

    /// The off-diagonal components of this decomposition.
    pub fn off_diagonal(&self) -> &VectorN<N, DimDiff<D, U1>>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        &self.off_diagonal
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixN<N, D> {
        householder::assemble_q(&self.tri)
    }

    /// Recomputes the original symmetric matrix.
    pub fn recompose(mut self) -> MatrixN<N, D> {
        let q = self.q();
        self.tri.fill_lower_triangle(N::zero(), 2);
        self.tri.fill_upper_triangle(N::zero(), 2);

        for i in 0..self.off_diagonal.len() {
            self.tri[(i + 1, i)] = self.off_diagonal[i];
            self.tri[(i, i + 1)] = self.off_diagonal[i];
        }

        &q * self.tri * q.transpose()
    }
}

impl<N: Real, D: DimSub<U1>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
{
    /// Computes the tridiagonalization of this symmetric matrix.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    pub fn symmetric_tridiagonalize(self) -> SymmetricTridiagonal<N, D> {
        SymmetricTridiagonal::new(self.into_owned())
    }
}
