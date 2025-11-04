use super::qr::{QrReal, QrScalar};
use crate::qr::QrDecomposition;
use crate::qr_util;
use crate::sealed::Sealed;
use na::{Const, IsContiguous, Matrix, OVector, RealField, Vector};
use nalgebra::storage::RawStorageMut;
use nalgebra::{DefaultAllocator, Dim, DimMin, DimMinimum, OMatrix, Scalar, allocator::Allocator};
use num::float::TotalOrder;
use num::{Float, Zero};
use rank::{RankDeterminationAlgorithm, calculate_rank};

pub use qr_util::Error;
mod permutation;
#[cfg(test)]
mod test;
pub use permutation::Permutation;
/// Utility functionality to calculate the rank of matrices.
mod rank;

/// The column-pivoted QR decomposition of a rectangular matrix `A ∈ R^(m × n)`
/// with `m >= n`.
///
/// The columns of the matrix `A` are permuted such that `A P = Q R`, meaning
/// the column-permuted `A` is the product of `Q` and `R`, where `Q` is an orthonormal
/// matrix `Q^T Q = I` and `R` is upper triangular.
///
/// Note that most of the functionality is provided via the [`QrDecomposition`]
/// trait, which must be in scope for its functions to be used.
#[derive(Debug, Clone)]
pub struct ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: Scalar,
    R: DimMin<C, Output = C>,
    C: Dim,
{
    // QR decomposition, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    qr: OMatrix<T, R, C>,
    // Householder coefficients, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    tau: OVector<T, DimMinimum<R, C>>,
    // Permutation vector, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    // Note that permutation indices are 1-based in LAPACK
    jpvt: OVector<i32, C>,
    // Rank of the matrix
    rank: i32,
}

impl<T, R, C> Sealed for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: Scalar,
    R: DimMin<C, Output = C>,
    C: Dim,
{
}

/// Constructors
impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: QrScalar + Zero + RealField + TotalOrder + Float,
    R: DimMin<C, Output = C>,
    C: Dim,
{
    /// Try to create a new decomposition from the given matrix using the default
    /// strategy for rank determination of a matrix from its QR decomposition.
    pub fn new(m: OMatrix<T, R, C>) -> Result<Self, Error> {
        Self::with_rank_algo(m, Default::default())
    }

    /// Try to create a new decomposition from the given matrix and specify the
    /// strategy for rank determination. When in doubt, use the default strategy
    /// via the [`ColPivQR::new`] constructor.
    pub fn with_rank_algo(
        mut m: OMatrix<T, R, C>,
        rank_algo: RankDeterminationAlgorithm<T>,
    ) -> Result<Self, Error> {
        let (nrows, ncols) = m.shape_generic();

        if nrows.value() < ncols.value() {
            return Err(Error::Underdetermined);
        }

        let mut tau: OVector<T, DimMinimum<R, C>> =
            Vector::zeros_generic(nrows.min(ncols), Const::<1>);
        let mut jpvt: OVector<i32, C> = Vector::zeros_generic(ncols, Const::<1>);

        // SAFETY: matrix dimensions are slice dimensions, other inputs are according
        // to spec, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
        let lwork = unsafe {
            T::xgeqp3_work_size(
                nrows.value().try_into().expect("matrix dims out of bounds"),
                ncols.value().try_into().expect("matrix dims out of bounds"),
                m.as_mut_slice(),
                nrows.value().try_into().expect("matrix dims out of bounds"),
                jpvt.as_mut_slice(),
                tau.as_mut_slice(),
            )?
        };

        let mut work = vec![T::zero(); lwork as usize];

        // SAFETY: matrix dimensions are slice dimensions, other inputs are according
        // to spec, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
        unsafe {
            T::xgeqp3(
                nrows.value() as i32,
                ncols.value() as i32,
                m.as_mut_slice(),
                nrows.value() as i32,
                jpvt.as_mut_slice(),
                tau.as_mut_slice(),
                &mut work,
                lwork,
            )?;
        }

        let rank: i32 = calculate_rank(&m, rank_algo)
            .try_into()
            .map_err(|_| Error::Dimensions)?;

        Ok(Self {
            qr: m,
            rank,
            tau,
            jpvt,
        })
    }
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: QrScalar + Zero + RealField,
    R: DimMin<C, Output = C>,
    C: Dim,
{
    /// get the effective rank of the matrix computed using the stratey
    /// chosen at construction.
    #[inline]
    pub fn rank(&self) -> u16 {
        self.rank as u16
    }
    /// obtain the permutation `P` such that the `A P = Q R` ,
    /// meaning the column-permuted original matrix `A` is identical to
    /// `Q R`. This function performs a small allocation.
    pub fn p(&self) -> Permutation<C> {
        Permutation::new(self.jpvt.clone())
    }
}

impl<T, R, C> QrDecomposition<T, R, C> for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    R: DimMin<C, Output = C>,
    C: Dim,
    T: Scalar + RealField + QrReal,
{
    fn __lapack_qr_ref(&self) -> &OMatrix<T, R, C> {
        &self.qr
    }

    fn __lapack_tau_ref(&self) -> &OVector<T, DimMinimum<R, C>> {
        &self.tau
    }

    fn solve_mut<C2: Dim, S, S2>(
        &self,
        x: &mut Matrix<T, C, C2, S2>,
        b: Matrix<T, R, C2, S>,
    ) -> Result<(), Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous,
        S2: RawStorageMut<T, C, C2> + IsContiguous,
        T: Zero,
    {
        if self.nrows() < self.ncols() {
            return Err(Error::Underdetermined);
        }
        let rank = self.rank();
        qr_util::qr_solve_mut_with_rank_unpermuted(&self.qr, &self.tau, rank, x, b)?;
        self.p().permute_rows_mut(x)?;
        Ok(())
    }
}
