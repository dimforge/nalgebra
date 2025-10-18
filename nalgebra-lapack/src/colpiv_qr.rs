use super::qr::{QRReal, QrScalar};
use crate::sealed::Sealed;
use crate::{ComplexHelper, DiagonalKind, Side, Transposition, TriangularStructure, qr_util};
use crate::{LapackErrorCode, lapack_error::check_lapack_info};
use na::{ComplexField, Const, IsContiguous, Matrix, OVector, RealField, Storage, Vector};
use nalgebra::storage::RawStorageMut;
use nalgebra::{DefaultAllocator, Dim, DimMin, DimMinimum, OMatrix, Scalar, allocator::Allocator};
use num::float::TotalOrder;
use num::{Float, Zero};
use rank::{RankDeterminationAlgorithm, calculate_rank};

pub use qr_util::Error;
mod permutation;
#[cfg(test)]
mod test;
mod utility;
pub use permutation::Permutation;
/// utility functionality to calculate the rank of matrices
mod rank;

/// The column-pivoted QR-decomposition of a rectangular matrix `A ∈ R^(m ⨯ n)`
/// with `m >= n`.
///
/// The columns of the matrix `A` are permuted such that `A P = Q R`, meaning
/// the column-permuted `A` is the product of `Q` and `R`, where `Q` is an orthonormal
/// matrix `Q^T Q = I` and `R` is upper triangular.
#[derive(Debug, Clone)]
pub struct ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: Scalar + ComplexField,
    R: DimMin<C>,
    C: Dim,
{
    // qr decomposition, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    qr: OMatrix<T, R, C>,
    // householder coefficients, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    tau: OVector<T, DimMinimum<R, C>>,
    // permutation vector, see https://www.netlib.org/lapack/explore-html/d0/dea/group__geqp3.html
    // note that permutation indices are 1-based in LAPACK
    jpvt: OVector<i32, C>,
    // rank of the matrix
    rank: i32,
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrScalar + Zero + RealField + TotalOrder + Float,
    R: DimMin<C>,
    C: Dim,
{
    /// Try to create a new decomposition from the given matrix using the default
    /// strategy for rank determination of a matrix from its QR decomposition.
    pub fn new(m: OMatrix<T, R, C>) -> Result<Self, Error> {
        Self::with_rank_algo(m, Default::default())
    }

    /// Try to create a new decomposition from the given matrix and specify the
    /// strategy for rank determination. When in doubt, use the default strategy
    /// via the [ColPivQR::new]  constructor.
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
    T: ColPivQrScalar + Zero + RealField,
    R: DimMin<C>,
    C: Dim,
{
    /// get the effective rank of the matrix computed using the stratey
    /// chosen at construction.
    #[inline]
    pub fn rank(&self) -> u16 {
        self.rank as u16
    }

    #[inline]
    /// the number of rows of the original matrix `A`
    pub fn nrows(&self) -> usize {
        self.qr.nrows()
    }

    #[inline]
    /// the number of columns of the original matrix `A`
    pub fn ncols(&self) -> usize {
        self.qr.ncols()
    }

    /// obtain the permutation `P` such that the `A P = Q R` ,
    /// meaning the column-permuted original matrix `A` is identical to
    /// `Q R`. This function performs a small allocation.
    pub fn p(&self) -> Permutation<C> {
        Permutation::new(self.jpvt.clone())
    }
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrReal + Zero + RealField,
    R: DimMin<C>,
    C: Dim,
{
    /// Efficiently calculate the matrix product `Q B` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    pub fn q_mul_mut<C2, S>(&self, b: &mut Matrix<T, R, C2, S>) -> Result<(), Error>
    where
        C2: Dim,
        S: RawStorageMut<T, R, C2> + IsContiguous,
    {
        qr_util::q_mul_mut(&self.qr, &self.tau, b)?;
        Ok(())
    }

    /// Efficiently calculate the matrix product `Q^T B` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    pub fn q_tr_mul_mut<C2, S>(&self, b: &mut Matrix<T, R, C2, S>) -> Result<(), Error>
    where
        C2: Dim,
        S: RawStorageMut<T, R, C2> + IsContiguous,
    {
        qr_util::q_tr_mul_mut(&self.qr, &self.tau, b)?;
        Ok(())
    }

    /// Efficiently calculate the matrix product `B Q` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    pub fn mul_q_mut<R2, S>(&self, b: &mut Matrix<T, R2, R, S>) -> Result<(), Error>
    where
        R2: Dim,
        S: RawStorageMut<T, R2, R> + IsContiguous,
    {
        qr_util::mul_q_mut(&self.qr, &self.tau, b)?;
        Ok(())
    }

    /// Efficiently calculate the matrix product `B Q^T` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    pub fn mul_q_tr_mut<R2, S>(&self, b: &mut Matrix<T, R2, R, S>) -> Result<(), Error>
    where
        R2: Dim,
        S: RawStorageMut<T, R2, R> + IsContiguous,
    {
        qr_util::mul_q_tr_mut(&self.qr, &self.tau, b)?;
        Ok(())
    }

    /// Solve the overdetermined linear system with the given right hand side
    /// in a least squares sense, see the comments on [ColPivQR::solve_mut].
    pub fn solve<C2: Dim, S>(&self, rhs: Matrix<T, R, C2, S>) -> Result<OMatrix<T, C, C2>, Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous + Storage<T, R, C2>,
        T: Zero,
        DefaultAllocator: Allocator<C, C2> + Allocator<R, C2>,
    {
        let (_, c2) = rhs.shape_generic();
        let (_, c) = self.qr.shape_generic();
        let mut x = OMatrix::zeros_generic(c, c2);
        self.solve_mut(&mut x, rhs)?;
        Ok(x)
    }

    /// Solve the square or overdetermined system in `A X = B`, where `X ∈ R^(n ⨯ k)`,
    /// `B ∈ R^(m ⨯ k)`in a least-squares sense, such that `|| A X -B||^2`
    /// is minimized. The solution is placed into the matrix `X ∈ R^(m ⨯ k)`.
    ///
    /// Note that QR decomposition _does not_ give the minimum norm solution
    /// for `X`, only the residual is minimized which is typically what we want.
    ///
    /// This function performs a small allocation.
    pub fn solve_mut<C2: Dim, S, S2>(
        &self,
        x: &mut Matrix<T, C, C2, S2>,
        b: Matrix<T, R, C2, S>,
    ) -> Result<(), Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous,
        S2: RawStorageMut<T, C, C2> + IsContiguous,
        T: Zero,
    {
        let rank = self.rank();
        qr_util::qr_solve_mut_with_rank_unpermuted(&self.qr, &self.tau, rank, x, b)?;
        self.p().permute_rows_mut(x)?;
        Ok(())
    }
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: Scalar + ComplexField,
    R: DimMin<C>,
    C: Dim,
{
    /// Computes the orthonormal matrix `Q ∈ R^(m ⨯ n)` of this decomposition.
    /// Note that this matrix has _economy_ dimensions, which means it is not
    /// square unless `A` is square. It satisfies `Q^T Q = I`. Note further
    /// that is is typically not necessary to compute `Q` explicitly. Rather,
    /// check if some of the provided multiplication functions can help to
    /// calculate the matrix products `Q B`, `B Q`, `Q^T B`, `B Q^T` more efficiently.
    ///
    /// This function allocates.
    #[inline]
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, <R as DimMin<C>>::Output>,
        T: ColPivQrReal + Zero + ComplexField,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return OMatrix::zeros_generic(nrows, min_nrows_ncols);
        }

        let mut q = self
            .qr
            .generic_view((0, 0), (nrows, min_nrows_ncols))
            .into_owned();

        let mut info = 0;
        let nrows = nrows.value() as i32;

        let lwork = T::xorgqr_work_size(
            nrows,
            min_nrows_ncols.value() as i32,
            self.tau.len() as i32,
            q.as_mut_slice(),
            nrows,
            self.tau.as_slice(),
            &mut info,
        );

        assert_eq!(check_lapack_info(info), Ok(()), "error in lapack backend");

        let mut work = vec![T::zero(); lwork as usize];

        T::xorgqr(
            nrows,
            min_nrows_ncols.value() as i32,
            self.tau.len() as i32,
            q.as_mut_slice(),
            nrows,
            self.tau.as_slice(),
            &mut work,
            lwork,
            &mut info,
        );

        assert_eq!(check_lapack_info(info), Ok(()), "error in lapack backend");

        q
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This function allocates.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let d = nrows.min(ncols);
        let m = self.qr.generic_view((0, 0), (d, d));
        m.upper_triangle()
    }
}

/// Utility trait to add a thin abstraction layer over lapack functionality for
/// column pivoted QR decomposition.
#[allow(missing_docs)]
pub trait ColPivQrScalar: ComplexField + QrScalar + Sealed {
    /// routine for column pivoting QR decomposition using level 3 BLAS,
    /// see <https://www.netlib.org/lapack/lug/node42.html>
    /// or <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/geqp3.html>
    unsafe fn xgeqp3(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    unsafe fn xgeqp3_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
    ) -> Result<i32, LapackErrorCode>;

    unsafe fn xtrtrs(
        uplo: TriangularStructure,
        trans: Transposition,
        diag: DiagonalKind,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    ) -> Result<(), LapackErrorCode>;

    unsafe fn xlapmt(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode>;

    unsafe fn xlapmr(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode>;
}

macro_rules! colpiv_qr_scalar_impl {
    (
        $type:ty,
        xgeqp3=$xgeqp3:path,
        xtrtrs=$xtrtrs:path,
        xlapmt=$xlapmt:path,
        xlapmr=$xlapmr:path $(,)?
    ) => {
        impl Sealed for $type {}
        impl ColPivQrScalar for $type {
            unsafe fn xgeqp3(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
                work: &mut [Self],
                lwork: i32,
            ) -> Result<(), LapackErrorCode> {
                let mut info = 0;
                unsafe { $xgeqp3(m, n, a, lda, jpvt, tau, work, lwork, &mut info) };
                check_lapack_info(info)
            }

            unsafe fn xgeqp3_work_size(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
            ) -> Result<i32, LapackErrorCode> {
                let mut work = [Zero::zero()];
                let lwork = -1 as i32;
                let mut info = 0;
                unsafe { $xgeqp3(m, n, a, lda, jpvt, tau, &mut work, lwork, &mut info) };
                check_lapack_info(info)?;
                Ok(work[0] as i32)
            }

            unsafe fn xtrtrs(
                uplo: TriangularStructure,
                trans: Transposition,
                diag: DiagonalKind,
                n: i32,
                nrhs: i32,
                a: &[Self],
                lda: i32,
                b: &mut [Self],
                ldb: i32,
            ) -> Result<(), LapackErrorCode> {
                let mut info = 0;
                let trans = match trans {
                    Transposition::No => b'N',
                    Transposition::Transpose => b'T',
                };

                unsafe {
                    $xtrtrs(
                        uplo.into_lapack_uplo_character(),
                        trans,
                        diag.into_lapack_diag_character(),
                        n,
                        nrhs,
                        a,
                        lda,
                        b,
                        ldb,
                        &mut info,
                    );
                }

                check_lapack_info(info)
            }

            unsafe fn xlapmt(
                forwrd: bool,
                m: i32,
                n: i32,
                x: &mut [Self],
                ldx: i32,
                k: &mut [i32],
            ) -> Result<(), LapackErrorCode> {
                debug_assert_eq!(k.len(), n as usize);

                let forward: [i32; 1] = [forwrd.then_some(1).unwrap_or(0)];
                unsafe { $xlapmt(forward.as_slice(), m, n, x, ldx, k) }
                Ok(())
            }

            unsafe fn xlapmr(
                forwrd: bool,
                m: i32,
                n: i32,
                x: &mut [Self],
                ldx: i32,
                k: &mut [i32],
            ) -> Result<(), LapackErrorCode> {
                debug_assert_eq!(k.len(), m as usize);

                let forward: [i32; 1] = [forwrd.then_some(1).unwrap_or(0)];
                unsafe { $xlapmr(forward.as_slice(), m, n, x, ldx, k) }
                Ok(())
            }
        }
    };
}

colpiv_qr_scalar_impl!(
    f32,
    xgeqp3 = lapack::sgeqp3,
    xtrtrs = lapack::strtrs,
    xlapmt = lapack::slapmt,
    xlapmr = lapack::slapmr
);

colpiv_qr_scalar_impl!(
    f64,
    xgeqp3 = lapack::dgeqp3,
    xtrtrs = lapack::dtrtrs,
    xlapmt = lapack::dlapmt,
    xlapmr = lapack::dlapmr
);

/// Trait implemented by reals for which Lapack function exist to compute the
/// column-pivoted QR decomposition.
// @note(geo-ant) This mirrors the behavior in the existing QR implementation
// without pivoting. I'm not 100% sure that we can't abstract over real and
// complex behavior in the scalar trait, but I'll keep it like this for now.
#[allow(missing_docs)]
pub trait ColPivQrReal: ColPivQrScalar + QRReal {
    unsafe fn xormqr(
        side: Side,
        trans: Transposition,
        m: i32,
        n: i32,
        k: i32,
        a: &[Self],
        lda: i32,
        tau: &[Self],
        c: &mut [Self],
        ldc: i32,
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    unsafe fn xormqr_work_size(
        side: Side,
        trans: Transposition,
        m: i32,
        n: i32,
        k: i32,
        a: &[Self],
        lda: i32,
        tau: &[Self],
        c: &mut [Self],
        ldc: i32,
    ) -> Result<i32, LapackErrorCode>;
}

macro_rules! colpiv_qr_real_impl {
    (
        $type:ty,
        xormqr = $xormqr:path $(,)?
    ) => {
        impl ColPivQrReal for $type {
            unsafe fn xormqr(
                side: Side,
                trans: Transposition,
                m: i32,
                n: i32,
                k: i32,
                a: &[Self],
                lda: i32,
                tau: &[Self],
                c: &mut [Self],
                ldc: i32,
                work: &mut [Self],
                lwork: i32,
            ) -> Result<(), LapackErrorCode> {
                let mut info = 0;
                let side = side.into_lapack_side_character();

                // this would be different for complex numbers!
                let trans = match trans {
                    Transposition::No => b'N',
                    Transposition::Transpose => b'T',
                };

                unsafe {
                    $xormqr(
                        side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &mut info,
                    );
                }
                check_lapack_info(info)
            }

            unsafe fn xormqr_work_size(
                side: Side,
                trans: Transposition,
                m: i32,
                n: i32,
                k: i32,
                a: &[Self],
                lda: i32,
                tau: &[Self],
                c: &mut [Self],
                ldc: i32,
            ) -> Result<i32, LapackErrorCode> {
                let mut info = 0;
                let side = side.into_lapack_side_character();

                // this would be different for complex numbers!
                let trans = match trans {
                    Transposition::No => b'N',
                    Transposition::Transpose => b'T',
                };

                let mut work = [Zero::zero()];
                let lwork = -1 as i32;
                unsafe {
                    $xormqr(
                        side, trans, m, n, k, a, lda, tau, c, ldc, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info(info)?;
                // for complex numbers: real part
                Ok(ComplexHelper::real_part(work[0]) as i32)
            }
        }
    };
}

colpiv_qr_real_impl!(f32, xormqr = lapack::sormqr);
colpiv_qr_real_impl!(f64, xormqr = lapack::dormqr);
