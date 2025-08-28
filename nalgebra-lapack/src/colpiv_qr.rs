use error::{LapackErrorCode, check_lapack_info};
use na::{ComplexField, Const, IsContiguous, Matrix, OVector, RealField, Storage, Vector};
use nalgebra::storage::RawStorageMut;
use nalgebra::{DefaultAllocator, Dim, DimMin, DimMinimum, OMatrix, Scalar, allocator::Allocator};
use num::float::TotalOrder;
use num::{Float, Zero};
use rank::{RankEstimationAlgo, calculate_rank};

pub mod error;
use crate::ComplexHelper;

use super::qr::{QRReal, QRScalar};
#[cfg(test)]
mod test;
mod utility;

mod permutation;
pub use permutation::Permutation;

/// utility functionality to calculate the rank of matrices
mod rank;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Error in lapack backend (code: {0})")]
    Backend(#[from] LapackErrorCode),
    #[error("Wrong matrix dimensions")]
    Dimension,
    #[error("Solving underdetermined systems not supported")]
    Underdetermined,
    #[error("Matrix has rank zero")]
    ZeroRank,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// Indicates the side from which a matrix multiplication is to be performed.
pub enum Side {
    /// perform multiplication from the left
    Left,
    /// perform multiplication from the right
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// Indicates whether or not to transpose a matrix during a matrix
/// operation.
// @note(geo-ant) once we add complex, we can refactor this
// to conjugate transpose (or hermitian transpose).
pub enum Transposition {
    /// don't transpose, i.e. leave the matrix as is
    No,
    /// transpose the matrix.
    Transpose,
}

/// describes the type of a triangular matrix
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TriangularStructure {
    /// upper triangular
    Upper,
    /// lower triangular
    Lower,
}

/// property of the diagonal of a triangular matrix
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DiagonalKind {
    /// diagonal entries all have value of 1
    Unit,
    /// diagonal elements are arbitrary
    NonUnit,
}

/// todo
pub struct ColPivQr<T, R, C>
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

impl<T, R, C> ColPivQr<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrScalar + Zero + RealField + TotalOrder + Float,
    R: DimMin<C>,
    C: Dim,
{
    ///@todo(geo-ant) maybe add another constructor that allows giving a workspace array,
    // so we don't have to allocate in here
    pub fn new(mut m: OMatrix<T, R, C>, rank_algo: RankEstimationAlgo<T>) -> Result<Self, Error> {
        let (nrows, ncols) = m.shape_generic();
        let mut tau: OVector<T, DimMinimum<R, C>> =
            Vector::zeros_generic(nrows.min(ncols), Const::<1>);
        let mut jpvt: OVector<i32, C> = Vector::zeros_generic(ncols, Const::<1>);

        let lwork = T::xgeqp3_work_size(
            nrows.value().try_into().expect("matrix dims out of bounds"),
            ncols.value().try_into().expect("matrix dims out of bounds"),
            m.as_mut_slice(),
            nrows.value().try_into().expect("matrix dims out of bounds"),
            jpvt.as_mut_slice(),
            tau.as_mut_slice(),
        )?;

        let mut work = vec![T::zero(); lwork as usize];

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

        let rank: i32 = calculate_rank(&m, rank_algo)
            .try_into()
            .map_err(|_| Error::Dimension)?;

        Ok(Self {
            qr: m,
            rank,
            tau,
            jpvt,
        })
    }
}

impl<T, R, C> ColPivQr<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrScalar + Zero + RealField,
    R: DimMin<C>,
    C: Dim,
{
    /// get the effective rank of the matrix
    #[inline]
    pub fn rank(&self) -> u16 {
        self.rank as u16
    }

    #[inline]
    /// the number of rows of the original matrix
    pub fn nrows(&self) -> usize {
        self.qr.nrows()
    }

    #[inline]
    /// the number of columns of the original matrix
    pub fn ncols(&self) -> usize {
        self.qr.ncols()
    }

    /// obtain the permutation matrix $\boldsymbol{P}$,
    /// such that $\boldsymbol{A P}= \boldsymbol{Q R}$. This function
    /// allocates a copy of the stored permutation vector.
    pub fn p(&self) -> Permutation<C> {
        Permutation::new(self.jpvt.clone())
    }
}

impl<T, R, C> ColPivQr<T, R, C>
where
    DefaultAllocator: Allocator<R, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<DimMinimum<R, C>>
        + Allocator<C>,
    T: ColPivQrReal + Zero + RealField,
    R: DimMin<C>,
    C: Dim,
{
    ///
    //@todo(geo-ant): this is Q*B
    pub fn q_mul_mut<C2, S2>(&self, b: &mut Matrix<T, C, C2, S2>) -> Result<(), Error>
    where
        C2: Dim,
        S2: RawStorageMut<T, C, C2> + IsContiguous,
    {
        assert_eq!(b.nrows(), self.ncols());
        // SAFETY: matrix has the correct dimensions for operation Q*B
        unsafe { self.multiply_q_mut(b, Side::Left, Transposition::No) }
    }

    ///
    //@todo(geo-ant): this is Q^T*B
    pub fn q_tr_mul_mut<C2, S2>(&self, b: &mut Matrix<T, R, C2, S2>) -> Result<(), Error>
    where
        C2: Dim,
        S2: RawStorageMut<T, R, C2> + IsContiguous,
    {
        assert_eq!(b.nrows(), self.nrows());
        // SAFETY: matrix has the correct dimensions for operation Q*B
        unsafe { self.multiply_q_mut(b, Side::Left, Transposition::Transpose) }
    }

    ///
    //@todo(geo-ant): this is Q*B
    pub fn mul_q_mut<R2, S2>(&self, b: &mut Matrix<T, R2, R, S2>) -> Result<(), Error>
    where
        R2: Dim,
        S2: RawStorageMut<T, R2, R> + IsContiguous,
    {
        assert_eq!(b.ncols(), self.nrows());
        // SAFETY: matrix has the correct dimensions for operation B*Q
        unsafe { self.multiply_q_mut(b, Side::Right, Transposition::No) }
    }

    ///
    //@todo(geo-ant): this is B*Q^T
    pub fn mul_q_tr_mut<R2, S2>(&self, b: &mut Matrix<T, R2, C, S2>) -> Result<(), Error>
    where
        R2: Dim,
        S2: RawStorageMut<T, R2, C> + IsContiguous,
    {
        assert_eq!(b.ncols(), self.nrows());
        // SAFETY: matrix has the correct dimensions for operation Q*B
        unsafe { self.multiply_q_mut(b, Side::Right, Transposition::Transpose) }
    }

    /// Multiplies the provided matrix by Q, requiring contiguous column-major storage
    //@todo(geo-ant) comment
    unsafe fn multiply_q_mut<R2, C2, S2>(
        &self,
        mat: &mut Matrix<T, R2, C2, S2>,
        side: Side,
        transpose: Transposition,
    ) -> Result<(), Error>
    where
        S2: RawStorageMut<T, R2, C2> + IsContiguous,
        R2: Dim,
        C2: Dim,
    {
        //@todo test matrix dimensions
        let a = self.qr.as_slice();
        let lda = self
            .qr
            .nrows()
            .try_into()
            .expect("integer dimension out of range");
        let m = mat
            .nrows()
            .try_into()
            .expect("integer dimension out of range");
        let n = mat
            .ncols()
            .try_into()
            .expect("integer dimension out of range");
        let k = self
            .tau
            .len()
            .try_into()
            .expect("integer dimension out of range");
        let ldc = mat
            .nrows()
            .try_into()
            .expect("integer dimension out of range");
        let c = mat.as_mut_slice();
        let trans = transpose;
        let tau = self.tau.as_slice();

        let lwork = T::xormqr_work_size(side, transpose, m, n, k, a, lda, tau, c, ldc).unwrap();
        let mut work = vec![T::zero(); lwork as usize];
        T::xormqr(side, trans, m, n, k, a, lda, tau, c, ldc, &mut work, lwork).unwrap();
        Ok(())
    }

    ///
    //@todo
    pub fn solve<C2: Dim, S>(&self, rhs: &Matrix<T, R, C2, S>) -> Result<OMatrix<T, C, C2>, Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous + Storage<T, R, C2>,
        T: Zero,
        DefaultAllocator: Allocator<C, C2> + Allocator<R, C2>,
    {
        let (_, c2) = rhs.shape_generic();
        let (_, c) = self.qr.shape_generic();
        let mut rhs = rhs.clone_owned();
        let mut x = OMatrix::zeros_generic(c, c2);
        self.solve_mut(&mut rhs, &mut x)?;
        Ok(x)
    }

    ///
    //@todo(geo-ant) document!
    pub fn solve_mut<C2: Dim, S, S2>(
        &self,
        rhs: &mut Matrix<T, R, C2, S>,
        x: &mut Matrix<T, C, C2, S2>,
    ) -> Result<(), Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous,
        S2: RawStorageMut<T, C, C2> + IsContiguous,
        T: Zero,
    {
        //@todo(geo-ant) validate matrix dimensions!! Must be overdetermined (or square) here
        if rhs.nrows() != self.nrows() {
            return Err(Error::Dimension);
        }

        if self.nrows() < self.ncols() || self.nrows() == 0 || self.ncols() == 0 {
            return Err(Error::Dimension);
        }

        if x.ncols() != rhs.ncols() || x.nrows() != self.ncols() {
            return Err(Error::Dimension);
        }

        self.q_tr_mul_mut(rhs)?;

        let rank = self.rank();

        if rank == 0 {
            return Err(Error::ZeroRank);
        }

        if (rank as usize) < self.ncols() {
            x.view_mut((rank as usize, 0), (x.nrows() - rank as usize, x.ncols()))
                .iter_mut()
                .for_each(|val| val.set_zero());
        }

        let x_cols = x.ncols();
        x.view_mut((0, 0), (rank as usize, x_cols))
            .copy_from(&rhs.view((0, 0), (rank as usize, x_cols)));

        let ldb: i32 = x
            .nrows()
            .try_into()
            .expect("integer dimensions out of bounds");

        T::xtrtrs(
            TriangularStructure::Upper,
            Transposition::No,
            DiagonalKind::NonUnit,
            rank.try_into().expect("rank out of bounds"),
            x.ncols()
                .try_into()
                .expect("integer dimensions out of bounds"),
            self.qr.as_slice(),
            self.qr
                .nrows()
                .try_into()
                .expect("integer dimensions out of bounds"),
            x.as_mut_slice(),
            ldb,
        )
            //@todo
        .unwrap();

        self.p().permute_rows_mut(x)?;
        Ok(())
    }
}

impl<T, R, C> ColPivQr<T, R, C>
where
    DefaultAllocator: Allocator<R, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<DimMinimum<R, C>>
        + Allocator<C>,
    T: ColPivQrReal + Zero + ComplexField,
    R: DimMin<C>,
    C: Dim,
    DefaultAllocator:,
{
    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[inline]
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.qr.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return OMatrix::from_element_generic(nrows, min_nrows_ncols, T::zero());
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

        debug_assert!(check_lapack_info(info).is_ok(), "error in lapack backend");

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

        debug_assert!(check_lapack_info(info).is_ok(), "error in lapack backend");

        q
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
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

pub trait ColPivQrScalar: ComplexField + QRScalar {
    /// routine for column pivoting QR decomposition using level 3 BLAS,
    /// see https://www.netlib.org/lapack/lug/node42.html
    /// or https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/geqp3.html
    fn xgeqp3(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    fn xgeqp3_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
    ) -> Result<i32, LapackErrorCode>;

    fn xtrtrs(
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

    fn xlapmt(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode>;

    fn xlapmr(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode>;
}

impl ColPivQrScalar for f32 {
    fn xgeqp3(
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
        unsafe { lapack::sgeqp3(m, n, a, lda, jpvt, tau, work, lwork, &mut info) };
        check_lapack_info(info)
    }

    fn xgeqp3_work_size(
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
        unsafe { lapack::sgeqp3(m, n, a, lda, jpvt, tau, &mut work, lwork, &mut info) };
        check_lapack_info(info)?;
        Ok(work[0] as i32)
    }

    fn xtrtrs(
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
            lapack::strtrs(
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

    fn xlapmt(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode> {
        debug_assert_eq!(k.len(), n as usize);

        let forward: [i32; 1] = [forwrd.then_some(1).unwrap_or(0)];
        unsafe { lapack::slapmt(forward.as_slice(), m, n, x, ldx, k) }
        Ok(())
    }

    fn xlapmr(
        forwrd: bool,
        m: i32,
        n: i32,
        x: &mut [Self],
        ldx: i32,
        k: &mut [i32],
    ) -> Result<(), LapackErrorCode> {
        debug_assert_eq!(k.len(), m as usize);

        let forward: [i32; 1] = [forwrd.then_some(1).unwrap_or(0)];
        unsafe { lapack::slapmr(forward.as_slice(), m, n, x, ldx, k) }
        Ok(())
    }
}

/// Trait implemented by reals for which Lapack function exist to compute the
/// column-pivoted QR decomposition.
// @note(geo-ant) This mirrors the behavior in the existing QR implementation
// without pivoting. I'm not 100% sure that we can't abstract over real and
// complex behavior in the scalar trait, but I'll keep it like this for now.
pub trait ColPivQrReal: ColPivQrScalar + QRReal {
    fn xormqr(
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

    fn xormqr_work_size(
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

impl ColPivQrReal for f32 {
    fn xormqr(
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
            lapack::sormqr(
                side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &mut info,
            );
        }
        check_lapack_info(info)
    }

    fn xormqr_work_size(
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
            lapack::sormqr(
                side, trans, m, n, k, a, lda, tau, c, ldc, &mut work, lwork, &mut info,
            );
        }
        check_lapack_info(info)?;
        // for complex numbers: real part
        Ok(ComplexHelper::real_part(work[0]) as i32)
    }
}
