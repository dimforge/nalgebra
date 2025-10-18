use crate::{ComplexHelper, DiagonalKind, Side, Transposition, TriangularStructure, qr_util};
use crate::{LapackErrorCode, lapack_error::check_lapack_info};
use lapack;
use na::allocator::Allocator;
use na::dimension::{Const, Dim, DimMin, DimMinimum};
use na::{
    ComplexField, DefaultAllocator, IsContiguous, Matrix, OMatrix, OVector, RawStorageMut,
    RealField, Scalar, Storage,
};
use num::Zero;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

pub use crate::qr_util::Error;

/// The QR decomposition of a general matrix `A`, where `A = Q R`.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    T: Scalar,
    R: DimMin<C>,
    C: Dim,
{
    qr: OMatrix<T, R, C>,
    tau: OVector<T, DimMinimum<R, C>>,
}

impl<T: Scalar + Copy, R: DimMin<C>, C: Dim> Copy for QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T, R, C> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    T: Scalar,
    R: DimMin<C>,
    C: Dim,
{
    /// Computes the QR decomposition of the matrix `m`.
    pub fn new(mut m: OMatrix<T, R, C>) -> Result<Self, Error>
    where
        T: QrScalar + Zero,
    {
        let (nrows, ncols) = m.shape_generic();

        let mut tau = Matrix::zeros_generic(nrows.min(ncols), Const::<1>);

        if nrows.value() == 0 || ncols.value() == 0 {
            return Ok(Self { qr: m, tau });
        }

        let lwork = unsafe {
            T::xgeqrf_work_size(
                nrows.value() as i32,
                ncols.value() as i32,
                m.as_mut_slice(),
                nrows.value() as i32,
                tau.as_mut_slice(),
            )?
        };

        let mut work = vec![T::zero(); lwork as usize];

        unsafe {
            T::xgeqrf(
                nrows.value() as i32,
                ncols.value() as i32,
                m.as_mut_slice(),
                nrows.value() as i32,
                tau.as_mut_slice(),
                &mut work,
                lwork,
            )?;
        }

        Ok(Self { qr: m, tau })
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
        T: Zero,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle()
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
}

impl<T, R, C> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    T: Scalar + QrReal + RealField,
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
    /// in a least squares sense, see the comments on [QR::solve_mut].
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
    /// Note that QR decomposition _does not_ typically give the minimum norm solution
    /// for `X`, only the residual is minimized which is typically what we want.
    ///
    /// The system must be full rank. For system that are not full rank or very
    /// ill-conditioned, use the column-pivoted QR decomposition.
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
        // since we use QR decomposition without column pivoting, we assume
        // full rank.
        let rank = self
            .nrows()
            .min(self.ncols())
            .try_into()
            .expect("integer dimensions out of bounds");
        qr_util::qr_solve_mut_with_rank_unpermuted(&self.qr, &self.tau, rank, x, b)?;
        Ok(())
    }
}

impl<T: QrReal + Zero, R: DimMin<C>, C: Dim> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    // where
    //     DefaultAllocator:
    //         Allocator<R, C> + + Allocator<DimMinimum<R, C>>,
{
    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[inline]
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return OMatrix::from_element_generic(nrows, min_nrows_ncols, T::zero());
        }

        let mut q = self
            .qr
            .generic_view((0, 0), (nrows, min_nrows_ncols))
            .into_owned();

        let nrows = nrows.value() as i32;

        let lwork = unsafe {
            T::xorgqr_work_size(
                nrows,
                min_nrows_ncols.value() as i32,
                self.tau.len() as i32,
                q.as_mut_slice(),
                nrows,
                self.tau.as_slice(),
            )
        }
        .expect("unexpected error in lapack backend");

        let mut work = vec![T::zero(); lwork as usize];

        unsafe {
            T::xorgqr(
                nrows,
                min_nrows_ncols.value() as i32,
                self.tau.len() as i32,
                q.as_mut_slice(),
                nrows,
                self.tau.as_slice(),
                &mut work,
                lwork,
            )
        }
        .expect("unexpected error in lapack backend");

        q
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalar types for which Lapack function exist to compute the
/// QR decomposition.
#[allow(missing_docs)]
pub trait QrScalar: ComplexField + Scalar + Copy {
    unsafe fn xgeqrf(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    unsafe fn xgeqrf_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
    ) -> Result<i32, LapackErrorCode>;

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

macro_rules! qr_scalar_impl(
    ($type:ty,
        xgeqrf = $xgeqrf: path,
        xgeqp3=$xgeqp3:path,
        xtrtrs=$xtrtrs:path,
        xlapmt=$xlapmt:path,
        xlapmr=$xlapmr:path $(,)?) => (
        impl QrScalar for $type {
            #[inline]
            unsafe fn xgeqrf(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self],
                      work: &mut [Self], lwork: i32) -> Result<(),LapackErrorCode> {
                let mut info = 0;
                unsafe { $xgeqrf(m, n, a, lda, tau, work, lwork, &mut info) }
                check_lapack_info(info)
            }

            #[inline]
            unsafe fn xgeqrf_work_size(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self]) -> Result<i32, LapackErrorCode> {
                let mut info = 0;
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgeqrf(m, n, a, lda, tau, &mut work, lwork, &mut info); }
                check_lapack_info(info)?;
                Ok(ComplexHelper::real_part(work[0]) as i32)
            }

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
    )
);

/// Trait implemented by reals for which Lapack function exist to compute the
/// QR decomposition.
pub trait QrReal: QrScalar {
    #[allow(missing_docs)]
    unsafe fn xorgqr(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    #[allow(missing_docs)]
    unsafe fn xorgqr_work_size(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
    ) -> Result<i32, LapackErrorCode>;

    #[allow(missing_docs)]
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

    #[allow(missing_docs)]
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

macro_rules! qr_real_impl(
    ($type:ty, xorgqr = $xorgqr:path, xormqr = $xormqr:path) => (
        impl QrReal for $type {
            #[inline]
            unsafe fn xorgqr(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self],
                      work: &mut [Self], lwork: i32) -> Result<(),LapackErrorCode> {
                let mut info = 0;
                unsafe { $xorgqr(m, n, k, a, lda, tau, work, lwork, &mut info) }
                check_lapack_info(info)
            }

            #[inline]
            unsafe fn xorgqr_work_size(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self]) -> Result<i32,LapackErrorCode> {
                let mut info = 0;
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xorgqr(m, n, k, a, lda, tau, &mut work, lwork, &mut info); }
                check_lapack_info(info)?;
                Ok(ComplexHelper::real_part(work[0]) as i32)
            }

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
    )
);

qr_scalar_impl!(
    f32,
    xgeqrf = lapack::sgeqrf,
    xgeqp3 = lapack::sgeqp3,
    xtrtrs = lapack::strtrs,
    xlapmt = lapack::slapmt,
    xlapmr = lapack::slapmr
);
qr_scalar_impl!(
    f64,
    xgeqrf = lapack::dgeqrf,
    xgeqp3 = lapack::dgeqp3,
    xtrtrs = lapack::dtrtrs,
    xlapmt = lapack::dlapmt,
    xlapmr = lapack::dlapmr
);
// @todo(geo-ant) maybe re-enable at a later date,
// but for now: if we ain't testin' it, we ain't implementin' it
// qr_scalar_impl!(Complex<f32>, lapack::cgeqrf);
// qr_scalar_impl!(Complex<f64>, lapack::zgeqrf);

qr_real_impl!(f32, xorgqr = lapack::sorgqr, xormqr = lapack::sormqr);
qr_real_impl!(f64, xorgqr = lapack::dorgqr, xormqr = lapack::dormqr);
