use crate::ComplexHelper;
use crate::{LapackErrorCode, lapack_error::check_lapack_info};
use lapack;
use na::allocator::Allocator;
use na::dimension::{Const, Dim, DimMin, DimMinimum};
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar};
use num::Zero;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

pub use crate::qr_util::Error;

/// The QR decomposition of a general matrix.
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
pub struct QR<T: Scalar, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
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

impl<T: QrScalar + Zero, R: DimMin<C>, C: Dim> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<DimMinimum<R, C>>,
{
    /// Computes the QR decomposition of the matrix `m`.
    pub fn new(mut m: OMatrix<T, R, C>) -> Result<Self, Error> {
        let (nrows, ncols) = m.shape_generic();

        let mut tau = Matrix::zeros_generic(nrows.min(ncols), Const::<1>);

        if nrows.value() == 0 || ncols.value() == 0 {
            return Ok(Self { qr: m, tau });
        }

        let lwork = T::xgeqrf_work_size(
            nrows.value() as i32,
            ncols.value() as i32,
            m.as_mut_slice(),
            nrows.value() as i32,
            tau.as_mut_slice(),
        )?;

        let mut work = vec![T::zero(); lwork as usize];

        T::xgeqrf(
            nrows.value() as i32,
            ncols.value() as i32,
            m.as_mut_slice(),
            nrows.value() as i32,
            tau.as_mut_slice(),
            &mut work,
            lwork,
        )?;

        Ok(Self { qr: m, tau })
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C> {
        let (nrows, ncols) = self.qr.shape_generic();
        self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }
}

impl<T: QrReal + Zero, R: DimMin<C>, C: Dim> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<DimMinimum<R, C>>,
{
    /// Retrieves the matrices `(Q, R)` of this decompositions.
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    ) {
        (self.q(), self.r())
    }

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

        let nrows = nrows.value() as i32;

        let lwork = T::xorgqr_work_size(
            nrows,
            min_nrows_ncols.value() as i32,
            self.tau.len() as i32,
            q.as_mut_slice(),
            nrows,
            self.tau.as_slice(),
        )
        .expect("unexpected error in lapack backend");

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
        )
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
pub trait QrScalar: Scalar + Copy {
    fn xgeqrf(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
    ) -> Result<(), LapackErrorCode>;

    fn xgeqrf_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
    ) -> Result<i32, LapackErrorCode>;
}

/// Trait implemented by reals for which Lapack function exist to compute the
/// QR decomposition.
pub trait QrReal: QrScalar {
    #[allow(missing_docs)]
    fn xorgqr(
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
    fn xorgqr_work_size(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
    ) -> Result<i32, LapackErrorCode>;
}

macro_rules! qr_scalar_impl(
    ($N: ty, $xgeqrf: path) => (
        impl QrScalar for $N {
            #[inline]
            fn xgeqrf(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self],
                      work: &mut [Self], lwork: i32) -> Result<(),LapackErrorCode> {
                let mut info = 0;
                unsafe { $xgeqrf(m, n, a, lda, tau, work, lwork, &mut info) }
                check_lapack_info(info)
            }

            #[inline]
            fn xgeqrf_work_size(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self]) -> Result<i32, LapackErrorCode> {
                let mut info = 0;
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgeqrf(m, n, a, lda, tau, &mut work, lwork, &mut info); }
                check_lapack_info(info)?;
                Ok(ComplexHelper::real_part(work[0]) as i32)
            }
        }
    )
);

macro_rules! qr_real_impl(
    ($N: ty, $xorgqr: path) => (
        impl QrReal for $N {
            #[inline]
            fn xorgqr(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self],
                      work: &mut [Self], lwork: i32) -> Result<(),LapackErrorCode> {
                let mut info = 0;
                unsafe { $xorgqr(m, n, k, a, lda, tau, work, lwork, &mut info) }
                check_lapack_info(info)
            }

            #[inline]
            fn xorgqr_work_size(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self]) -> Result<i32,LapackErrorCode> {
                let mut info = 0;
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xorgqr(m, n, k, a, lda, tau, &mut work, lwork, &mut info); }
                check_lapack_info(info)?;
                Ok(ComplexHelper::real_part(work[0]) as i32)
            }
        }
    )
);

qr_scalar_impl!(f32, lapack::sgeqrf);
qr_scalar_impl!(f64, lapack::dgeqrf);
// @todo(geo-ant) maybe re-enable at a later date,
// but for now: if we ain't testin' it, we ain't implementin' it
// qr_scalar_impl!(Complex<f32>, lapack::cgeqrf);
// qr_scalar_impl!(Complex<f64>, lapack::zgeqrf);

qr_real_impl!(f32, lapack::sorgqr);
qr_real_impl!(f64, lapack::dorgqr);
