#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use na::allocator::Allocator;
use na::dimension::{Dim, DimMin, DimMinimum, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixMN, Scalar, VectorN};
use ComplexHelper;

use lapack;

/// The QR decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct QR<N: Scalar, R: DimMin<C>, C: Dim>
where DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>
{
    qr: MatrixMN<N, R, C>,
    tau: VectorN<N, DimMinimum<R, C>>,
}

impl<N: Scalar, R: DimMin<C>, C: Dim> Copy for QR<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
{}

impl<N: QRScalar + Zero, R: DimMin<C>, C: Dim> QR<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, DimMinimum<R, C>>
{
    /// Computes the QR decomposition of the matrix `m`.
    pub fn new(mut m: MatrixMN<N, R, C>) -> QR<N, R, C> {
        let (nrows, ncols) = m.data.shape();

        let mut info = 0;
        let mut tau = unsafe { Matrix::new_uninitialized_generic(nrows.min(ncols), U1) };

        if nrows.value() == 0 || ncols.value() == 0 {
            return QR { qr: m, tau: tau };
        }

        let lwork = N::xgeqrf_work_size(
            nrows.value() as i32,
            ncols.value() as i32,
            m.as_mut_slice(),
            nrows.value() as i32,
            tau.as_mut_slice(),
            &mut info,
        );

        let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

        N::xgeqrf(
            nrows.value() as i32,
            ncols.value() as i32,
            m.as_mut_slice(),
            nrows.value() as i32,
            tau.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );

        QR { qr: m, tau: tau }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    pub fn r(&self) -> MatrixMN<N, DimMinimum<R, C>, C> {
        let (nrows, ncols) = self.qr.data.shape();
        self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }
}

impl<N: QRReal + Zero, R: DimMin<C>, C: Dim> QR<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, DimMinimum<R, C>>
{
    /// Retrieves the matrices `(Q, R)` of this decompositions.
    pub fn unpack(
        self,
    ) -> (
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
    ) {
        (self.q(), self.r())
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[inline]
    pub fn q(&self) -> MatrixMN<N, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.qr.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return MatrixMN::from_element_generic(nrows, min_nrows_ncols, N::zero());
        }

        let mut q = self
            .qr
            .generic_slice((0, 0), (nrows, min_nrows_ncols))
            .into_owned();

        let mut info = 0;
        let nrows = nrows.value() as i32;

        let lwork = N::xorgqr_work_size(
            nrows,
            min_nrows_ncols.value() as i32,
            self.tau.len() as i32,
            q.as_mut_slice(),
            nrows,
            self.tau.as_slice(),
            &mut info,
        );

        let mut work = vec![N::zero(); lwork as usize];

        N::xorgqr(
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
pub trait QRScalar: Scalar {
    fn xgeqrf(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );

    fn xgeqrf_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        info: &mut i32,
    ) -> i32;
}

/// Trait implemented by reals for which Lapack function exist to compute the
/// QR decomposition.
pub trait QRReal: QRScalar {
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
        info: &mut i32,
    );

    #[allow(missing_docs)]
    fn xorgqr_work_size(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        info: &mut i32,
    ) -> i32;
}

macro_rules! qr_scalar_impl(
    ($N: ty, $xgeqrf: path) => (
        impl QRScalar for $N {
            #[inline]
            fn xgeqrf(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self],
                      work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xgeqrf(m, n, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xgeqrf_work_size(m: i32, n: i32, a: &mut [Self], lda: i32, tau: &mut [Self],
                                info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgeqrf(m, n, a, lda, tau, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

macro_rules! qr_real_impl(
    ($N: ty, $xorgqr: path) => (
        impl QRReal for $N {
            #[inline]
            fn xorgqr(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self],
                      work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xorgqr(m, n, k, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xorgqr_work_size(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &[Self],
                                info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xorgqr(m, n, k, a, lda, tau, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

qr_scalar_impl!(f32, lapack::sgeqrf);
qr_scalar_impl!(f64, lapack::dgeqrf);
qr_scalar_impl!(Complex<f32>, lapack::cgeqrf);
qr_scalar_impl!(Complex<f64>, lapack::zgeqrf);

qr_real_impl!(f32, lapack::sorgqr);
qr_real_impl!(f64, lapack::dorgqr);
