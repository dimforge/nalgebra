use error::{ErrorCode, check_lapack_info};
use na::{Const, Matrix};
use nalgebra::{DefaultAllocator, Dim, DimMin, DimMinimum, OMatrix, Scalar, allocator::Allocator};
use num::Zero;

pub mod error;

use super::qr::{QRReal, QRScalar};

pub struct ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    T: Scalar,
    R: DimMin<C>,
    C: Dim,
{
    qr: OMatrix<T, R, C>,
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrScalar,
    R: DimMin<C>,
    C: Dim,
{
    pub fn new(mut m: OMatrix<T, R, C>) -> Option<Self> {
        let (nrows, ncols) = m.shape_generic();
        let mut info = 0;
        let mut tau = Matrix::zeros_generic(nrows.min(ncols), Const::<1>);
        let mut jpvt= Matrix::zeros_generic(ncols, Const::<1>);

        let lwork = T::xgeqp3_work_size(nrows.value(), ncols.value(), m.as_mut_slice(), nrows.value(), jpvt, tau)

        
        todo!()
    }
}

pub trait ColPivQrScalar: QRScalar {
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
    ) -> Result<(), ErrorCode>;

    fn xgeqp3_work_size(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
    ) -> Result<i32, ErrorCode>;
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
    ) -> Result<(), ErrorCode> {
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
    ) -> Result<i32, ErrorCode> {
        let mut work = [Zero::zero()];
        let lwork = -1 as i32;
        let mut info = 0;
        unsafe { lapack::sgeqp3(m, n, a, lda, jpvt, tau, &mut work, lwork, &mut info) };
        check_lapack_info(info)?;
        Ok(work[0] as i32)
    }
}
