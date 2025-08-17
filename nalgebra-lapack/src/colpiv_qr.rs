use error::{ErrorCode, check_lapack_info};
use na::{ComplexField, Const, Matrix, OVector, Vector};
use nalgebra::{DefaultAllocator, Dim, DimMin, DimMinimum, OMatrix, Scalar, allocator::Allocator};
use num::{ConstOne, Zero};

pub mod error;

use super::qr::{QRReal, QRScalar};

mod test;

pub struct ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: Scalar + ComplexField,
    R: DimMin<C>,
    C: Dim,
{
    qr: OMatrix<T, R, C>,
    tau: OVector<T, DimMinimum<R, C>>,
    jpvt: OVector<i32, C>,
    eps: T::RealField,
}

impl<T, R, C> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    T: ColPivQrScalar + Zero + ComplexField,
    R: DimMin<C>,
    C: Dim,
{
    //@todo(geo-ant) maybe add another constructor that allows giving a workspace array,
    // so we don't have to allocate in here
    pub fn new(mut m: OMatrix<T, R, C>, eps: T::RealField) -> Option<Self> {
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
        )
        .ok()?;

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
        )
        .ok()?;

        Some(Self {
            qr: m,
            tau,
            jpvt,
            eps,
        })
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
