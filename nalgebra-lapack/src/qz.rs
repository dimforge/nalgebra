#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use simba::scalar::RealField;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Const, Dim};
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar};

use lapack;

/// QZ decomposition of a pair of N*N square matrices.
///
/// Retrieves the left and right matrices of Schur Vectors (VSL and VSR)
/// the upper-quasitriangular matrix `S` and upper triangular matrix `T` such that the
/// decomposed input matrix `a`  equals `VSL * S * VSL.transpose()` and
/// decomposed input matrix `b`  equals `VSL * T * VSL.transpose()`.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(serialize = "DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
         OVector<T, D>: Serialize,
         OMatrix<T, D, D>: Serialize")
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(deserialize = "DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
         OVector<T, D>: Deserialize<'de>,
         OMatrix<T, D, D>: Deserialize<'de>")
    )
)]
#[derive(Clone, Debug)]
pub struct QZ<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    alphar: OVector<T, D>,
    alphai: OVector<T, D>,
    beta: OVector<T, D>,
    vsl: OMatrix<T, D, D>,
    s: OMatrix<T, D, D>,
    vsr: OMatrix<T, D, D>,
    t: OMatrix<T, D, D>,
}

impl<T: Scalar + Copy, D: Dim> Copy for QZ<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
    OMatrix<T, D, D>: Copy,
    OVector<T, D>: Copy,
{
}

impl<T: QZScalar + RealField, D: Dim> QZ<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Attempts to compute the QZ decomposition of input real square matrices `a` and `b`.
    ///
    /// i.e retrieves the left and right matrices of Schur Vectors (VSL and VSR)
    /// the upper-quasitriangular matrix `S` and upper triangular matrix `T` such that the
    /// decomposed matrix `a`  equals `VSL * S * VSL.transpose()` and
    /// decomposed matrix `b`  equals `VSL * T * VSL.transpose()`.
    ///
    /// Panics if the method did not converge.
    pub fn new(a: OMatrix<T, D, D>, b: OMatrix<T, D, D>) -> Self {
        Self::try_new(a, b).expect("QZ decomposition: convergence failed.")
    }

    /// Computes the decomposition of input matrices `a` and `b` into a pair of matrices of Schur vectors
    /// , a quasi-upper triangular matrix and an upper-triangular matrix .
    ///
    /// Returns `None` if the method did not converge.
    pub fn try_new(mut a: OMatrix<T, D, D>, mut b: OMatrix<T, D, D>) -> Option<Self> {
        assert!(
            a.is_square() && b.is_square(),
            "Unable to compute the qz decomposition of non-square matrices."
        );

        assert!(
            a.shape_generic() ==  b.shape_generic(),
            "Unable to compute the qz decomposition of two square matrices of different dimensions."
        );

        let (nrows, ncols) = a.shape_generic();
        let n = nrows.value();

        let mut info = 0;

        let mut alphar = Matrix::zeros_generic(nrows, Const::<1>);
        let mut alphai = Matrix::zeros_generic(nrows, Const::<1>);
        let mut beta = Matrix::zeros_generic(nrows, Const::<1>);
        let mut vsl = Matrix::zeros_generic(nrows, ncols);
        let mut vsr = Matrix::zeros_generic(nrows, ncols);
        // Placeholders:
        let mut bwork = [0i32];
        let mut unused = 0;

        let lwork = T::xgges_work_size(
            b'V',
            b'V',
            b'N',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            b.as_mut_slice(),
            n as i32,
            &mut unused,
            alphar.as_mut_slice(),
            alphai.as_mut_slice(),
            beta.as_mut_slice(),
            vsl.as_mut_slice(),
            n as i32,
            vsr.as_mut_slice(),
            n as i32,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        let mut work = vec![T::zero(); lwork as usize];

        T::xgges(
            b'V',
            b'V',
            b'N',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            b.as_mut_slice(),
            n as i32,
            &mut unused,
            alphar.as_mut_slice(),
            alphai.as_mut_slice(),
            beta.as_mut_slice(),
            vsl.as_mut_slice(),
            n as i32,
            vsr.as_mut_slice(),
            n as i32,
            &mut work,
            lwork,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        Some(QZ {
            alphar,
            alphai,
            beta,
            vsl,
            s: a,
            vsr,
            t: b,
        })
    }

    /// Retrieves the left and right matrices of Schur Vectors (VSL and VSR)
    /// the upper-quasitriangular matrix `S` and upper triangular matrix `T` such that the
    /// decomposed input matrix `a`  equals `VSL * S * VSL.transpose()` and
    /// decomposed input matrix `b`  equals `VSL * T * VSL.transpose()`.
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, D, D>,
        OMatrix<T, D, D>,
        OMatrix<T, D, D>,
        OMatrix<T, D, D>,
    ) {
        (self.vsl, self.s, self.t, self.vsr)
    }

    /// outputs the unprocessed (almost) version of  generalized eigenvalues ((alphar, alpai), beta)
    /// straight from LAPACK
    #[must_use]
    pub fn raw_eigenvalues(&self) -> OVector<(Complex<T>, T), D>
    where
        DefaultAllocator: Allocator<(Complex<T>, T), D>,
    {
        let mut out = Matrix::from_element_generic(
            self.vsl.shape_generic().0,
            Const::<1>,
            (Complex::zero(), T::RealField::zero()),
        );

        for i in 0..out.len() {
            out[i] = (
                Complex::new(self.alphar[i].clone(), self.alphai[i].clone()),
                self.beta[i].clone(),
            )
        }

        out
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the RealField QZ decomposition.
pub trait QZScalar: Scalar {
    #[allow(missing_docs)]
    fn xgges(
        jobvsl: u8,
        jobvsr: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        sdim: &mut i32,
        alphar: &mut [Self],
        alphai: &mut [Self],
        beta: &mut [Self],
        vsl: &mut [Self],
        ldvsl: i32,
        vsr: &mut [Self],
        ldvsr: i32,
        work: &mut [Self],
        lwork: i32,
        bwork: &mut [i32],
        info: &mut i32,
    );

    #[allow(missing_docs)]
    fn xgges_work_size(
        jobvsl: u8,
        jobvsr: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        sdim: &mut i32,
        alphar: &mut [Self],
        alphai: &mut [Self],
        beta: &mut [Self],
        vsl: &mut [Self],
        ldvsl: i32,
        vsr: &mut [Self],
        ldvsr: i32,
        bwork: &mut [i32],
        info: &mut i32,
    ) -> i32;
}

macro_rules! qz_scalar_impl (
    ($N: ty, $xgges: path) => (
        impl QZScalar for $N {
            #[inline]
            fn xgges(jobvsl:  u8,
                     jobvsr:  u8,
                     sort:   u8,
                     // select: ???
                     n:      i32,
                     a:      &mut [$N],
                     lda:    i32,
                     b:      &mut [$N],
                     ldb:    i32,
                     sdim:   &mut i32,
                     alphar: &mut [$N],
                     alphai: &mut [$N],
                     beta  : &mut [$N],
                     vsl:    &mut [$N],
                     ldvsl:  i32,
                     vsr:    &mut [$N],
                     ldvsr:  i32,
                     work:   &mut [$N],
                     lwork:  i32,
                     bwork:  &mut [i32],
                     info:   &mut i32) {
                unsafe { $xgges(jobvsl, jobvsr, sort, None, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info); }
            }


            #[inline]
            fn xgges_work_size(jobvsl:  u8,
                               jobvsr:  u8,
                               sort:   u8,
                               // select: ???
                               n:      i32,
                               a:      &mut [$N],
                               lda:    i32,
                               b:      &mut [$N],
                               ldb:    i32,
                               sdim:   &mut i32,
                               alphar: &mut [$N],
                               alphai: &mut [$N],
                               beta  : &mut [$N],
                               vsl:    &mut [$N],
                               ldvsl:  i32,
                               vsr:    &mut [$N],
                               ldvsr:  i32,
                               bwork:  &mut [i32],
                               info:   &mut i32)
                               -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xgges(jobvsl, jobvsr, sort, None, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, &mut work, lwork, bwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

qz_scalar_impl!(f32, lapack::sgges);
qz_scalar_impl!(f64, lapack::dgges);
