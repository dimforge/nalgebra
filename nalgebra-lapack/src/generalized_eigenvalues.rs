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

/// Generalized eigenvalues and generalized eigenvectors(left and right) of a pair of N*N square matrices.
///
/// Each generalized eigenvalue (lambda) satisfies determinant(A - lambda*B) = 0
///
/// The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
/// of (A,B) satisfies
///
/// A * v(j) = lambda(j) * B * v(j).
///
/// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
/// of (A,B) satisfies
///
/// u(j)**H * A  = lambda(j) * u(j)**H * B .
/// where u(j)**H is the conjugate-transpose of u(j).
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
pub struct GE<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    alphar: OVector<T, D>,
    alphai: OVector<T, D>,
    beta: OVector<T, D>,
    vsl: OMatrix<T, D, D>,
    vsr: OMatrix<T, D, D>,
}

impl<T: Scalar + Copy, D: Dim> Copy for GE<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
    OMatrix<T, D, D>: Copy,
    OVector<T, D>: Copy,
{
}

impl<T: GEScalar + RealField + Copy, D: Dim> GE<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Attempts to compute the generalized eigenvalues, and left and right associated eigenvectors
    /// via the raw returns from LAPACK's dggev and sggev routines
    ///
    /// Each generalized eigenvalue (lambda) satisfies determinant(A - lambda*B) = 0
    ///
    /// The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
    /// of (A,B) satisfies
    ///
    /// A * v(j) = lambda(j) * B * v(j).
    ///
    /// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
    /// of (A,B) satisfies
    ///
    /// u(j)**H * A  = lambda(j) * u(j)**H * B .
    /// where u(j)**H is the conjugate-transpose of u(j).
    ///
    /// Panics if the method did not converge.
    pub fn new(a: OMatrix<T, D, D>, b: OMatrix<T, D, D>) -> Self {
        Self::try_new(a, b).expect("Calculation of generalized eigenvalues failed.")
    }

    /// Attempts to compute the generalized eigenvalues (and eigenvectors) via the raw returns from LAPACK's
    /// dggev and sggev routines
    ///
    ///  Each generalized eigenvalue (lambda) satisfies determinant(A - lambda*B) = 0
    ///
    ///  The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
    ///  of (A,B) satisfies
    ///
    ///  A * v(j) = lambda(j) * B * v(j).
    ///
    ///  The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
    ///  of (A,B) satisfies
    ///
    ///  u(j)**H * A  = lambda(j) * u(j)**H * B .
    ///  where u(j)**H is the conjugate-transpose of u(j).
    ///
    /// Returns `None` if the method did not converge.
    pub fn try_new(mut a: OMatrix<T, D, D>, mut b: OMatrix<T, D, D>) -> Option<Self> {
        assert!(
            a.is_square() && b.is_square(),
            "Unable to compute the generalized eigenvalues of non-square matrices."
        );

        assert!(
            a.shape_generic() ==  b.shape_generic(),
            "Unable to compute the generalized eigenvalues of two square matrices of different dimensions."
        );

        let (nrows, ncols) = a.shape_generic();
        let n = nrows.value();

        let mut info = 0;

        let mut alphar = Matrix::zeros_generic(nrows, Const::<1>);
        let mut alphai = Matrix::zeros_generic(nrows, Const::<1>);
        let mut beta = Matrix::zeros_generic(nrows, Const::<1>);
        let mut vsl = Matrix::zeros_generic(nrows, ncols);
        let mut vsr = Matrix::zeros_generic(nrows, ncols);

        let lwork = T::xggev_work_size(
            b'V',
            b'V',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            b.as_mut_slice(),
            n as i32,
            alphar.as_mut_slice(),
            alphai.as_mut_slice(),
            beta.as_mut_slice(),
            vsl.as_mut_slice(),
            n as i32,
            vsr.as_mut_slice(),
            n as i32,
            &mut info,
        );
        lapack_check!(info);

        let mut work = vec![T::zero(); lwork as usize];

        T::xggev(
            b'V',
            b'V',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            b.as_mut_slice(),
            n as i32,
            alphar.as_mut_slice(),
            alphai.as_mut_slice(),
            beta.as_mut_slice(),
            vsl.as_mut_slice(),
            n as i32,
            vsr.as_mut_slice(),
            n as i32,
            &mut work,
            lwork,
            &mut info,
        );
        lapack_check!(info);

        Some(GE {
            alphar,
            alphai,
            beta,
            vsl,
            vsr,
        })
    }

    /// Calculates the generalized eigenvectors (left and right) associated with the generalized eigenvalues
    /// Outputs two matrices, the first one containing the left eigenvectors of the generalized eigenvalues
    /// as columns and the second matrix contains the right eigenvectors of the generalized eigenvalues
    /// as columns
    ///
    ///  The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
    ///  of (A,B) satisfies
    ///
    ///  A * v(j) = lambda(j) * B * v(j).
    ///
    ///  The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
    ///  of (A,B) satisfies
    ///
    ///  u(j)**H * A  = lambda(j) * u(j)**H * B .
    ///  where u(j)**H is the conjugate-transpose of u(j).
    ///
    /// What is going on below?
    ///  If the j-th and (j+1)-th eigenvalues form a complex conjugate pair,
    ///  then u(j) = VSL(:,j)+i*VSL(:,j+1) and u(j+1) = VSL(:,j)-i*VSL(:,j+1).
    ///  and then v(j) = VSR(:,j)+i*VSR(:,j+1) and v(j+1) = VSR(:,j)-i*VSR(:,j+1).
    pub fn eigenvectors(self) -> (OMatrix<Complex<T>, D, D>, OMatrix<Complex<T>, D, D>)
    where
        DefaultAllocator:
            Allocator<Complex<T>, D, D> + Allocator<Complex<T>, D> + Allocator<(Complex<T>, T), D>,
    {
        let n = self.vsl.shape().0;

        let mut l = self
            .vsl
            .clone()
            .map(|x| Complex::new(x, T::RealField::zero()));

        let mut r = self
            .vsr
            .clone()
            .map(|x| Complex::new(x, T::RealField::zero()));

        let eigenvalues = &self.eigenvalues();

        let mut c = 0;

        let epsilon = T::RealField::default_epsilon();

        while c < n {
            if eigenvalues[c].im.abs() > epsilon && c + 1 < n && {
                let e_conj = eigenvalues[c].conj();
                let e = eigenvalues[c + 1];
                (&e_conj.re).ulps_eq(&e.re, epsilon, 6) && (&e_conj.im).ulps_eq(&e.im, epsilon, 6)
            } {
                // taking care of the left eigenvector matrix
                l.column_mut(c).zip_apply(&self.vsl.column(c + 1), |r, i| {
                    *r = Complex::new(r.re.clone(), i.clone());
                });
                l.column_mut(c + 1).zip_apply(&self.vsl.column(c), |i, r| {
                    *i = Complex::new(r.clone(), -i.re.clone());
                });

                // taking care of the right eigenvector matrix
                r.column_mut(c).zip_apply(&self.vsr.column(c + 1), |r, i| {
                    *r = Complex::new(r.re.clone(), i.clone());
                });
                r.column_mut(c + 1).zip_apply(&self.vsr.column(c), |i, r| {
                    *i = Complex::new(r.clone(), -i.re.clone());
                });

                c += 2;
            } else {
                c += 1;
            }
        }

        (l, r)
    }

    /// computes the generalized eigenvalues i.e values of lambda  that satisfy the following equation
    /// determinant(A - lambda* B) = 0
    #[must_use]
    pub fn eigenvalues(&self) -> OVector<Complex<T>, D>
    where
        DefaultAllocator: Allocator<Complex<T>, D>,
    {
        let mut out = Matrix::zeros_generic(self.vsl.shape_generic().0, Const::<1>);

        for i in 0..out.len() {
            out[i] = if self.beta[i].clone().abs() < T::RealField::default_epsilon() {
                Complex::zero()
            } else {
                Complex::new(self.alphar[i].clone(), self.alphai[i].clone())
                    * (Complex::new(self.beta[i].clone(), T::RealField::zero()).inv())
            }
        }

        out
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
            out[i] = (Complex::new(self.alphar[i], self.alphai[i]), self.beta[i])
        }

        out
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the RealField GE decomposition.
pub trait GEScalar: Scalar {
    #[allow(missing_docs)]
    fn xggev(
        jobvsl: u8,
        jobvsr: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        alphar: &mut [Self],
        alphai: &mut [Self],
        beta: &mut [Self],
        vsl: &mut [Self],
        ldvsl: i32,
        vsr: &mut [Self],
        ldvsr: i32,
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );

    #[allow(missing_docs)]
    fn xggev_work_size(
        jobvsl: u8,
        jobvsr: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        alphar: &mut [Self],
        alphai: &mut [Self],
        beta: &mut [Self],
        vsl: &mut [Self],
        ldvsl: i32,
        vsr: &mut [Self],
        ldvsr: i32,
        info: &mut i32,
    ) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xggev: path) => (
        impl GEScalar for $N {
            #[inline]
            fn xggev(jobvsl:  u8,
                     jobvsr:  u8,
                     n:      i32,
                     a:      &mut [$N],
                     lda:    i32,
                     b:      &mut [$N],
                     ldb:    i32,
                     alphar: &mut [$N],
                     alphai: &mut [$N],
                     beta  : &mut [$N],
                     vsl:    &mut [$N],
                     ldvsl:  i32,
                     vsr:    &mut [$N],
                     ldvsr:  i32,
                     work:   &mut [$N],
                     lwork:  i32,
                     info:   &mut i32) {
                unsafe { $xggev(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, info); }
            }


            #[inline]
            fn xggev_work_size(jobvsl:  u8,
                               jobvsr:  u8,
                               n:      i32,
                               a:      &mut [$N],
                               lda:    i32,
                               b:      &mut [$N],
                               ldb:    i32,
                               alphar: &mut [$N],
                               alphai: &mut [$N],
                               beta  : &mut [$N],
                               vsl:    &mut [$N],
                               ldvsl:  i32,
                               vsr:    &mut [$N],
                               ldvsr:  i32,
                               info:   &mut i32)
                               -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xggev(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::sggev);
real_eigensystem_scalar_impl!(f64, lapack::dggev);
