#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use alga::general::Real;

use na::allocator::Allocator;
use na::dimension::{Dim, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixN, Scalar, VectorN};
use ComplexHelper;

use lapack;

/// Eigendecomposition of a real square matrix with real eigenvalues.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Eigen<N: Scalar, D: Dim>
where DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>
{
    /// The eigenvalues of the decomposed matrix.
    pub eigenvalues: VectorN<N, D>,
    /// The (right) eigenvectors of the decomposed matrix.
    pub eigenvectors: Option<MatrixN<N, D>>,
    /// The left eigenvectors of the decomposed matrix.
    pub left_eigenvectors: Option<MatrixN<N, D>>,
}

impl<N: Scalar, D: Dim> Copy for Eigen<N, D>
where
    DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>,
    VectorN<N, D>: Copy,
    MatrixN<N, D>: Copy,
{}

impl<N: EigenScalar + Real, D: Dim> Eigen<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Computes the eigenvalues and eigenvectors of the square matrix `m`.
    ///
    /// If `eigenvectors` is `false` then, the eigenvectors are not computed explicitly.
    pub fn new(
        mut m: MatrixN<N, D>,
        left_eigenvectors: bool,
        eigenvectors: bool,
    ) -> Option<Eigen<N, D>>
    {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let ljob = if left_eigenvectors { b'V' } else { b'N' };
        let rjob = if eigenvectors { b'V' } else { b'N' };

        let (nrows, ncols) = m.data.shape();
        let n = nrows.value();

        let lda = n as i32;

        let mut wr = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        // FIXME: Tap into the workspace.
        let mut wi = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };

        let mut info = 0;
        let mut placeholder1 = [N::zero()];
        let mut placeholder2 = [N::zero()];

        let lwork = N::xgeev_work_size(
            ljob,
            rjob,
            n as i32,
            m.as_mut_slice(),
            lda,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            &mut placeholder1,
            n as i32,
            &mut placeholder2,
            n as i32,
            &mut info,
        );

        lapack_check!(info);

        let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

        match (left_eigenvectors, eigenvectors) {
            (true, true) => {
                let mut vl = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };
                let mut vr = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(
                    ljob,
                    rjob,
                    n as i32,
                    m.as_mut_slice(),
                    lda,
                    wr.as_mut_slice(),
                    wi.as_mut_slice(),
                    &mut vl.as_mut_slice(),
                    n as i32,
                    &mut vr.as_mut_slice(),
                    n as i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(Eigen {
                        eigenvalues: wr,
                        left_eigenvectors: Some(vl),
                        eigenvectors: Some(vr),
                    });
                }
            }
            (true, false) => {
                let mut vl = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(
                    ljob,
                    rjob,
                    n as i32,
                    m.as_mut_slice(),
                    lda,
                    wr.as_mut_slice(),
                    wi.as_mut_slice(),
                    &mut vl.as_mut_slice(),
                    n as i32,
                    &mut placeholder2,
                    1 as i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(Eigen {
                        eigenvalues: wr,
                        left_eigenvectors: Some(vl),
                        eigenvectors: None,
                    });
                }
            }
            (false, true) => {
                let mut vr = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(
                    ljob,
                    rjob,
                    n as i32,
                    m.as_mut_slice(),
                    lda,
                    wr.as_mut_slice(),
                    wi.as_mut_slice(),
                    &mut placeholder1,
                    1 as i32,
                    &mut vr.as_mut_slice(),
                    n as i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(Eigen {
                        eigenvalues: wr,
                        left_eigenvectors: None,
                        eigenvectors: Some(vr),
                    });
                }
            }
            (false, false) => {
                N::xgeev(
                    ljob,
                    rjob,
                    n as i32,
                    m.as_mut_slice(),
                    lda,
                    wr.as_mut_slice(),
                    wi.as_mut_slice(),
                    &mut placeholder1,
                    1 as i32,
                    &mut placeholder2,
                    1 as i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(Eigen {
                        eigenvalues: wr,
                        left_eigenvectors: None,
                        eigenvectors: None,
                    });
                }
            }
        }

        None
    }

    /// The complex eigenvalues of the given matrix.
    ///
    /// Panics if the eigenvalue computation does not converge.
    pub fn complex_eigenvalues(mut m: MatrixN<N, D>) -> VectorN<Complex<N>, D>
    where DefaultAllocator: Allocator<Complex<N>, D> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let nrows = m.data.shape().0;
        let n = nrows.value();

        let lda = n as i32;

        let mut wr = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        let mut wi = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };

        let mut info = 0;
        let mut placeholder1 = [N::zero()];
        let mut placeholder2 = [N::zero()];

        let lwork = N::xgeev_work_size(
            b'N',
            b'N',
            n as i32,
            m.as_mut_slice(),
            lda,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            &mut placeholder1,
            n as i32,
            &mut placeholder2,
            n as i32,
            &mut info,
        );

        lapack_panic!(info);

        let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

        N::xgeev(
            b'N',
            b'N',
            n as i32,
            m.as_mut_slice(),
            lda,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            &mut placeholder1,
            1 as i32,
            &mut placeholder2,
            1 as i32,
            &mut work,
            lwork,
            &mut info,
        );
        lapack_panic!(info);

        let mut res = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };

        for i in 0..res.len() {
            res[i] = Complex::new(wr[i], wi[i]);
        }

        res
    }

    /// The determinant of the decomposed matrix.
    #[inline]
    pub fn determinant(&self) -> N {
        let mut det = N::one();
        for e in self.eigenvalues.iter() {
            det *= *e;
        }

        det
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalar type for which Lapack function exist to compute the
/// eigendecomposition.
pub trait EigenScalar: Scalar {
    #[allow(missing_docs)]
    fn xgeev(
        jobvl: u8,
        jobvr: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vl: &mut [Self],
        ldvl: i32,
        vr: &mut [Self],
        ldvr: i32,
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xgeev_work_size(
        jobvl: u8,
        jobvr: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vl: &mut [Self],
        ldvl: i32,
        vr: &mut [Self],
        ldvr: i32,
        info: &mut i32,
    ) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xgeev: path) => (
        impl EigenScalar for $N {
            #[inline]
            fn xgeev(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
                     wr: &mut [Self], wi: &mut [Self],
                     vl: &mut [Self], ldvl: i32, vr: &mut [Self], ldvr: i32,
                     work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info) }
            }


            #[inline]
            fn xgeev_work_size(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
                               wr: &mut [Self], wi: &mut [Self], vl: &mut [Self], ldvl: i32,
                               vr: &mut [Self], ldvr: i32, info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &mut work, lwork, info) };
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::sgeev);
real_eigensystem_scalar_impl!(f64, lapack::dgeev);

//// FIXME: decomposition of complex matrix and matrices with complex eigenvalues.
// eigensystem_complex_impl!(f32, lapack::cgeev);
// eigensystem_complex_impl!(f64, lapack::zgeev);
