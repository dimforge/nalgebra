use num::Zero;

use alga::general::Real;

use ::ComplexHelper;
use na::{Scalar, DefaultAllocator, Matrix, VectorN, MatrixN};
use na::dimension::{Dim, U1};
use na::storage::Storage;
use na::allocator::Allocator;

use lapack::fortran as interface;

/// Eigendecomposition of a real square matrix with real eigenvalues.
pub struct RealEigensystem<N: Scalar, D: Dim>
    where DefaultAllocator: Allocator<N, D> +
             Allocator<N, D, D> {
    pub eigenvalues:       VectorN<N, D>,
    pub eigenvectors:      Option<MatrixN<N, D>>,
    pub left_eigenvectors: Option<MatrixN<N, D>>
}


impl<N: RealEigensystemScalar + Real, D: Dim> RealEigensystem<N, D>
    where DefaultAllocator: Allocator<N, D, D> +
                            Allocator<N, D> {
    /// Computes the eigenvalues and eigenvectors of the square matrix `m`.
    ///
    /// If `eigenvectors` is `false` then, the eigenvectors are not computed explicitly.
    pub fn new(mut m: MatrixN<N, D>, left_eigenvectors: bool, eigenvectors: bool)
           -> Option<RealEigensystem<N, D>> {

        assert!(m.is_square(), "Unable to compute the eigenvalue decomposition of a non-square matrix.");

        let ljob = if left_eigenvectors  { b'V' } else { b'N' };
        let rjob = if eigenvectors { b'V' } else { b'N' };

        let (nrows, ncols) = m.data.shape();
        let n = nrows.value();

        let lda = n as i32;

        let mut wr = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        // FIXME: Tap into the workspace.
        let mut wi = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };


        let mut info = 0;
        let mut placeholder1 = [ N::zero() ];
        let mut placeholder2 = [ N::zero() ];

        let lwork = N::xgeev_work_size(ljob, rjob, n as i32, m.as_mut_slice(), lda,
                                       wr.as_mut_slice(), wi.as_mut_slice(), &mut placeholder1,
                                       n as i32, &mut placeholder2, n as i32, &mut info);

        lapack_check!(info);

        let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

        match (left_eigenvectors, eigenvectors) {
            (true, true) => {
                let mut vl = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };
                let mut vr = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(ljob, rjob, n as i32, m.as_mut_slice(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), &mut vl.as_mut_slice(), n as i32, &mut vr.as_mut_slice(),
                    n as i32, &mut work, lwork, &mut info);
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(RealEigensystem {
                        eigenvalues: wr, left_eigenvectors: Some(vl), eigenvectors: Some(vr)
                    })
                }
            },
            (true, false) => {
                let mut vl = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(ljob, rjob, n as i32, m.as_mut_slice(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), &mut vl.as_mut_slice(), n as i32, &mut placeholder2,
                    1 as i32, &mut work, lwork, &mut info);
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(RealEigensystem {
                        eigenvalues: wr, left_eigenvectors: Some(vl), eigenvectors: None
                    });
                }
            },
            (false, true) => {
                let mut vr = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

                N::xgeev(ljob, rjob, n as i32, m.as_mut_slice(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), &mut placeholder1, 1 as i32, &mut vr.as_mut_slice(),
                    n as i32, &mut work, lwork, &mut info);
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(RealEigensystem {
                        eigenvalues: wr, left_eigenvectors: None, eigenvectors: Some(vr)
                    });
                }
            },
            (false, false) => {
                N::xgeev(ljob, rjob, n as i32, m.as_mut_slice(), lda, wr.as_mut_slice(),
                    wi.as_mut_slice(), &mut placeholder1, 1 as i32, &mut placeholder2,
                    1 as i32, &mut work, lwork, &mut info);
                lapack_check!(info);

                if wi.iter().all(|e| e.is_zero()) {
                    return Some(RealEigensystem {
                        eigenvalues: wr, left_eigenvectors: None, eigenvectors: None
                    });
                }
            }
        }

        None
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
pub trait RealEigensystemScalar: Scalar {
    fn xgeev(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
             wr: &mut [Self], wi: &mut [Self],
             vl: &mut [Self], ldvl: i32, vr: &mut [Self], ldvr: i32,
             work: &mut [Self], lwork: i32, info: &mut i32);
    fn xgeev_work_size(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
                       wr: &mut [Self], wi: &mut [Self], vl: &mut [Self], ldvl: i32,
                       vr: &mut [Self], ldvr: i32, info: &mut i32) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xgeev: path) => (
        impl RealEigensystemScalar for $N {
            #[inline]
            fn xgeev(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
                     wr: &mut [Self], wi: &mut [Self],
                     vl: &mut [Self], ldvl: i32, vr: &mut [Self], ldvr: i32,
                     work: &mut [Self], lwork: i32, info: &mut i32) {
                $xgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
            }


            #[inline]
            fn xgeev_work_size(jobvl: u8, jobvr: u8, n: i32, a: &mut [Self], lda: i32,
                               wr: &mut [Self], wi: &mut [Self], vl: &mut [Self], ldvl: i32,
                               vr: &mut [Self], ldvr: i32, info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                $xgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &mut work, lwork, info);
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, interface::sgeev);
real_eigensystem_scalar_impl!(f64, interface::dgeev);

//// FIXME: decomposition of complex matrix and matrices with complex eigenvalues.
// eigensystem_complex_impl!(f32, interface::cgeev);
// eigensystem_complex_impl!(f64, interface::zgeev);
