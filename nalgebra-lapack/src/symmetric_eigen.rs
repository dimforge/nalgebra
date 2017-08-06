use num::Zero;
use std::ops::MulAssign;

use alga::general::Real;

use ::ComplexHelper;
use na::{Scalar, DefaultAllocator, Matrix, VectorN, MatrixN};
use na::dimension::{Dim, U1};
use na::storage::Storage;
use na::allocator::Allocator;

use lapack::fortran as interface;

/// SymmetricEigendecomposition of a real square matrix with real eigenvalues.
pub struct SymmetricEigen<N: Scalar, D: Dim>
    where DefaultAllocator: Allocator<N, D> +
    Allocator<N, D, D> {
    pub eigenvalues:  VectorN<N, D>,
    pub eigenvectors: MatrixN<N, D>,
}


impl<N: SymmetricEigenScalar + Real, D: Dim> SymmetricEigen<N, D>
    where DefaultAllocator: Allocator<N, D, D> +
                            Allocator<N, D> {

    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Panics if the method did not converge.
    pub fn new(m: MatrixN<N, D>) -> Self {
        let (vals, vecs) = Self::do_decompose(m, true).expect("SymmetricEigen: convergence failure.");
        SymmetricEigen { eigenvalues: vals, eigenvectors: vecs.unwrap() }
    }

    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Returns `None` if the method did not converge.
    pub fn try_new(m: MatrixN<N, D>) -> Option<Self> {
        Self::do_decompose(m, true).map(|(vals, vecs)| {
            SymmetricEigen { eigenvalues: vals, eigenvectors: vecs.unwrap() }
        })
    }

    fn do_decompose(mut m: MatrixN<N, D>, eigenvectors: bool) -> Option<(VectorN<N, D>, Option<MatrixN<N, D>>)> {
        assert!(m.is_square(), "Unable to compute the eigenvalue decomposition of a non-square matrix.");

        let jobz = if eigenvectors { b'V' } else { b'N' };

        let (nrows, ncols) = m.data.shape();
        let n = nrows.value();

        let lda = n as i32;

        let mut values = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        let mut info = 0;

        let lwork = N::xsyev_work_size(jobz, b'L', n as i32, m.as_mut_slice(), lda, &mut info);
        lapack_check!(info);

        let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

        N::xsyev(jobz, b'L', n as i32, m.as_mut_slice(), lda, values.as_mut_slice(), &mut work, lwork, &mut info);
        lapack_check!(info);

        let vectors = if eigenvectors { Some(m) } else { None };
        Some((values, vectors))
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Panics if the method does not converge.
    pub fn eigenvalues(mut m: MatrixN<N, D>) -> VectorN<N, D> {
        Self::do_decompose(m, false).expect("SymmetricEigen eigenvalues: convergence failure.").0
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Returns `None` if the method does not converge.
    pub fn try_eigenvalues(mut m: MatrixN<N, D>) -> Option<VectorN<N, D>> {
        Self::do_decompose(m, false).map(|res| res.0)
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

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the eigenvalues have been manually modified.
    pub fn recompose(&self) -> MatrixN<N, D> {
        let mut u_t = self.eigenvectors.clone();
        for i in 0 .. self.eigenvalues.len() {
            let val = self.eigenvalues[i];
            u_t.column_mut(i).mul_assign(val);
        }
        u_t.transpose_mut();
        &self.eigenvectors * u_t
    }
}


/*
 *
 * Lapack functions dispatch.
 *
 */
pub trait SymmetricEigenScalar: Scalar {
    fn xsyev(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self],
             lwork: i32, info: &mut i32);
    fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xsyev: path) => (
        impl SymmetricEigenScalar for $N {
            #[inline]
            fn xsyev(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self],
                     lwork: i32, info: &mut i32) {
                $xsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
            }


            #[inline]
            fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let mut w    = [ Zero::zero() ];
                let lwork    = -1 as i32;

                $xsyev(jobz, uplo, n, a, lda, &mut w, &mut work, lwork, info);
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, interface::ssyev);
real_eigensystem_scalar_impl!(f64, interface::dsyev);
