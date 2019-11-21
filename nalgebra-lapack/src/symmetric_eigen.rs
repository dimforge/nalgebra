#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use std::ops::MulAssign;

use alga::general::RealField;

use na::allocator::Allocator;
use na::dimension::{Dim, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixN, Scalar, VectorN};
use crate::ComplexHelper;

use lapack;

/// Eigendecomposition of a real square symmetric matrix with real eigenvalues.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, D>,
         VectorN<N, D>: Deserialize<'de>,
         MatrixN<N, D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct SymmetricEigen<N: Scalar + Copy, D: Dim>
where DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>
{
    /// The eigenvectors of the decomposed matrix.
    pub eigenvectors: MatrixN<N, D>,

    /// The unsorted eigenvalues of the decomposed matrix.
    pub eigenvalues: VectorN<N, D>,
}

impl<N: Scalar + Copy, D: Dim> Copy for SymmetricEigen<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
    MatrixN<N, D>: Copy,
    VectorN<N, D>: Copy,
{}

impl<N: SymmetricEigenScalar + RealField, D: Dim> SymmetricEigen<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Panics if the method did not converge.
    pub fn new(m: MatrixN<N, D>) -> Self {
        let (vals, vecs) =
            Self::do_decompose(m, true).expect("SymmetricEigen: convergence failure.");
        Self {
            eigenvalues: vals,
            eigenvectors: vecs.unwrap(),
        }
    }

    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Returns `None` if the method did not converge.
    pub fn try_new(m: MatrixN<N, D>) -> Option<Self> {
        Self::do_decompose(m, true).map(|(vals, vecs)| SymmetricEigen {
            eigenvalues: vals,
            eigenvectors: vecs.unwrap(),
        })
    }

    fn do_decompose(
        mut m: MatrixN<N, D>,
        eigenvectors: bool,
    ) -> Option<(VectorN<N, D>, Option<MatrixN<N, D>>)>
    {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let jobz = if eigenvectors { b'V' } else { b'N' };

        let nrows = m.data.shape().0;
        let n = nrows.value();

        let lda = n as i32;

        let mut values = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        let mut info = 0;

        let lwork = N::xsyev_work_size(jobz, b'L', n as i32, m.as_mut_slice(), lda, &mut info);
        lapack_check!(info);

        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        N::xsyev(
            jobz,
            b'L',
            n as i32,
            m.as_mut_slice(),
            lda,
            values.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );
        lapack_check!(info);

        let vectors = if eigenvectors { Some(m) } else { None };
        Some((values, vectors))
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Panics if the method does not converge.
    pub fn eigenvalues(m: MatrixN<N, D>) -> VectorN<N, D> {
        Self::do_decompose(m, false)
            .expect("SymmetricEigen eigenvalues: convergence failure.")
            .0
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Returns `None` if the method does not converge.
    pub fn try_eigenvalues(m: MatrixN<N, D>) -> Option<VectorN<N, D>> {
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
        for i in 0..self.eigenvalues.len() {
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
/// Trait implemented by scalars for which Lapack implements the eigendecomposition of symmetric
/// real matrices.
pub trait SymmetricEigenScalar: Scalar + Copy {
    #[allow(missing_docs)]
    fn xsyev(
        jobz: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        w: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32)
        -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xsyev: path) => (
        impl SymmetricEigenScalar for $N {
            #[inline]
            fn xsyev(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self],
                     lwork: i32, info: &mut i32) {
                unsafe { $xsyev(jobz, uplo, n, a, lda, w, work, lwork, info) }
            }


            #[inline]
            fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let mut w    = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xsyev(jobz, uplo, n, a, lda, &mut w, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::ssyev);
real_eigensystem_scalar_impl!(f64, lapack::dsyev);
