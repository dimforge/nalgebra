#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use simba::scalar::RealField;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Dim, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixN, Scalar, VectorN};

use lapack;

/// Eigendecomposition of a real square matrix with real eigenvalues.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(serialize = "DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Serialize")
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(deserialize = "DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Deserialize<'de>")
    )
)]
#[derive(Clone, Debug)]
pub struct Schur<N: Scalar, D: Dim>
where DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>
{
    re: VectorN<N, D>,
    im: VectorN<N, D>,
    t: MatrixN<N, D>,
    q: MatrixN<N, D>,
}

impl<N: Scalar + Copy, D: Dim> Copy for Schur<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
    MatrixN<N, D>: Copy,
    VectorN<N, D>: Copy,
{
}

impl<N: SchurScalar + RealField, D: Dim> Schur<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Computes the eigenvalues and real Schur form of the matrix `m`.
    ///
    /// Panics if the method did not converge.
    pub fn new(m: MatrixN<N, D>) -> Self {
        Self::try_new(m).expect("Schur decomposition: convergence failed.")
    }

    /// Computes the eigenvalues and real Schur form of the matrix `m`.
    ///
    /// Returns `None` if the method did not converge.
    pub fn try_new(mut m: MatrixN<N, D>) -> Option<Self> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let (nrows, ncols) = m.data.shape();
        let n = nrows.value();

        let lda = n as i32;

        let mut info = 0;

        let mut wr = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        let mut wi = unsafe { Matrix::new_uninitialized_generic(nrows, U1) };
        let mut q = unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };
        // Placeholders:
        let mut bwork = [0i32];
        let mut unused = 0;

        let lwork = N::xgees_work_size(
            b'V',
            b'N',
            n as i32,
            m.as_mut_slice(),
            lda,
            &mut unused,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            q.as_mut_slice(),
            n as i32,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        N::xgees(
            b'V',
            b'N',
            n as i32,
            m.as_mut_slice(),
            lda,
            &mut unused,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            q.as_mut_slice(),
            n as i32,
            &mut work,
            lwork,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        Some(Schur {
            re: wr,
            im: wi,
            t: m,
            q: q,
        })
    }

    /// Retrieves the unitary matrix `Q` and the upper-quasitriangular matrix `T` such that the
    /// decomposed matrix equals `Q * T * Q.transpose()`.
    pub fn unpack(self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        (self.q, self.t)
    }

    /// Computes the real eigenvalues of the decomposed matrix.
    ///
    /// Return `None` if some eigenvalues are complex.
    pub fn eigenvalues(&self) -> Option<VectorN<N, D>> {
        if self.im.iter().all(|e| e.is_zero()) {
            Some(self.re.clone())
        } else {
            None
        }
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    pub fn complex_eigenvalues(&self) -> VectorN<Complex<N>, D>
    where DefaultAllocator: Allocator<Complex<N>, D> {
        let mut out = unsafe { VectorN::new_uninitialized_generic(self.t.data.shape().0, U1) };

        for i in 0..out.len() {
            out[i] = Complex::new(self.re[i], self.im[i])
        }

        out
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the RealField Schur decomposition.
pub trait SchurScalar: Scalar {
    #[allow(missing_docs)]
    fn xgees(
        jobvs: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        sdim: &mut i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vs: &mut [Self],
        ldvs: i32,
        work: &mut [Self],
        lwork: i32,
        bwork: &mut [i32],
        info: &mut i32,
    );

    #[allow(missing_docs)]
    fn xgees_work_size(
        jobvs: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        sdim: &mut i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vs: &mut [Self],
        ldvs: i32,
        bwork: &mut [i32],
        info: &mut i32,
    ) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xgees: path) => (
        impl SchurScalar for $N {
            #[inline]
            fn xgees(jobvs:  u8,
                     sort:   u8,
                     // select: ???
                     n:      i32,
                     a:      &mut [$N],
                     lda:    i32,
                     sdim:   &mut i32,
                     wr:     &mut [$N],
                     wi:     &mut [$N],
                     vs:     &mut [$N],
                     ldvs:   i32,
                     work:   &mut [$N],
                     lwork:  i32,
                     bwork:  &mut [i32],
                     info:   &mut i32) {
                unsafe { $xgees(jobvs, sort, None, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info); }
            }


            #[inline]
            fn xgees_work_size(jobvs:  u8,
                               sort:   u8,
                               // select: ???
                               n:      i32,
                               a:      &mut [$N],
                               lda:    i32,
                               sdim:   &mut i32,
                               wr:     &mut [$N],
                               wi:     &mut [$N],
                               vs:     &mut [$N],
                               ldvs:   i32,
                               bwork:  &mut [i32],
                               info:   &mut i32)
                               -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xgees(jobvs, sort, None, n, a, lda, sdim, wr, wi, vs, ldvs, &mut work, lwork, bwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::sgees);
real_eigensystem_scalar_impl!(f64, lapack::dgees);
