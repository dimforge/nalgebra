#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

use num::Zero;
use num_complex::Complex;

use na::allocator::Allocator;
use na::dimension::Dim;
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Scalar};

use lapack;

/// The cholesky decomposion of a symmetric-definite-positive matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            serialize = "DefaultAllocator: Allocator<N, D>,
         MatrixN<N, D>: Serialize"
        )
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            deserialize = "DefaultAllocator: Allocator<N, D>,
         MatrixN<N, D>: Deserialize<'de>"
        )
    )
)]
#[derive(Clone, Debug)]
pub struct Cholesky<N: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    l: MatrixN<N, D>,
}

impl<N: Scalar, D: Dim> Copy for Cholesky<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    MatrixN<N, D>: Copy,
{
}

impl<N: CholeskyScalar + Zero, D: Dim> Cholesky<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Complutes the cholesky decomposition of the given symmetric-definite-positive square
    /// matrix.
    ///
    /// Only the lower-triangular part of the input matrix is considered.
    #[inline]
    pub fn new(mut m: MatrixN<N, D>) -> Option<Self> {
        // FIXME: check symmetry as well?
        assert!(
            m.is_square(),
            "Unable to compute the cholesky decomposition of a non-square matrix."
        );

        let uplo = b'L';
        let dim = m.nrows() as i32;
        let mut info = 0;

        N::xpotrf(uplo, dim, m.as_mut_slice(), dim, &mut info);
        lapack_check!(info);

        Some(Cholesky { l: m })
    }

    /// Retrieves the lower-triangular factor of the cholesky decomposition.
    pub fn unpack(mut self) -> MatrixN<N, D> {
        self.l.fill_upper_triangle(Zero::zero(), 1);
        self.l
    }

    /// Retrieves the lower-triangular factor of che cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// This is an allocation-less version of `self.l()`. The values of the strict upper-triangular
    /// part are garbage and should be ignored by further computations.
    pub fn unpack_dirty(self) -> MatrixN<N, D> {
        self.l
    }

    /// Retrieves the lower-triangular factor of the cholesky decomposition.
    pub fn l(&self) -> MatrixN<N, D> {
        let mut res = self.l.clone();
        res.fill_upper_triangle(Zero::zero(), 1);
        res
    }

    /// Retrieves the lower-triangular factor of the cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// This is an allocation-less version of `self.l()`. The values of the strict upper-triangular
    /// part are garbage and should be ignored by further computations.
    pub fn l_dirty(&self) -> &MatrixN<N, D> {
        &self.l
    }

    /// Solves the symmetric-definite-positive linear system `self * x = b`, where `x` is the
    /// unknown to be determined.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
    {
        let mut res = b.clone_owned();
        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves in-place the symmetric-definite-positive linear system `self * x = b`, where `x` is
    /// the unknown to be determined.
    pub fn solve_mut<R2: Dim, C2: Dim>(&self, b: &mut MatrixMN<N, R2, C2>) -> bool
    where
        DefaultAllocator: Allocator<N, R2, C2>,
    {
        let dim = self.l.nrows();

        assert!(
            b.nrows() == dim,
            "The number of rows of `b` must be equal to the dimension of the matrix `a`."
        );

        let nrhs = b.ncols() as i32;
        let lda = dim as i32;
        let ldb = dim as i32;
        let mut info = 0;

        N::xpotrs(
            b'L',
            dim as i32,
            nrhs,
            self.l.as_slice(),
            lda,
            b.as_mut_slice(),
            ldb,
            &mut info,
        );
        lapack_test!(info)
    }

    /// Computes the inverse of the decomposed matrix.
    pub fn inverse(mut self) -> Option<MatrixN<N, D>> {
        let dim = self.l.nrows();
        let mut info = 0;

        N::xpotri(
            b'L',
            dim as i32,
            self.l.as_mut_slice(),
            dim as i32,
            &mut info,
        );
        lapack_check!(info);

        // Copy lower triangle to upper triangle.
        for i in 0..dim {
            for j in i + 1..dim {
                unsafe { *self.l.get_unchecked_mut(i, j) = *self.l.get_unchecked(j, i) };
            }
        }

        Some(self.l)
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by floats (`f32`, `f64`) and complex floats (`Complex<f32>`, `Complex<f64>`)
/// supported by the cholesky decompotition.
pub trait CholeskyScalar: Scalar {
    #[allow(missing_docs)]
    fn xpotrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32);
    #[allow(missing_docs)]
    fn xpotrs(
        uplo: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xpotri(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32);
}

macro_rules! cholesky_scalar_impl(
    ($N: ty, $xpotrf: path, $xpotrs: path, $xpotri: path) => (
        impl CholeskyScalar for $N {
            #[inline]
            fn xpotrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) {
                unsafe { $xpotrf(uplo, n, a, lda, info) }
            }

            #[inline]
            fn xpotrs(uplo: u8, n: i32, nrhs: i32, a: &[Self], lda: i32,
                      b: &mut [Self], ldb: i32, info: &mut i32) {
                unsafe { $xpotrs(uplo, n, nrhs, a, lda, b, ldb, info) }
            }

            #[inline]
            fn xpotri(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) {
                unsafe { $xpotri(uplo, n, a, lda, info) }
            }
        }
    )
);

cholesky_scalar_impl!(f32, lapack::spotrf, lapack::spotrs, lapack::spotri);
cholesky_scalar_impl!(f64, lapack::dpotrf, lapack::dpotrs, lapack::dpotri);
cholesky_scalar_impl!(Complex<f32>, lapack::cpotrf, lapack::cpotrs, lapack::cpotri);
cholesky_scalar_impl!(Complex<f64>, lapack::zpotrf, lapack::zpotrs, lapack::zpotri);
