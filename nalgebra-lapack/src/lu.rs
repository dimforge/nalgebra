use num::{One, Zero};
use num_complex::Complex;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Dim, DimMin, DimMinimum, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Scalar, VectorN};

use lapack;

/// LU decomposition with partial pivoting.
///
/// This decomposes a matrix `M` with m rows and n columns into three parts:
/// * `L` which is a `m × min(m, n)` lower-triangular matrix.
/// * `U` which is a `min(m, n) × n` upper-triangular matrix.
/// * `P` which is a `m * m` permutation matrix.
///
/// Those are such that `M == P * L * U`.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<i32, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<i32, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct LU<N: Scalar, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<i32, DimMinimum<R, C>> + Allocator<N, R, C>,
{
    lu: MatrixMN<N, R, C>,
    p: VectorN<i32, DimMinimum<R, C>>,
}

impl<N: Scalar + Copy, R: DimMin<C>, C: Dim> Copy for LU<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<i32, DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    VectorN<i32, DimMinimum<R, C>>: Copy,
{
}

impl<N: LUScalar, R: Dim, C: Dim> LU<N, R, C>
where
    N: Zero + One,
    R: DimMin<C>,
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, R, R>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<i32, DimMinimum<R, C>>,
{
    /// Computes the LU decomposition with partial (row) pivoting of `matrix`.
    pub fn new(mut m: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = m.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let nrows = nrows.value() as i32;
        let ncols = ncols.value() as i32;

        let mut ipiv: VectorN<i32, _> = Matrix::zeros_generic(min_nrows_ncols, U1);

        let mut info = 0;

        N::xgetrf(
            nrows,
            ncols,
            m.as_mut_slice(),
            nrows,
            ipiv.as_mut_slice(),
            &mut info,
        );
        lapack_panic!(info);

        Self { lu: m, p: ipiv }
    }

    /// Gets the lower-triangular matrix part of the decomposition.
    #[inline]
    pub fn l(&self) -> MatrixMN<N, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.lu.data.shape();
        let mut res = self.lu.columns_generic(0, nrows.min(ncols)).into_owned();

        res.fill_upper_triangle(Zero::zero(), 1);
        res.fill_diagonal(One::one());

        res
    }

    /// Gets the upper-triangular matrix part of the decomposition.
    #[inline]
    pub fn u(&self) -> MatrixMN<N, DimMinimum<R, C>, C> {
        let (nrows, ncols) = self.lu.data.shape();
        let mut res = self.lu.rows_generic(0, nrows.min(ncols)).into_owned();

        res.fill_lower_triangle(Zero::zero(), 1);

        res
    }

    /// Gets the row permutation matrix of this decomposition.
    ///
    /// Computing the permutation matrix explicitly is costly and usually not necessary.
    /// To permute rows of a matrix or vector, use the method `self.permute(...)` instead.
    #[inline]
    pub fn p(&self) -> MatrixN<N, R> {
        let (dim, _) = self.lu.data.shape();
        let mut id = Matrix::identity_generic(dim, dim);
        self.permute(&mut id);

        id
    }

    // TODO: when we support resizing a matrix, we could add unwrap_u/unwrap_l that would
    // re-use the memory from the internal matrix!

    /// Gets the LAPACK permutation indices.
    #[inline]
    pub fn permutation_indices(&self) -> &VectorN<i32, DimMinimum<R, C>> {
        &self.p
    }

    /// Applies the permutation matrix to a given matrix or vector in-place.
    #[inline]
    pub fn permute<C2: Dim>(&self, rhs: &mut MatrixMN<N, R, C2>)
    where
        DefaultAllocator: Allocator<N, R, C2>,
    {
        let (nrows, ncols) = rhs.shape();

        N::xlaswp(
            ncols as i32,
            rhs.as_mut_slice(),
            nrows as i32,
            1,
            self.p.len() as i32,
            self.p.as_slice(),
            -1,
        );
    }

    fn generic_solve_mut<R2: Dim, C2: Dim>(&self, trans: u8, b: &mut MatrixMN<N, R2, C2>) -> bool
    where
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        let dim = self.lu.nrows();

        assert!(
            self.lu.is_square(),
            "Unable to solve a set of under/over-determined equations."
        );
        assert!(
            b.nrows() == dim,
            "The number of rows of `b` must be equal to the dimension of the matrix `a`."
        );

        let nrhs = b.ncols() as i32;
        let lda = dim as i32;
        let ldb = dim as i32;
        let mut info = 0;

        N::xgetrs(
            trans,
            dim as i32,
            nrhs,
            self.lu.as_slice(),
            lda,
            self.p.as_slice(),
            b.as_mut_slice(),
            ldb,
            &mut info,
        );
        lapack_test!(info)
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        let mut res = b.clone_owned();
        if self.generic_solve_mut(b'N', &mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self.transpose() * x = b`, where `x` is the unknown to be
    /// determined.
    pub fn solve_transpose<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        let mut res = b.clone_owned();
        if self.generic_solve_mut(b'T', &mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self.adjoint() * x = b`, where `x` is the unknown to
    /// be determined.
    pub fn solve_conjugate_transpose<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        let mut res = b.clone_owned();
        if self.generic_solve_mut(b'T', &mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves in-place the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `false` if no solution was found (the decomposed matrix is singular).
    pub fn solve_mut<R2: Dim, C2: Dim>(&self, b: &mut MatrixMN<N, R2, C2>) -> bool
    where
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        self.generic_solve_mut(b'N', b)
    }

    /// Solves in-place the linear system `self.transpose() * x = b`, where `x` is the unknown to be
    /// determined.
    ///
    /// Returns `false` if no solution was found (the decomposed matrix is singular).
    pub fn solve_transpose_mut<R2: Dim, C2: Dim>(&self, b: &mut MatrixMN<N, R2, C2>) -> bool
    where
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        self.generic_solve_mut(b'T', b)
    }

    /// Solves in-place the linear system `self.adjoint() * x = b`, where `x` is the unknown to
    /// be determined.
    ///
    /// Returns `false` if no solution was found (the decomposed matrix is singular).
    pub fn solve_adjoint_mut<R2: Dim, C2: Dim>(&self, b: &mut MatrixMN<N, R2, C2>) -> bool
    where
        DefaultAllocator: Allocator<N, R2, C2> + Allocator<i32, R2>,
    {
        self.generic_solve_mut(b'T', b)
    }
}

impl<N: LUScalar, D: Dim> LU<N, D, D>
where
    N: Zero + One,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<i32, D>,
{
    /// Computes the inverse of the decomposed matrix.
    pub fn inverse(mut self) -> Option<MatrixN<N, D>> {
        let dim = self.lu.nrows() as i32;
        let mut info = 0;
        let lwork = N::xgetri_work_size(
            dim,
            self.lu.as_mut_slice(),
            dim,
            self.p.as_mut_slice(),
            &mut info,
        );
        lapack_check!(info);

        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        N::xgetri(
            dim,
            self.lu.as_mut_slice(),
            dim,
            self.p.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );
        lapack_check!(info);

        Some(self.lu)
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the LU decomposition.
pub trait LUScalar: Scalar + Copy {
    #[allow(missing_docs)]
    fn xgetrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    #[allow(missing_docs)]
    fn xlaswp(n: i32, a: &mut [Self], lda: i32, k1: i32, k2: i32, ipiv: &[i32], incx: i32);
    #[allow(missing_docs)]
    fn xgetrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xgetri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xgetri_work_size(n: i32, a: &mut [Self], lda: i32, ipiv: &[i32], info: &mut i32) -> i32;
}

macro_rules! lup_scalar_impl(
    ($N: ty, $xgetrf: path, $xlaswp: path, $xgetrs: path, $xgetri: path) => (
        impl LUScalar for $N {
            #[inline]
            fn xgetrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
                unsafe { $xgetrf(m, n, a, lda, ipiv, info) }
            }

            #[inline]
            fn xlaswp(n: i32, a: &mut [Self], lda: i32, k1: i32, k2: i32, ipiv: &[i32], incx: i32) {
                unsafe { $xlaswp(n, a, lda, k1, k2, ipiv, incx) }
            }

            #[inline]
            fn xgetrs(trans: u8, n: i32, nrhs: i32, a: &[Self], lda: i32, ipiv: &[i32],
                      b: &mut [Self], ldb: i32, info: &mut i32) {
                unsafe { $xgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info) }
            }

            #[inline]
            fn xgetri(n: i32, a: &mut [Self], lda: i32, ipiv: &[i32],
                      work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xgetri(n, a, lda, ipiv, work, lwork, info) }
            }

            #[inline]
            fn xgetri_work_size(n: i32, a: &mut [Self], lda: i32, ipiv: &[i32], info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgetri(n, a, lda, ipiv, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

lup_scalar_impl!(
    f32,
    lapack::sgetrf,
    lapack::slaswp,
    lapack::sgetrs,
    lapack::sgetri
);
lup_scalar_impl!(
    f64,
    lapack::dgetrf,
    lapack::dlaswp,
    lapack::dgetrs,
    lapack::dgetri
);
lup_scalar_impl!(
    Complex<f32>,
    lapack::cgetrf,
    lapack::claswp,
    lapack::cgetrs,
    lapack::cgetri
);
lup_scalar_impl!(
    Complex<f64>,
    lapack::zgetrf,
    lapack::zlaswp,
    lapack::zgetrs,
    lapack::zgetri
);
