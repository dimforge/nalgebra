#[cfg(feature = "serde-serialize")]
use serde;

use std::cmp;
use num::Signed;

use na::{Scalar, Matrix, VectorN, MatrixN, MatrixMN,
         DefaultAllocator};
use na::dimension::{Dim, DimMin, DimMinimum, U1};
use na::storage::Storage;
use na::allocator::Allocator;

use lapack::fortran as interface;


/// The SVD decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde-serialize",
    serde(bound(serialize =
        "DefaultAllocator: Allocator<N, DimMinimum<R, C>> +
                           Allocator<N, R, R> +
                           Allocator<N, C, C>,
         MatrixN<N, R>: serde::Serialize,
         MatrixN<N, C>: serde::Serialize,
         VectorN<N, DimMinimum<R, C>>: serde::Serialize")))]
#[cfg_attr(feature = "serde-serialize",
    serde(bound(serialize =
        "DefaultAllocator: Allocator<N, DimMinimum<R, C>> +
                           Allocator<N, R, R> +
                           Allocator<N, C, C>,
         MatrixN<N, R>: serde::Deserialize<'de>,
         MatrixN<N, C>: serde::Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: serde::Deserialize<'de>")))]
#[derive(Clone, Debug)]
pub struct SVD<N: Scalar, R: DimMin<C>, C: Dim>
    where DefaultAllocator: Allocator<N, R, R>                 +
                            Allocator<N, DimMinimum<R, C>> +
                            Allocator<N, C, C> {
    /// The left-singular vectors `U` of this SVD.
    pub u:  MatrixN<N, R>, // FIXME: should be MatrixMN<N, R, DimMinimum<R, C>>
    /// The right-singular vectors `V^t` of this SVD.
    pub vt: MatrixN<N, C>, // FIXME: should be MatrixMN<N, DimMinimum<R, C>, C>
    /// The singular values of this SVD.
    pub singular_values: VectorN<N, DimMinimum<R, C>>
}

impl<N: Scalar, R: DimMin<C>, C: Dim> Copy for SVD<N, R, C>
    where DefaultAllocator: Allocator<N, C, C> +
                            Allocator<N, R, R> +
                            Allocator<N, DimMinimum<R, C>>,
          MatrixMN<N, R, R>: Copy,
          MatrixMN<N, C, C>: Copy,
          VectorN<N, DimMinimum<R, C>>: Copy { }

/// Trait implemented by floats (`f32`, `f64`) and complex floats (`Complex<f32>`, `Complex<f64>`)
/// supported by the Singular Value Decompotition.
pub trait SVDScalar<R: DimMin<C>, C: Dim>: Scalar
    where DefaultAllocator: Allocator<Self, R, R>                 +
                            Allocator<Self, R, C>                 +
                            Allocator<Self, DimMinimum<R, C>> +
                            Allocator<Self, C, C> {
    /// Computes the SVD decomposition of `m`.
    fn compute(m: MatrixMN<Self, R, C>) -> Option<SVD<Self, R, C>>;
}

impl<N: SVDScalar<R, C>, R: DimMin<C>, C: Dim> SVD<N, R, C>
    where DefaultAllocator: Allocator<N, R, R>             +
                            Allocator<N, R, C>             +
                            Allocator<N, DimMinimum<R, C>> +
                            Allocator<N, C, C> {
    /// Computes the Singular Value Decomposition of `matrix`.
    pub fn new(m: MatrixMN<N, R, C>) -> Option<Self> {
        N::compute(m)
    }
}

macro_rules! svd_impl(
    ($t: ty, $lapack_func: path) => (
        impl<R: Dim, C: Dim> SVDScalar<R, C> for $t
                where R: DimMin<C>,
                      DefaultAllocator: Allocator<$t, R, C> +
                                        Allocator<$t, R, R> +
                                        Allocator<$t, C, C> +
                                        Allocator<$t, DimMinimum<R, C>> {

            fn compute(mut m: MatrixMN<$t, R, C>) -> Option<SVD<$t, R, C>> {
                let (nrows, ncols) = m.data.shape();

                if nrows.value() == 0 || ncols.value() == 0 {
                    return None;
                }

                let job = b'A';

                let lda = nrows.value() as i32;

                let mut u  = unsafe { Matrix::new_uninitialized_generic(nrows, nrows) };
                let mut s  = unsafe { Matrix::new_uninitialized_generic(nrows.min(ncols), U1) };
                let mut vt = unsafe { Matrix::new_uninitialized_generic(ncols, ncols) };

                let ldu  = nrows.value();
                let ldvt = ncols.value();

                let mut work  = [ 0.0 ];
                let mut lwork = -1 as i32;
                let mut info  = 0;
                let mut iwork = unsafe { ::uninitialized_vec(8 * cmp::min(nrows.value(), ncols.value())) };

                $lapack_func(job, nrows.value() as i32, ncols.value() as i32, m.as_mut_slice(),
                    lda, &mut s.as_mut_slice(), u.as_mut_slice(), ldu as i32, vt.as_mut_slice(),
                    ldvt as i32, &mut work, lwork, &mut iwork, &mut info);
                lapack_check!(info);

                lwork = work[0] as i32;
                let mut work = unsafe { ::uninitialized_vec(lwork as usize) };

                $lapack_func(job, nrows.value() as i32, ncols.value() as i32, m.as_mut_slice(),
                    lda, &mut s.as_mut_slice(), u.as_mut_slice(), ldu as i32, vt.as_mut_slice(),
                    ldvt as i32, &mut work, lwork, &mut iwork, &mut info);
                lapack_check!(info);

                Some(SVD { u: u, singular_values: s, vt: vt })
            }
        }

        impl<R: DimMin<C>, C: Dim> SVD<$t, R, C>
            // FIXME: All those bounds…
            where DefaultAllocator: Allocator<$t, R, C>                 +
                                    Allocator<$t, C, R>                 +
                                    Allocator<$t, U1, R>                +
                                    Allocator<$t, U1, C>                +
                                    Allocator<$t, R, R>                 +
                                    Allocator<$t, DimMinimum<R, C>> +
                                    Allocator<$t, DimMinimum<R, C>, R>  +
                                    Allocator<$t, DimMinimum<R, C>, C>  +
                                    Allocator<$t, R, DimMinimum<R, C>>  +
                                    Allocator<$t, C, C> {
            /// Reconstructs the matrix from its decomposition.
            ///
            /// Useful if some components (e.g. some singular values) of this decomposition have
            /// been manually changed by the user.
            #[inline]
            pub fn recompose(self) -> MatrixMN<$t, R, C> {
                let nrows           = self.u.data.shape().0;
                let ncols           = self.vt.data.shape().1;
                let min_nrows_ncols = nrows.min(ncols);

                let mut res: MatrixMN<_, R, C> = Matrix::zeros_generic(nrows, ncols);

                {
                    let mut sres = res.generic_slice_mut((0, 0), (min_nrows_ncols, ncols));
                    sres.copy_from(&self.vt.rows_generic(0, min_nrows_ncols));

                    for i in 0 .. min_nrows_ncols.value() {
                        let eigval  = self.singular_values[i];
                        let mut row = sres.row_mut(i);
                        row *= eigval;
                    }
                }

                self.u * res
            }

            /// Computes the pseudo-inverse of the decomposed matrix.
            ///
            /// All singular value below epsilon will be set to zero instead of being inverted.
            #[inline]
            pub fn pseudo_inverse(&self, epsilon: $t) -> MatrixMN<$t, C, R> {
                let nrows           = self.u.data.shape().0;
                let ncols           = self.vt.data.shape().1;
                let min_nrows_ncols = nrows.min(ncols);

                let mut res: MatrixMN<_, C, R> = Matrix::zeros_generic(ncols, nrows);

                {
                    let mut sres = res.generic_slice_mut((0, 0), (min_nrows_ncols, nrows));
                    self.u.columns_generic(0, min_nrows_ncols).transpose_to(&mut sres);

                    for i in 0 .. min_nrows_ncols.value() {
                        let eigval  = self.singular_values[i];
                        let mut row = sres.row_mut(i);

                        if eigval.abs() > epsilon {
                            row /= eigval
                        }
                        else {
                            row.fill(0.0);
                        }
                    }
                }

                self.vt.tr_mul(&res)
            }

            /// The rank of the decomposed matrix.
            ///
            /// This is the number of singular values that are not too small (i.e. greater than
            /// the given `epsilon`).
            #[inline]
            pub fn rank(&self, epsilon: $t) -> usize {
                let mut i = 0;

                for e in self.singular_values.as_slice().iter() {
                    if e.abs() > epsilon {
                        i += 1;
                    }
                }

                i
            }

            // FIXME: add methods to retrieve the null-space and column-space? (Respectively
            // corresponding to the zero and non-zero singular values).
        }
    );
);

/*
macro_rules! svd_complex_impl(
    ($name: ident, $t: ty, $lapack_func: path) => (
        impl SVDScalar for Complex<$t> {
            fn compute<R: Dim, C: Dim, S>(mut m: Matrix<$t, R, C, S>) -> Option<SVD<$t, R, C, S::Alloc>>
            Option<(MatrixN<Complex<$t>, R, S::Alloc>,
                    VectorN<$t, DimMinimum<R, C>, S::Alloc>,
                    MatrixN<Complex<$t>, C, S::Alloc>)>
            where R: DimMin<C>,
                  S: ContiguousStorage<Complex<$t>, R, C>,
                  S::Alloc: OwnedAllocator<Complex<$t>, R, C, S> +
                            Allocator<Complex<$t>, R, R>         +
                            Allocator<Complex<$t>, C, C>         +
                            Allocator<$t, DimMinimum<R, C>> {
            let (nrows, ncols) = m.data.shape();

            if nrows.value() == 0 || ncols.value() == 0 {
                return None;
            }

            let jobu  = b'A';
            let jobvt = b'A';

            let lda = nrows.value() as i32;
            let min_nrows_ncols = nrows.min(ncols);


            let mut u  = unsafe { Matrix::new_uninitialized_generic(nrows, nrows) };
            let mut s  = unsafe { Matrix::new_uninitialized_generic(min_nrows_ncols, U1) };
            let mut vt = unsafe { Matrix::new_uninitialized_generic(ncols, ncols) };

            let ldu  = nrows.value();
            let ldvt = ncols.value();

            let mut work  = [ Complex::new(0.0, 0.0) ];
            let mut lwork = -1 as i32;
            let mut rwork = vec![ 0.0; (5 * min_nrows_ncols.value()) ];
            let mut info  = 0;

            $lapack_func(jobu, jobvt, nrows.value() as i32, ncols.value() as i32, m.as_mut_slice(),
                lda, s.as_mut_slice(), u.as_mut_slice(), ldu as i32, vt.as_mut_slice(),
                ldvt as i32, &mut work, lwork, &mut rwork, &mut info);
            lapack_check!(info);

            lwork = work[0].re as i32;
            let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];

            $lapack_func(jobu, jobvt, nrows.value() as i32, ncols.value() as i32, m.as_mut_slice(),
                lda, s.as_mut_slice(), u.as_mut_slice(), ldu as i32, vt.as_mut_slice(),
                ldvt as i32, &mut work, lwork, &mut rwork, &mut info);
            lapack_check!(info);

            Some((u, s, vt))
        }
    );
);
*/

svd_impl!(f32, interface::sgesdd);
svd_impl!(f64, interface::dgesdd);
// svd_complex_impl!(lapack_svd_complex_f32, f32, interface::cgesvd);
// svd_complex_impl!(lapack_svd_complex_f64, f64, interface::zgesvd);
