use num::Zero;
use num_complex::Complex;

use na::allocator::Allocator;
use na::dimension::{DimDiff, DimSub, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, MatrixN, Scalar, VectorN};
use crate::ComplexHelper;

use lapack;

/// The Hessenberg decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Serialize,
         VectorN<N, DimDiff<D, U1>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Deserialize<'de>,
         VectorN<N, DimDiff<D, U1>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Hessenberg<N: Scalar, D: DimSub<U1>>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    h: MatrixN<N, D>,
    tau: VectorN<N, DimDiff<D, U1>>,
}

impl<N: Scalar, D: DimSub<U1>> Copy for Hessenberg<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
    MatrixN<N, D>: Copy,
    VectorN<N, DimDiff<D, U1>>: Copy,
{}

impl<N: HessenbergScalar + Zero, D: DimSub<U1>> Hessenberg<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the hessenberg decomposition of the matrix `m`.
    pub fn new(mut m: MatrixN<N, D>) -> Self {
        let nrows = m.data.shape().0;
        let n = nrows.value() as i32;

        assert!(
            m.is_square(),
            "Unable to compute the hessenberg decomposition of a non-square matrix."
        );
        assert!(
            !m.is_empty(),
            "Unable to compute the hessenberg decomposition of an empty matrix."
        );

        let mut tau = unsafe { Matrix::new_uninitialized_generic(nrows.sub(U1), U1) };

        let mut info = 0;
        let lwork =
            N::xgehrd_work_size(n, 1, n, m.as_mut_slice(), n, tau.as_mut_slice(), &mut info);
        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        lapack_panic!(info);

        N::xgehrd(
            n,
            1,
            n,
            m.as_mut_slice(),
            n,
            tau.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );
        lapack_panic!(info);

        Self { h: m, tau: tau }
    }

    /// Computes the hessenberg matrix of this decomposition.
    #[inline]
    pub fn h(&self) -> MatrixN<N, D> {
        let mut h = self.h.clone_owned();
        h.fill_lower_triangle(N::zero(), 2);

        h
    }
}

impl<N: HessenbergReal + Zero, D: DimSub<U1>> Hessenberg<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the matrices `(Q, H)` of this decomposition.
    #[inline]
    pub fn unpack(self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        (self.q(), self.h())
    }

    /// Computes the unitary matrix `Q` of this decomposition.
    #[inline]
    pub fn q(&self) -> MatrixN<N, D> {
        let n = self.h.nrows() as i32;
        let mut q = self.h.clone_owned();
        let mut info = 0;

        let lwork =
            N::xorghr_work_size(n, 1, n, q.as_mut_slice(), n, self.tau.as_slice(), &mut info);
        let mut work = vec![N::zero(); lwork as usize];

        N::xorghr(
            n,
            1,
            n,
            q.as_mut_slice(),
            n,
            self.tau.as_slice(),
            &mut work,
            lwork,
            &mut info,
        );

        q
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
pub trait HessenbergScalar: Scalar {
    fn xgehrd(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    fn xgehrd_work_size(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        info: &mut i32,
    ) -> i32;
}

/// Trait implemented by scalars for which Lapack implements the hessenberg decomposition.
pub trait HessenbergReal: HessenbergScalar {
    #[allow(missing_docs)]
    fn xorghr(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xorghr_work_size(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        info: &mut i32,
    ) -> i32;
}

macro_rules! hessenberg_scalar_impl(
    ($N: ty, $xgehrd: path) => (
        impl HessenbergScalar for $N {
            #[inline]
            fn xgehrd(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                      tau: &mut [Self], work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xgehrd(n, ilo, ihi, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xgehrd_work_size(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                                tau: &mut [Self], info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgehrd(n, ilo, ihi, a, lda, tau, &mut work, lwork, info) };
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

macro_rules! hessenberg_real_impl(
    ($N: ty, $xorghr: path) => (
        impl HessenbergReal for $N {
            #[inline]
            fn xorghr(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32, tau: &[Self],
                      work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xorghr(n, ilo, ihi, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xorghr_work_size(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                                tau: &[Self], info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xorghr(n, ilo, ihi, a, lda, tau, &mut work, lwork, info) };
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

hessenberg_scalar_impl!(f32, lapack::sgehrd);
hessenberg_scalar_impl!(f64, lapack::dgehrd);
hessenberg_scalar_impl!(Complex<f32>, lapack::cgehrd);
hessenberg_scalar_impl!(Complex<f64>, lapack::zgehrd);

hessenberg_real_impl!(f32, lapack::sorghr);
hessenberg_real_impl!(f64, lapack::dorghr);
