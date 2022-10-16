#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use simba::scalar::RealField;

use crate::ComplexHelper;
use na::dimension::{Const, Dim, DimName};
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar, allocator::Allocator};

use lapack;

/// Eigendecomposition of a real square matrix with real or complex eigenvalues.
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
         OVector<T, D>: Serialize,
         OMatrix<T, D, D>: Deserialize<'de>")
    )
)]
#[derive(Clone, Debug)]
pub struct Eigen<T: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// The real parts of eigenvalues of the decomposed matrix.
    pub eigenvalues_re: OVector<T, D>,
    /// The imaginary parts of the eigenvalues of the decomposed matrix.
    pub eigenvalues_im: OVector<T, D>,
    /// The (right) eigenvectors of the decomposed matrix.
    pub eigenvectors: Option<OMatrix<T, D, D>>,
    /// The left eigenvectors of the decomposed matrix.
    pub left_eigenvectors: Option<OMatrix<T, D, D>>,
}

impl<T: Scalar + Copy, D: DimName> Copy for Eigen<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
    OVector<T, D>: Copy,
    OMatrix<T, D, D>: Copy,
{
}

impl<T: EigenScalar + RealField, D: DimName> Eigen<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Computes the eigenvalues and eigenvectors of the square matrix `m`.
    ///
    /// If `eigenvectors` is `false` then, the eigenvectors are not computed explicitly.
    pub fn new(
        mut m: OMatrix<T, D, D>,
        left_eigenvectors: bool,
        eigenvectors: bool,
    ) -> Option<Eigen<T, D>> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let ljob = if left_eigenvectors { b'V' } else { b'N' };
        let rjob = if eigenvectors { b'V' } else { b'N' };

        let (nrows, ncols) = m.shape_generic();
        let n = nrows.value();

        let lda = n as i32;

        // TODO: avoid the initialization?
        let mut wr = Matrix::zeros_generic(nrows, Const::<1>);
        // TODO: Tap into the workspace.
        let mut wi = Matrix::zeros_generic(nrows, Const::<1>);

        let mut info = 0;
        let mut placeholder1 = [T::zero()];
        let mut placeholder2 = [T::zero()];

        let lwork = T::xgeev_work_size(
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

        let mut work = vec![T::zero(); lwork as usize];
        let mut vl = if left_eigenvectors {
            Some(Matrix::zeros_generic(nrows, ncols))
        } else {
            None
        };
        let mut vr = if eigenvectors {
            Some(Matrix::zeros_generic(nrows, ncols))
        } else {
            None
        };

        let vl_ref = vl
            .as_mut()
            .map(|m| m.as_mut_slice())
            .unwrap_or(&mut placeholder1);
        let vr_ref = vr
            .as_mut()
            .map(|m| m.as_mut_slice())
            .unwrap_or(&mut placeholder2);

        T::xgeev(
            ljob,
            rjob,
            n as i32,
            m.as_mut_slice(),
            lda,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            vl_ref,
            if left_eigenvectors { n as i32 } else { 1 },
            vr_ref,
            if eigenvectors { n as i32 } else { 1 },
            &mut work,
            lwork,
            &mut info,
        );
        lapack_check!(info);

        Some(Self {
            eigenvalues_re: wr,
            eigenvalues_im: wi,
            left_eigenvectors: vl,
            eigenvectors: vr
        })
    }

    /// Returns `true` if all the eigenvalues are real.
    pub fn eigenvalues_are_real(&self) -> bool {
        self.eigenvalues_im.iter().all(|e| e.is_zero())
    }

    /// The determinant of the decomposed matrix.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> Complex<T> {
        let mut det: Complex<T> = na::one();
        for (re, im) in self.eigenvalues_re.iter().zip(self.eigenvalues_im.iter()) {
            det *= Complex::new(re.clone(), im.clone());
        }

        det
    }

    /// Returns a tuple of vectors. The elements of the tuple are the complex eigenvalues, complex left eigenvectors and complex right eigenvectors respectively. 
    /// The elements appear as conjugate pairs within each vector, with the positive of the pair always being first.
    pub fn get_complex_elements(&self) -> (Option<Vec<Complex<T>>>, Option<Vec<OVector<Complex<T>, D>>>, Option<Vec<OVector<Complex<T>, D>>>) where DefaultAllocator: Allocator<Complex<T>, D> {
        match !self.eigenvalues_are_real() {
            true => (None, None, None),
            false => {
                let number_of_elements = self.eigenvalues_re.nrows();
                let number_of_complex_entries = self.eigenvalues_im.iter().fold(0, |acc, e| if !e.is_zero() {acc + 1} else {acc});
                let mut eigenvalues = Vec::<Complex<T>>::with_capacity(2*number_of_complex_entries);
                let mut eigenvectors = match self.eigenvectors.is_some() {
                    true => Some(Vec::<OVector<Complex<T>, D>>::with_capacity(2*number_of_complex_entries)),
                    false => None
                };
                let mut left_eigenvectors = match self.left_eigenvectors.is_some() {
                    true => Some(Vec::<OVector<Complex<T>, D>>::with_capacity(2*number_of_complex_entries)),
                    false => None
                };

                for mut c in 0..number_of_elements {
                    if self.eigenvalues_im[c] != T::zero() {
                        //Complex conjugate pairs of eigenvalues appear consecutively with the eigenvalue having the positive imaginary part first.
                        eigenvalues.push(Complex::<T>::new(self.eigenvalues_re[c].clone(), self.eigenvalues_im[c].clone()));
                        eigenvalues.push(Complex::<T>::new(self.eigenvalues_re[c].clone(), -self.eigenvalues_im[c].clone()));

                        if eigenvectors.is_some() {
                            let mut vec = OVector::<Complex<T>, D>::zeros();
                            let mut vec_conj = OVector::<Complex<T>, D>::zeros();

                            for r in 0..number_of_elements {
                                vec[r] = Complex::<T>::new((&self.eigenvectors.as_ref()).unwrap()[(r,c)].clone(),(&self.eigenvectors.as_ref()).unwrap()[(r,c+1)].clone());
                                vec_conj[r] = Complex::<T>::new((&self.eigenvectors.as_ref()).unwrap()[(r,c)].clone(),-(&self.eigenvectors.as_ref()).unwrap()[(r,c+1)].clone());
                            }
    
                            eigenvectors.as_mut().unwrap().push(vec);
                            eigenvectors.as_mut().unwrap().push(vec_conj);
                        }

                        if left_eigenvectors.is_some() {
                            let mut vec = OVector::<Complex<T>, D>::zeros();
                            let mut vec_conj = OVector::<Complex<T>, D>::zeros();

                            for r in 0..number_of_elements {
                                vec[r] = Complex::<T>::new((&self.left_eigenvectors.as_ref()).unwrap()[(r,c)].clone(),(&self.left_eigenvectors.as_ref()).unwrap()[(r,c+1)].clone());
                                vec_conj[r] = Complex::<T>::new((&self.left_eigenvectors.as_ref()).unwrap()[(r,c)].clone(),-(&self.left_eigenvectors.as_ref()).unwrap()[(r,c+1)].clone());
                            }
    
                            left_eigenvectors.as_mut().unwrap().push(vec);
                            left_eigenvectors.as_mut().unwrap().push(vec_conj);
                        }
                        //skip next entry
                        c += 1;
                    }
                }
                (Some(eigenvalues), left_eigenvectors, eigenvectors)
            }
        }


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

//// TODO: decomposition of complex matrix and matrices with complex eigenvalues.
// eigensystem_complex_impl!(f32, lapack::cgeev);
// eigensystem_complex_impl!(f64, lapack::zgeev);
