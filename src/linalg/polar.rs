#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;

use crate::{DMatrix};
use crate::linalg::SVD;
use alga::general::{ComplexField};

/// Polar Decomposition of a general matrix.
// #[derive(Clone, Debug)]
pub struct Polar<N: ComplexField>
{
    /// The rotation matrix
    pub r: Option<DMatrix<N>>,
    /// The left hermitian matrix (A = PR)
    pub p_l: Option<DMatrix<N>>,
    /// The right hermitian matrix (A = RP)
    pub p_r: Option<DMatrix<N>>
}

// impl<N: ComplexField, R: DimMin<C>, C: DimName> Copy for Polar<N, R, C>
// where
//     DefaultAllocator: Allocator<N, R, C>
//     + Allocator<N, R, R>
//     + Allocator<N, C, C>,

//     DMatrix<N, R, C>: Copy,
//     DMatrix<N, R, R>: Copy,
//     DMatrix<N, C, C>: Copy,
// {
// }

impl<N: ComplexField> Polar<N>
where
{
    /// Computes the Polar Decomposition of the matrix using its SVD
    pub fn new(matrix: DMatrix<N>) -> Self {
        Self::try_new(
            matrix,
            N::RealField::default_epsilon(),
            0
        ).unwrap()
    }

    /// Attempts to compute the Polar Decomposition using the SVD of the matrix
    ///
    /// # Arguments
    ///
    /// * `eps`           − tolerance used to determine when a value converged to 0.
    /// * `max_niter`     − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(
        matrix: DMatrix<N>,
        eps: N::RealField,
        max_niter: usize
    ) -> Option<Self> {

        let svd_opt = SVD::try_new(matrix, true, true, eps, max_niter);

        if let Some(svd) = svd_opt {

            let r: Option<DMatrix<N>> =
                if let (Some(u), Some(v_t)) = (&svd.u, &svd.v_t) {
                    Some(u*v_t)
                } else {
                    None
                };

            let sigma: DMatrix<N> = DMatrix::from_diagonal(&svd.singular_values.map(|e| N::from_real(e)));;

            let p_r =
                if let Some(v_t)  = &svd.v_t {
                    Some(v_t.adjoint()*&sigma*v_t)
                } else {
                    None
                };

            let p_l =
                if let Some(u) = svd.u {
                    Some(u.adjoint()*&sigma*u)
                } else {
                    None
                };

            Some(Self {
                r,
                p_l,
                p_r,
            })
        } else {
            None
        }
    }
}

//impl<N: ComplexField, R: DimMin<C>, C: DimName, S: Storage<N, R, C>> Matrix<N, R, C, S>
//where
//    DefaultAllocator: Allocator<N, R, C>
//        + Allocator<N, R, R>
//        + Allocator<N, C, C>,
//{
//    /// Computes the Polar Decomposition of the matrix using its SVD
//    pub fn polar(self) -> Polar<N, R, C> {
//        Polar::new(self.into_owned())
//    }

//    /// Attempts to compute the Polar Decomposition using the SVD of the matrix
//    ///
//    /// # Arguments
//    ///
//    /// * `eps`       − tolerance used to determine when a value converged to 0.
//    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
//    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
//    /// continues indefinitely until convergence.
//    pub fn try_polar(
//        self,
//        eps: N::RealField,
//        max_niter: usize,
//    ) -> Option<Polar<N, R, C>> {
//        Polar::try_new(self.into_owned(), eps, max_niter)
//    }
//}
