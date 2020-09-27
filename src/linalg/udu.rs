#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixN};
use crate::dimension::Dim;
use simba::scalar::ComplexField;

/// UDU factorization
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct UDU<N: ComplexField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// The upper triangular matrix resulting from the factorization
    pub u: MatrixN<N, D>,
    /// The diagonal matrix resulting from the factorization
    pub d: MatrixN<N, D>,
}

impl<N: ComplexField, D: Dim> Copy for UDU<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    MatrixN<N, D>: Copy,
{
}

impl<N: ComplexField, D: Dim> UDU<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Computes the UDU^T factorization
    /// NOTE: The provided matrix MUST be symmetric, and no verification is done in this regard.
    /// Ref.: "Optimal control and estimation-Dover Publications", Robert F. Stengel, (1994) page 360
    pub fn new(p: MatrixN<N, D>) -> Self {
        let n = p.ncols();
        let n_as_dim = D::from_usize(n);

        let mut d = MatrixN::<N, D>::zeros_generic(n_as_dim, n_as_dim);
        let mut u = MatrixN::<N, D>::zeros_generic(n_as_dim, n_as_dim);

        d[(n - 1, n - 1)] = p[(n - 1, n - 1)];
        u[(n - 1, n - 1)] = N::one();

        for j in (0..n - 1).rev() {
            u[(j, n - 1)] = p[(j, n - 1)] / d[(n - 1, n - 1)];
        }

        for j in (0..n - 1).rev() {
            for k in j + 1..n {
                d[(j, j)] = d[(j, j)] + d[(k, k)] * u[(j, k)].powi(2);
            }

            d[(j, j)] = p[(j, j)] - d[(j, j)];

            for i in (0..=j).rev() {
                for k in j + 1..n {
                    u[(i, j)] = u[(i, j)] + d[(k, k)] * u[(j, k)] * u[(i, k)];
                }

                u[(i, j)] = p[(i, j)] - u[(i, j)];

                u[(i, j)] /= d[(j, j)];
            }

            u[(j, j)] = N::one();
        }

        Self { u, d }
    }
}
