#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixN, VectorN, U1};
use crate::dimension::Dim;
use crate::storage::Storage;
use simba::scalar::RealField;

/// UDU factorization.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "VectorN<N, D>: Serialize, MatrixN<N, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "VectorN<N, D>: Deserialize<'de>, MatrixN<N, D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct UDU<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>,
{
    /// The upper triangular matrix resulting from the factorization
    pub u: MatrixN<N, D>,
    /// The diagonal matrix resulting from the factorization
    pub d: VectorN<N, D>,
}

impl<N: RealField, D: Dim> Copy for UDU<N, D>
where
    DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>,
    VectorN<N, D>: Copy,
    MatrixN<N, D>: Copy,
{
}

impl<N: RealField, D: Dim> UDU<N, D>
where
    DefaultAllocator: Allocator<N, D> + Allocator<N, D, D>,
{
    /// Computes the UDU^T factorization.
    ///
    /// The input matrix `p` is assumed to be symmetric and this decomposition will only read
    /// the upper-triangular part of `p`.
    ///
    /// Ref.: "Optimal control and estimation-Dover Publications", Robert F. Stengel, (1994) page 360
    pub fn new(p: MatrixN<N, D>) -> Self {
        let n = p.ncols();
        let n_dim = p.data.shape().1;

        let mut d = VectorN::zeros_generic(n_dim, U1);
        let mut u = MatrixN::zeros_generic(n_dim, n_dim);

        d[n - 1] = p[(n - 1, n - 1)];
        u.column_mut(n - 1)
            .axpy(N::one() / d[n - 1], &p.column(n - 1), N::zero());

        for j in (0..n - 1).rev() {
            let mut d_j = d[j];
            for k in j + 1..n {
                d_j += d[k] * u[(j, k)].powi(2);
            }

            d[j] = p[(j, j)] - d_j;

            for i in (0..=j).rev() {
                let mut u_ij = u[(i, j)];
                for k in j + 1..n {
                    u_ij += d[k] * u[(j, k)] * u[(i, k)];
                }

                u[(i, j)] = (p[(i, j)] - u_ij) / d[j];
            }

            u[(j, j)] = N::one();
        }

        Self { u, d }
    }

    /// Returns the diagonal elements as a matrix
    pub fn d_matrix(&self) -> MatrixN<N, D> {
        MatrixN::from_diagonal(&self.d)
    }
}
