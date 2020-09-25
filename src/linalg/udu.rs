#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixN};
use crate::dimension::DimName;
use simba::scalar::ComplexField;

/// UDU factorization
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct UDU<N: ComplexField, D: DimName>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// The upper triangular matrix resulting from the factorization
    pub u: MatrixN<N, D>,
    /// The diagonal matrix resulting from the factorization
    pub d: MatrixN<N, D>,
}

impl<N: ComplexField, D: DimName> Copy for UDU<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    MatrixN<N, D>: Copy,
{
}

impl<N: ComplexField, D: DimName> UDU<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Computes the UDU^T factorization as per NASA's "Navigation Filter Best Practices", NTRS document ID 20180003657
    /// section 7.2 page 64.
    /// NOTE: The provided matrix MUST be symmetric.
    pub fn new(matrix: MatrixN<N, D>) -> Self {
        let mut d = MatrixN::<N, D>::zeros();
        let mut u = MatrixN::<N, D>::zeros();

        let n = matrix.ncols();

        d[(n, n)] = matrix[(n, n)];
        u[(n, n)] = N::one();

        for j in (0..n - 1).rev() {
            u[(j, n)] = matrix[(j, n)] / d[(n, n)];
        }

        for j in (1..n).rev() {
            d[(j, j)] = matrix[(j, j)];
            for k in (j + 1..n).rev() {
                d[(j, j)] = d[(j, j)] + d[(k, k)] * u[(j, k)].powi(2);
            }

            u[(j, j)] = N::one();

            for i in (0..j - 1).rev() {
                u[(i, j)] = matrix[(i, j)];
                for k in j + 1..n {
                    u[(i, j)] = u[(i, j)] + d[(k, k)] * u[(i, k)] * u[(j, k)];
                }
                u[(i, j)] /= d[(j, j)];
            }
        }

        Self { u, d }
    }
}
