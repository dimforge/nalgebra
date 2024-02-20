use nalgebra::{DMatrix, RealField};
use std::error::Error;
use std::ops::{Div, SubAssign};

/// LU decomposition of a square matrix.
#[derive(Debug)]
pub struct LuDecomposition<T> {
    l: DMatrix<T>,
    u: DMatrix<T>,
}

impl<T: RealField> LuDecomposition<T>
where
    T: Div<T, Output = T> + SubAssign + Copy,
{
    /// Computes the LU decomposition of a given square matrix.
    pub fn decompose(matrix: &DMatrix<T>) -> Result<Self, Box<dyn Error>> {
        if matrix.nrows() != matrix.ncols() {
            return Err("Matrix is not square".into());
        }

        let n = matrix.nrows();
        let mut l = DMatrix::zeros(n, n);
        let mut u = matrix.clone();

        // LU decomposition algorithm
        for i in 0..n {
            for j in 0..=i {
                let sum: T = (0..j)
                    .map(|k| l[(i, k)] * u[(k, j)])
                    .fold(T::zero(), |acc, x| acc + x);
                l[(i, j)] = u[(i, j)] - sum;
            }

            for j in (i + 1)..n {
                let sum: T = (0..i)
                    .map(|k| l[(i, k)] * u[(k, j)])
                    .fold(T::zero(), |acc, x| acc + x);
                if l[(i, i)].is_zero() {
                    return Err("Matrix is singular and cannot be decomposed".into());
                }
                u[(i, j)] = (u[(i, j)] - sum) / l[(i, i)];
            }
        }

        Ok(LuDecomposition { l, u })
    }

    /// Returns a reference to the lower triangular matrix `l`.
    pub fn l(&self) -> &DMatrix<T> {
        &self.l
    }

    /// Returns a reference to the upper triangular matrix `u`.
    pub fn u(&self) -> &DMatrix<T> {
        &self.u
    }

    /// Prints the L and U matrices.
    pub fn print_matrices(&self) {
        println!("L matrix:\n{}", self.l);
        println!("U matrix:\n{}", self.u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::assert_equal;
    use nalgebra::DMatrix;

    #[test]
    fn test_lu_decomposition() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[2.0, 3.0, 1.0, 4.0, 1.0, -1.0, -1.0, 3.0, 2.0]);
        let lu = LuDecomposition::decompose(&matrix).unwrap();

        // Print L and U matrices
        let l = lu.l();
        let u = lu.u();
        assert_equal(
            l.iter(),
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    2.0,
                    0.0,
                    0.0,
                    4.0,
                    -5.0,
                    0.0,
                    -1.0,
                    4.5,
                    -0.19999999999999973,
                ],
            )
            .iter(),
        );
        assert_equal(
            u.iter(),
            DMatrix::from_row_slice(3, 3, &[2.0, 1.5, 0.5, 4.0, 1.0, 0.6, -1.0, 3.0, 2.0]).iter(),
        );
    }

    #[test]
    fn test_lu_decomposition_singular() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]);
        let lu = LuDecomposition::decompose(&matrix);
        assert!(lu.is_err());
    }

    #[test]
    fn test_lu_decomposition_non_square() {
        let matrix = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lu = LuDecomposition::decompose(&matrix);
        assert!(lu.is_err());
    }

    #[test]
    fn test_lu_decomposition_identity() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let lu = LuDecomposition::decompose(&matrix).unwrap();

        // Print L and U matrices
        let l = lu.l();
        let u = lu.u();
        assert_equal(
            l.iter(),
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).iter(),
        );
        assert_equal(
            u.iter(),
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).iter(),
        );
    }
}
