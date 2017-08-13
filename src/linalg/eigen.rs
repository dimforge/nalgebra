use std::fmt::Display;
use std::cmp;
use num_complex::Complex;
use alga::general::Real;
use std::ops::MulAssign;

use core::{DefaultAllocator, SquareMatrix, VectorN, MatrixN, Hessenberg, Unit, Vector2, Vector3};
use core::dimension::{Dim, DimSub, DimDiff, Dynamic, U1, U2, U3};
use core::storage::Storage;
use constraint::{ShapeConstraint, DimEq};
use allocator::Allocator;

use linalg::householder;
use linalg::RealSchur;
use geometry::{Reflection, UnitComplex};


/// The eigendecomposition of a matrix with real eigenvalues.
#[derive(Clone, Debug)]
pub struct RealEigen<N: Real, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + 
                            Allocator<N, D> {
    pub eigenvectors: MatrixN<N, D>,
    pub eigenvalues:  VectorN<N, D>
}


impl<N: Real, D: Dim> Copy for RealEigen<N, D>
    where DefaultAllocator: Allocator<N, D, D> + 
                            Allocator<N, D>,
          MatrixN<N, D>: Copy,
          VectorN<N, D>: Copy { }

impl<N: Real, D: Dim> RealEigen<N, D>
    where D: DimSub<U1>,                        // For Hessenberg.
          ShapeConstraint: DimEq<Dynamic, DimDiff<D, U1>>, // For Hessenberg.
          DefaultAllocator: Allocator<N, D, DimDiff<D, U1>> + // For Hessenberg.
                            Allocator<N, DimDiff<D, U1>>    + // For Hessenberg.
                            Allocator<N, D, D>              + 
                            Allocator<N, D>,
          // XXX: for debug
          DefaultAllocator: Allocator<usize, D, D>,
          MatrixN<N, D>: Display {

    /// Computes the eigendecomposition of a diagonalizable matrix with real eigenvalues.
    pub fn new(m: MatrixN<N, D>) -> Option<RealEigen<N, D>> {
        assert!(m.is_square(), "Unable to compute the eigendecomposition of a non-square matrix.");

        let dim = m.nrows();
        let (mut eigenvectors, mut eigenvalues) = RealSchur::new(m, 0).unwrap().unpack();

        println!("Schur eigenvalues: {}", eigenvalues);

        // Check that the eigenvalues are all real.
        for i in 0 .. dim - 1 {
            if !eigenvalues[(i + 1, i)].is_zero() {
                return None;
            }
        }

        for j in 1 .. dim {
            for i in 0 .. j {
                let diff = eigenvalues[(i, i)] - eigenvalues[(j, j)];

                if diff.is_zero() && !eigenvalues[(i, j)].is_zero() {
                    return None;
                }

                let z = -eigenvalues[(i, j)] / diff;

                for k in j + 1 .. dim {
                    eigenvalues[(i, k)] -= z * eigenvalues[(j, k)];
                }

                for k in 0 .. dim {
                    eigenvectors[(k, j)] += z * eigenvectors[(k, i)];
                }
            }
        }

        // Normalize the eigenvector basis.
        for i in 0 .. dim {
            let _ = eigenvectors.column_mut(i).normalize_mut();
        }

        Some(RealEigen {
            eigenvectors: eigenvectors,
            eigenvalues:  eigenvalues.diagonal()
        })
    }
}
