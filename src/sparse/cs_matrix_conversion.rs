use alga::general::{ClosedAdd, ClosedMul};
use num::{One, Zero};
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Range};
use std::slice;

use allocator::Allocator;
use constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use sparse::{CsMatrix, CsStorage, CsVector};
use storage::{Storage, StorageMut};
use {DefaultAllocator, Dim, Matrix, MatrixMN, Real, Scalar, Vector, VectorN, U1};

impl<'a, N: Scalar + Zero, R: Dim, C: Dim, S> From<CsMatrix<N, R, C, S>> for MatrixMN<N, R, C>
where
    S: CsStorage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    fn from(m: CsMatrix<N, R, C, S>) -> Self {
        let (nrows, ncols) = m.data.shape();
        let mut res = MatrixMN::zeros_generic(nrows, ncols);

        for j in 0..ncols.value() {
            for (i, val) in m.data.column_entries(j) {
                res[(i, j)] = val;
            }
        }

        res
    }
}

impl<'a, N: Scalar + Zero, R: Dim, C: Dim, S> From<Matrix<N, R, C, S>> for CsMatrix<N, R, C>
where
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C> + Allocator<usize, C>,
{
    fn from(m: Matrix<N, R, C, S>) -> Self {
        let (nrows, ncols) = m.data.shape();
        let len = m.iter().filter(|e| !e.is_zero()).count();
        let mut res = CsMatrix::new_uninitialized_generic(nrows, ncols, len);
        let mut nz = 0;

        for j in 0..ncols.value() {
            let column = m.column(j);
            res.data.p[j] = nz;

            for i in 0..nrows.value() {
                if !column[i].is_zero() {
                    res.data.i[nz] = i;
                    res.data.vals[nz] = column[i];
                    nz += 1;
                }
            }
        }

        res
    }
}
