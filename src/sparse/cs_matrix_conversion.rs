use num::Zero;
use simba::scalar::ClosedAdd;

use crate::allocator::Allocator;
use crate::sparse::cs_utils;
use crate::sparse::{CsMatrix, CsStorage};
use crate::storage::Storage;
use crate::{DefaultAllocator, Dim, Dynamic, Matrix, MatrixMN, Scalar};

impl<'a, N: Scalar + Zero + ClosedAdd> CsMatrix<N> {
    /// Creates a column-compressed sparse matrix from a sparse matrix in triplet form.
    pub fn from_triplet(
        nrows: usize,
        ncols: usize,
        irows: &[usize],
        icols: &[usize],
        vals: &[N],
    ) -> Self {
        Self::from_triplet_generic(Dynamic::new(nrows), Dynamic::new(ncols), irows, icols, vals)
    }
}

impl<'a, N: Scalar + Zero + ClosedAdd, R: Dim, C: Dim> CsMatrix<N, R, C>
where
    DefaultAllocator: Allocator<usize, C> + Allocator<N, R>,
{
    /// Creates a column-compressed sparse matrix from a sparse matrix in triplet form.
    pub fn from_triplet_generic(
        nrows: R,
        ncols: C,
        irows: &[usize],
        icols: &[usize],
        vals: &[N],
    ) -> Self {
        assert!(vals.len() == irows.len());
        assert!(vals.len() == icols.len());

        let mut res = CsMatrix::new_uninitialized_generic(nrows, ncols, vals.len());
        let mut workspace = res.data.p.clone();

        // Column count.
        for j in icols.iter().cloned() {
            workspace[j] += 1;
        }

        let _ = cs_utils::cumsum(&mut workspace, &mut res.data.p);

        // Fill i and vals.
        for ((i, j), val) in irows
            .iter()
            .cloned()
            .zip(icols.iter().cloned())
            .zip(vals.iter().cloned())
        {
            let offset = workspace[j];
            res.data.i[offset] = i;
            res.data.vals[offset] = val;
            workspace[j] = offset + 1;
        }

        // Sort the result.
        res.sort();
        res.dedup();
        res
    }
}

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
                    res.data.vals[nz] = column[i].inlined_clone();
                    nz += 1;
                }
            }
        }

        res
    }
}
