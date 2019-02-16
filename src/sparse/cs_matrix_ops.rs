use alga::general::{ClosedAdd, ClosedMul};
use num::{One, Zero};
use std::ops::{Add, Mul};

use allocator::Allocator;
use constraint::{AreMultipliable, DimEq, ShapeConstraint};
use sparse::{CsMatrix, CsStorage, CsStorageMut, CsVector};
use storage::StorageMut;
use {DefaultAllocator, Dim, Scalar, Vector, VectorN, U1};

impl<N: Scalar, R: Dim, C: Dim, S: CsStorage<N, R, C>> CsMatrix<N, R, C, S> {
    fn scatter<R2: Dim, C2: Dim>(
        &self,
        j: usize,
        beta: N,
        timestamps: &mut [usize],
        timestamp: usize,
        workspace: &mut [N],
        mut nz: usize,
        res: &mut CsMatrix<N, R2, C2>,
    ) -> usize
    where
        N: ClosedAdd + ClosedMul,
        DefaultAllocator: Allocator<usize, C2>,
    {
        for (i, val) in self.data.column_entries(j) {
            if timestamps[i] < timestamp {
                timestamps[i] = timestamp;
                res.data.i[nz] = i;
                nz += 1;
                workspace[i] = val * beta;
            } else {
                workspace[i] += val * beta;
            }
        }

        nz
    }
}

/*
impl<N: Scalar, R, S> CsVector<N, R, S> {
    pub fn axpy(&mut self, alpha: N, x: CsVector<N, R, S>, beta: N) {
        // First, compute the number of non-zero entries.
        let mut nnzero = 0;

        // Allocate a size large enough.
        self.data.set_column_len(0, nnzero);

        // Fill with the axpy.
        let mut i = self.len();
        let mut j = x.len();
        let mut k = nnzero - 1;
        let mut rid1 = self.data.row_index(0, i - 1);
        let mut rid2 = x.data.row_index(0, j - 1);

        while k > 0 {
            if rid1 == rid2 {
                self.data.set_row_index(0, k, rid1);
                self[k] = alpha * x[j] + beta * self[k];
                i -= 1;
                j -= 1;
            } else if rid1 < rid2 {
                self.data.set_row_index(0, k, rid1);
                self[k] = beta * self[i];
                i -= 1;
            } else {
                self.data.set_row_index(0, k, rid2);
                self[k] = alpha * x[j];
                j -= 1;
            }

            k -= 1;
        }
    }
}
*/

impl<N: Scalar + Zero + ClosedAdd + ClosedMul, D: Dim, S: StorageMut<N, D>> Vector<N, D, S> {
    /// Perform a sparse axpy operation: `self = alpha * x + beta * self` operation.
    pub fn axpy_cs<D2: Dim, S2>(&mut self, alpha: N, x: &CsVector<N, D2, S2>, beta: N)
    where
        S2: CsStorage<N, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        if beta.is_zero() {
            for i in 0..x.len() {
                unsafe {
                    let k = x.data.row_index_unchecked(i);
                    let y = self.vget_unchecked_mut(k);
                    *y = alpha * *x.data.get_value_unchecked(i);
                }
            }
        } else {
            // Needed to be sure even components not present on `x` are multiplied.
            *self *= beta;

            for i in 0..x.len() {
                unsafe {
                    let k = x.data.row_index_unchecked(i);
                    let y = self.vget_unchecked_mut(k);
                    *y += alpha * *x.data.get_value_unchecked(i);
                }
            }
        }
    }

    /*
    pub fn gemv_sparse<R2: Dim, C2: Dim, S2>(&mut self, alpha: N, a: &CsMatrix<N, R2, C2, S2>, x: &DVector<N>, beta: N)
        where
            S2: CsStorage<N, R2, C2> {
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy_sparse(alpha * val, &col2, beta);
    
        for j in 1..ncols2 {
            let col2 = a.column(j);
            let val = unsafe { *x.vget_unchecked(j) };
    
            self.axpy_sparse(alpha * val, &col2, N::one());
        }
    }
    */
}

impl<'a, 'b, N, R1, R2, C1, C2, S1, S2> Mul<&'b CsMatrix<N, R2, C2, S2>>
    for &'a CsMatrix<N, R1, C1, S1>
where
    N: Scalar + ClosedAdd + ClosedMul + Zero,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    S1: CsStorage<N, R1, C1>,
    S2: CsStorage<N, R2, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
    DefaultAllocator: Allocator<usize, C2> + Allocator<usize, R1> + Allocator<N, R1>,
{
    type Output = CsMatrix<N, R1, C2>;

    fn mul(self, rhs: &'b CsMatrix<N, R2, C2, S2>) -> CsMatrix<N, R1, C2> {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();
        assert_eq!(
            ncols1.value(),
            nrows2.value(),
            "Mismatched dimensions for matrix multiplication."
        );

        let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        let mut workspace = VectorN::<N, R1>::zeros_generic(nrows1, U1);
        let mut nz = 0;

        for j in 0..ncols2.value() {
            res.data.p[j] = nz;
            let new_size_bound = nz + nrows1.value();
            res.data.i.resize(new_size_bound, 0);
            res.data.vals.resize(new_size_bound, N::zero());

            for (i, beta) in rhs.data.column_entries(j) {
                for (k, val) in self.data.column_entries(i) {
                    workspace[k] += val * beta;
                }
            }

            for (i, val) in workspace.as_mut_slice().iter_mut().enumerate() {
                if !val.is_zero() {
                    res.data.i[nz] = i;
                    res.data.vals[nz] = *val;
                    *val = N::zero();
                    nz += 1;
                }
            }
        }

        // NOTE: the following has a lower complexity, but is slower in many cases, likely because
        // of branching inside of the inner loop.
        //
        // let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        // let mut timestamps = VectorN::zeros_generic(nrows1, U1);
        // let mut workspace = unsafe { VectorN::new_uninitialized_generic(nrows1, U1) };
        // let mut nz = 0;
        //
        // for j in 0..ncols2.value() {
        //     res.data.p[j] = nz;
        //     let new_size_bound = nz + nrows1.value();
        //     res.data.i.resize(new_size_bound, 0);
        //     res.data.vals.resize(new_size_bound, N::zero());
        //
        //     for (i, val) in rhs.data.column_entries(j) {
        //         nz = self.scatter(
        //             i,
        //             val,
        //             timestamps.as_mut_slice(),
        //             j + 1,
        //             workspace.as_mut_slice(),
        //             nz,
        //             &mut res,
        //         );
        //     }
        //
        //     // Keep the output sorted.
        //     let range = res.data.p[j]..nz;
        //     res.data.i[range.clone()].sort();
        //
        //     for p in range {
        //         res.data.vals[p] = workspace[res.data.i[p]]
        //     }
        // }

        res.data.i.truncate(nz);
        res.data.i.shrink_to_fit();
        res.data.vals.truncate(nz);
        res.data.vals.shrink_to_fit();
        res
    }
}

impl<'a, 'b, N, R1, R2, C1, C2, S1, S2> Add<&'b CsMatrix<N, R2, C2, S2>>
    for &'a CsMatrix<N, R1, C1, S1>
where
    N: Scalar + ClosedAdd + ClosedMul + One,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    S1: CsStorage<N, R1, C1>,
    S2: CsStorage<N, R2, C2>,
    ShapeConstraint: DimEq<R1, R2> + DimEq<C1, C2>,
    DefaultAllocator: Allocator<usize, C2> + Allocator<usize, R1> + Allocator<N, R1>,
{
    type Output = CsMatrix<N, R1, C2>;

    fn add(self, rhs: &'b CsMatrix<N, R2, C2, S2>) -> CsMatrix<N, R1, C2> {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();
        assert_eq!(
            (nrows1.value(), ncols1.value()),
            (nrows2.value(), ncols2.value()),
            "Mismatched dimensions for matrix sum."
        );

        let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        let mut timestamps = VectorN::zeros_generic(nrows1, U1);
        let mut workspace = unsafe { VectorN::new_uninitialized_generic(nrows1, U1) };
        let mut nz = 0;

        for j in 0..ncols2.value() {
            res.data.p[j] = nz;

            nz = self.scatter(
                j,
                N::one(),
                timestamps.as_mut_slice(),
                j + 1,
                workspace.as_mut_slice(),
                nz,
                &mut res,
            );

            nz = rhs.scatter(
                j,
                N::one(),
                timestamps.as_mut_slice(),
                j + 1,
                workspace.as_mut_slice(),
                nz,
                &mut res,
            );

            // Keep the output sorted.
            let range = res.data.p[j]..nz;
            res.data.i[range.clone()].sort();

            for p in range {
                res.data.vals[p] = workspace[res.data.i[p]]
            }
        }

        res.data.i.truncate(nz);
        res.data.i.shrink_to_fit();
        res.data.vals.truncate(nz);
        res.data.vals.shrink_to_fit();
        res
    }
}

impl<'a, 'b, N, R, C, S> Mul<N> for CsMatrix<N, R, C, S>
where
    N: Scalar + ClosedAdd + ClosedMul + Zero,
    R: Dim,
    C: Dim,
    S: CsStorageMut<N, R, C>,
{
    type Output = Self;

    fn mul(mut self, rhs: N) -> Self {
        for e in self.values_mut() {
            *e *= rhs
        }

        self
    }
}
