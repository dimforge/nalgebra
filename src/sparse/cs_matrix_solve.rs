use allocator::Allocator;
use constraint::{SameNumberOfRows, ShapeConstraint};
use sparse::{CsMatrix, CsStorage, CsVector};
use storage::{Storage, StorageMut};
use {DefaultAllocator, Dim, Matrix, MatrixMN, Real, VectorN, U1};

impl<N: Real, D: Dim, S: CsStorage<N, D, D>> CsMatrix<N, D, D, S> {
    /// Solve a lower-triangular system with a dense right-hand-side.
    pub fn solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let mut b = b.clone_owned();
        if self.solve_lower_triangular_mut(&mut b) {
            Some(b)
        } else {
            None
        }
    }

    /// Solve a lower-triangular system with `self` transposed and a dense right-hand-side.
    pub fn tr_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let mut b = b.clone_owned();
        if self.tr_solve_lower_triangular_mut(&mut b) {
            Some(b)
        } else {
            None
        }
    }

    /// Solve in-place a lower-triangular system with a dense right-hand-side.
    pub fn solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let (nrows, ncols) = self.data.shape();
        assert_eq!(nrows.value(), ncols.value(), "The matrix must be square.");
        assert_eq!(nrows.value(), b.len(), "Mismatched matrix dimensions.");

        for j2 in 0..b.ncols() {
            let mut b = b.column_mut(j2);

            for j in 0..ncols.value() {
                let mut column = self.data.column_entries(j);
                let mut diag_found = false;

                while let Some((i, val)) = column.next() {
                    if i == j {
                        if val.is_zero() {
                            return false;
                        }

                        b[j] /= val;
                        diag_found = true;
                        break;
                    }
                }

                if !diag_found {
                    return false;
                }

                for (i, val) in column {
                    b[i] -= b[j] * val;
                }
            }
        }

        true
    }

    /// Solve a lower-triangular system with `self` transposed and a dense right-hand-side.
    pub fn tr_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let (nrows, ncols) = self.data.shape();
        assert_eq!(nrows.value(), ncols.value(), "The matrix must be square.");
        assert_eq!(nrows.value(), b.len(), "Mismatched matrix dimensions.");

        for j2 in 0..b.ncols() {
            let mut b = b.column_mut(j2);

            for j in (0..ncols.value()).rev() {
                let mut column = self.data.column_entries(j);
                let mut diag = None;

                while let Some((i, val)) = column.next() {
                    if i == j {
                        if val.is_zero() {
                            return false;
                        }

                        diag = Some(val);
                        break;
                    }
                }

                if let Some(diag) = diag {
                    for (i, val) in column {
                        b[j] -= val * b[i];
                    }

                    b[j] /= diag;
                } else {
                    return false;
                }
            }
        }

        true
    }

    /// Solve a lower-triangular system with a sparse right-hand-side.
    pub fn solve_lower_triangular_cs<D2: Dim, S2>(
        &self,
        b: &CsVector<N, D2, S2>,
    ) -> Option<CsVector<N, D2>>
    where
        S2: CsStorage<N, D2>,
        DefaultAllocator: Allocator<bool, D> + Allocator<N, D2> + Allocator<usize, D2>,
        ShapeConstraint: SameNumberOfRows<D, D2>,
    {
        let mut reach = Vec::new();
        // We don't compute a postordered reach here because it will be sorted after anyway.
        self.lower_triangular_reach(b, &mut reach);
        // We sort the reach so the result matrix has sorted indices.
        reach.sort();
        let mut workspace = unsafe { VectorN::new_uninitialized_generic(b.data.shape().0, U1) };

        for i in reach.iter().cloned() {
            workspace[i] = N::zero();
        }

        for (i, val) in b.data.column_entries(0) {
            workspace[i] = val;
        }

        for j in reach.iter().cloned() {
            let mut column = self.data.column_entries(j);
            let mut diag_found = false;

            while let Some((i, val)) = column.next() {
                if i == j {
                    if val.is_zero() {
                        break;
                    }

                    workspace[j] /= val;
                    diag_found = true;
                    break;
                }
            }

            if !diag_found {
                return None;
            }

            for (i, val) in column {
                workspace[i] -= workspace[j] * val;
            }
        }

        // Copy the result into a sparse vector.
        let mut result = CsVector::new_uninitialized_generic(b.data.shape().0, U1, reach.len());

        for (i, val) in reach.iter().zip(result.data.vals.iter_mut()) {
            *val = workspace[*i];
        }

        result.data.i = reach;
        Some(result)
    }

    /*
    // Computes the reachable, post-ordered, nodes from `b`.
    fn lower_triangular_reach_postordered<D2: Dim, S2>(
        &self,
        b: &CsVector<N, D2, S2>,
        xi: &mut Vec<usize>,
    ) where
        S2: CsStorage<N, D2>,
        DefaultAllocator: Allocator<bool, D>,
    {
        let mut visited = VectorN::repeat_generic(self.data.shape().1, U1, false);
        let mut stack = Vec::new();

        for i in b.data.column_range(0) {
            let row_index = b.data.row_index(i);

            if !visited[row_index] {
                let rng = self.data.column_range(row_index);
                stack.push((row_index, rng));
                self.lower_triangular_dfs(visited.as_mut_slice(), &mut stack, xi);
            }
        }
    }

    fn lower_triangular_dfs(
        &self,
        visited: &mut [bool],
        stack: &mut Vec<(usize, Range<usize>)>,
        xi: &mut Vec<usize>,
    )
    {
        'recursion: while let Some((j, rng)) = stack.pop() {
            visited[j] = true;

            for i in rng.clone() {
                let row_id = self.data.row_index(i);
                if row_id > j && !visited[row_id] {
                    stack.push((j, (i + 1)..rng.end));
                    stack.push((row_id, self.data.column_range(row_id)));
                    continue 'recursion;
                }
            }

            xi.push(j)
        }
    }
    */

    // Computes the nodes reachable from `b` in an arbitrary order.
    fn lower_triangular_reach<D2: Dim, S2>(&self, b: &CsVector<N, D2, S2>, xi: &mut Vec<usize>)
    where
        S2: CsStorage<N, D2>,
        DefaultAllocator: Allocator<bool, D>,
    {
        let mut visited = VectorN::repeat_generic(self.data.shape().1, U1, false);
        let mut stack = Vec::new();

        for irow in b.data.column_row_indices(0) {
            self.lower_triangular_bfs(irow, visited.as_mut_slice(), &mut stack, xi);
        }
    }

    fn lower_triangular_bfs(
        &self,
        start: usize,
        visited: &mut [bool],
        stack: &mut Vec<usize>,
        xi: &mut Vec<usize>,
    )
    {
        if !visited[start] {
            stack.clear();
            stack.push(start);
            xi.push(start);
            visited[start] = true;

            while let Some(j) = stack.pop() {
                for irow in self.data.column_row_indices(j) {
                    if irow > j && !visited[irow] {
                        stack.push(irow);
                        xi.push(irow);
                        visited[irow] = true;
                    }
                }
            }
        }
    }
}
