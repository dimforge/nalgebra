use crate::SparseEntryMut;
use crate::{CscBuilder, CscMatrix};
use nalgebra::{DMatrix, RealField};

/// Constructs an LU Factorization using a left-looking approach.
/// This means it will construct each column, starting from the leftmost one.
pub struct LeftLookingLUFactorization<T> {
    /// A single matrix stores both the lower and upper triangular components
    l_u: CscMatrix<T>,
}

impl<T: RealField + Copy> LeftLookingLUFactorization<T> {
    /// Returns the upper triangular part of this matrix.
    pub fn u(&self) -> CscMatrix<T> {
        self.l_u.upper_triangle()
    }

    /// Returns the joint L\U matrix. Here, `L` implicitly has 1 along the diagonal.
    pub fn lu(&self) -> &CscMatrix<T> {
        &self.l_u
    }

    /// Returns the lower triangular part of this matrix.
    pub fn l(&self) -> CscMatrix<T> {
        let mut l = self.l_u.lower_triangle();
        let n = self.l_u.nrows();
        for i in 0..n {
            if let SparseEntryMut::NonZero(v) = l.index_entry_mut(i, i) {
                *v = T::one();
            } else {
                unreachable!();
            }
        }
        l
    }

    /// Computes `x` in `LUx = b`, where `b` is a dense vector.
    pub fn solve(&self, b: &[T]) -> DMatrix<T> {
        let mut y = vec![T::zero(); b.len()];
        // Implementation: Solve two systems: Ly = b, then Ux = y.
        self.l_u.dense_lower_triangular_solve(b, &mut y, true);
        let mut out = y.clone();
        self.l_u.dense_upper_triangular_solve(&y, &mut out);

        DMatrix::from_vec(b.len(), 1, out)
    }

    /// Construct a new sparse LU factorization
    /// from a given CSC matrix.
    pub fn new(a: &CscMatrix<T>) -> Self {
        assert_eq!(a.nrows(), a.ncols());
        let n = a.nrows();

        // this initially starts as an identity  matrix.
        // but the ones are all implicit.
        let mut csc_builder = CscBuilder::new(n, n);

        let mut val_buf = vec![];
        let mut pat_buf = vec![];

        for (ci, col) in a.col_iter().enumerate() {
            let curr_mat = csc_builder.build();

            curr_mat
                .pattern()
                .sparse_lower_triangular_solve(col.row_indices(), &mut pat_buf);
            pat_buf.sort_unstable();
            val_buf.resize_with(pat_buf.len(), T::zero);

            // Solve the current column, assuming that it is lower triangular
            curr_mat.sparse_lower_triangular_solve_sorted(
                col.row_indices(),
                col.values(),
                &pat_buf,
                &mut val_buf,
                true,
            );

            // convert builder back to matrix
            csc_builder = CscBuilder::from_mat(curr_mat);
            assert!(csc_builder.revert_to_col(ci));
            let mut ukk = T::zero();
            for (row, val) in pat_buf.drain(..).zip(val_buf.drain(..)) {
                use std::cmp::Ordering;
                let val = match row.cmp(&ci) {
                    Ordering::Less => val,
                    Ordering::Equal => {
                        ukk = val;
                        val
                    }
                    Ordering::Greater => {
                        assert_ne!(ukk, T::zero());
                        val / ukk
                    }
                };
                assert_eq!(csc_builder.insert(row, ci, val), Ok(()));
            }
        }

        let l_u = csc_builder.build();
        Self { l_u }
    }
}
