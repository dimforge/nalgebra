// TODO: Remove this allowance
#![allow(missing_docs)]

use crate::pattern::SparsityPattern;
use crate::csc::CscMatrix;
use core::{mem, iter};
use nalgebra::{U1, VectorN, Dynamic, Scalar, RealField};
use num_traits::Zero;
use std::sync::Arc;
use std::ops::Add;

pub struct CscSymbolicCholesky {
    // Pattern of the original matrix that was decomposed
    m_pattern: Arc<SparsityPattern>,
    l_pattern: SparsityPattern,
    // u in this context is L^T, so that M = L L^T
    u_pattern: SparsityPattern
}

impl CscSymbolicCholesky {
    pub fn factor(pattern: &Arc<SparsityPattern>) -> Self {
        assert_eq!(pattern.major_dim(), pattern.minor_dim(),
            "Major and minor dimensions must be the same (square matrix).");

        // TODO: Temporary stopgap solution to make things work until we can refactor
        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        struct DummyVal;
        impl Zero for DummyVal {
            fn zero() -> Self {
                DummyVal
            }

            fn is_zero(&self) -> bool {
                true
            }
        }

        impl Add<DummyVal> for DummyVal {
            type Output = Self;

            fn add(self, rhs: DummyVal) -> Self::Output {
                rhs
            }
        }

        let dummy_vals = vec![DummyVal; pattern.nnz()];
        let dummy_csc = CscMatrix::try_from_pattern_and_values(Arc::clone(pattern), dummy_vals)
            .unwrap();
        let (l, u) = nonzero_pattern(&dummy_csc);
        // TODO: Don't clone unnecessarily
        Self {
            m_pattern: Arc::clone(pattern),
            l_pattern: l.pattern().as_ref().clone(),
            u_pattern: u.pattern().as_ref().clone()
        }
    }

    pub fn l_pattern(&self) -> &SparsityPattern {
        &self.l_pattern
    }
}

pub struct CscCholesky<T> {
    // Pattern of the original matrix
    m_pattern: Arc<SparsityPattern>,
    l_factor: CscMatrix<T>,
    u_pattern: SparsityPattern,
    work_x: Vec<T>,
    work_c: Vec<usize>
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CholeskyError {

}

impl<T: RealField> CscCholesky<T> {

    pub fn factor(matrix: &CscMatrix<T>) -> Result<Self, CholeskyError> {
        let symbolic = CscSymbolicCholesky::factor(&*matrix.pattern());
        assert_eq!(symbolic.l_pattern.nnz(), symbolic.u_pattern.nnz(),
            "u is just the transpose of l, so should have the same nnz");

        let l_nnz = symbolic.l_pattern.nnz();
        let l_values = vec![T::zero(); l_nnz];
        let l_factor = CscMatrix::try_from_pattern_and_values(Arc::new(symbolic.l_pattern), l_values)
            .unwrap();

        let mut factorization = CscCholesky {
            m_pattern: symbolic.m_pattern,
            l_factor,
            u_pattern: symbolic.u_pattern,
            work_x: vec![T::zero(); matrix.nrows()],
            // Fill with MAX so that things hopefully totally fail if values are not
            // overwritten. Might be easier to debug this way
            work_c: vec![usize::MAX, matrix.ncols()],
        };

        factorization.refactor(matrix.values())?;
        Ok(factorization)
    }

    pub fn refactor(&mut self, values: &[T]) -> Result<(), CholeskyError> {
        self.decompose_left_looking(values)
    }

    pub fn l(&self) -> &CscMatrix<T> {
        &self.l_factor
    }

    pub fn take_l(self)  -> CscMatrix<T> {
        self.l_factor
    }

    /// Perform a numerical left-looking cholesky decomposition of a matrix with the same structure as the
    /// one used to initialize `self`, but with different non-zero values provided by `values`.
    fn decompose_left_looking(&mut self, values: &[T]) -> Result<(), CholeskyError> {
        assert!(
            values.len() >= self.m_pattern.nnz(),
            // TODO: Improve error message
            "The set of values is too small."
        );

        let n = self.l_factor.nrows();

        // Reset `work_c` to the column pointers of `l`.
        self.work_c.clear();
        self.work_c.extend_from_slice(self.l_factor.col_offsets());

        unsafe {
            for k in 0..n {
                // Scatter the k-th column of the original matrix with the values provided.
                let range_begin = *self.m_pattern.major_offsets().get_unchecked(k);
                let range_end = *self.m_pattern.major_offsets().get_unchecked(k + 1);
                let range_k = range_begin..range_end;

                *self.work_x.get_unchecked_mut(k) = T::zero();
                for p in range_k.clone() {
                    let irow = *self.m_pattern.minor_indices().get_unchecked(p);

                    if irow >= k {
                        *self.work_x.get_unchecked_mut(irow) = *values.get_unchecked(p);
                    }
                }

                for &j in self.u_pattern.lane(k) {
                    let factor = -*self
                        .l_factor
                        .values()
                        .get_unchecked(*self.work_c.get_unchecked(j));
                    *self.work_c.get_unchecked_mut(j) += 1;

                    if j < k {
                        let col_j = self.l_factor.col(j);
                        let col_j_entries = col_j.row_indices().iter().zip(col_j.values());
                        for (&z, val) in col_j_entries {
                            if z >= k {
                                *self.work_x.get_unchecked_mut(z) += val.inlined_clone() * factor;
                            }
                        }
                    }
                }

                let diag = *self.work_x.get_unchecked(k);

                if diag > T::zero() {
                    let denom = diag.sqrt();

                    {
                        let (offsets, _, values) = self.l_factor.csc_data_mut();
                        *values
                            .get_unchecked_mut(*offsets.get_unchecked(k)) = denom;
                    }


                    let mut col_k = self.l_factor.col_mut(k);
                    let (col_k_rows, col_k_values) = col_k.rows_and_values_mut();
                    let col_k_entries = col_k_rows.iter().zip(col_k_values);
                    for (&p, val) in col_k_entries {
                        *val = *self.work_x.get_unchecked(p) / denom;
                        *self.work_x.get_unchecked_mut(p) = T::zero();
                    }
                } else {
                    // self.ok = false;
                    // TODO: Return indefinite error (i.e. encountered non-positive diagonal
                    unimplemented!()
                }
            }
        }

        Ok(())
    }

}




fn reach<T>(
    m: &CscMatrix<T>,
    j: usize,
    max_j: usize,
    tree: &[usize],
    marks: &mut Vec<bool>,
    out: &mut Vec<usize>,
) {
    marks.clear();
    marks.resize(tree.len(), false);

    // TODO: avoid all those allocations.
    let mut tmp = Vec::new();
    let mut res = Vec::new();

    for &irow in m.col(j).row_indices() {
        let mut curr = irow;
        while curr != usize::max_value() && curr <= max_j && !marks[curr] {
            marks[curr] = true;
            tmp.push(curr);
            curr = tree[curr];
        }

        tmp.append(&mut res);
        mem::swap(&mut tmp, &mut res);
    }

    // TODO: Is this right?
    res.sort_unstable();

    out.append(&mut res);
}

fn nonzero_pattern<T: Scalar + Zero>(
    m: &CscMatrix<T>
) -> (CscMatrix<T>, CscMatrix<T>) {
    // TODO: In order to stay as faithful as possible to the original implementation,
    // we here return full matrices, whereas we actually only need to construct sparsity patterns

    let etree = elimination_tree(m);
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let mut rows = Vec::with_capacity(m.nnz());
    // TODO: Use a Vec here instead
    let mut cols = unsafe { VectorN::new_uninitialized_generic(Dynamic::new(nrows), U1) };
    let mut marks = Vec::new();

    // NOTE: the following will actually compute the non-zero pattern of
    // the transpose of l.
    for i in 0..nrows {
        cols[i] = rows.len();
        reach(m, i, i, &etree, &mut marks, &mut rows);
    }

    // TODO: Get rid of this in particular
    let mut vals = Vec::with_capacity(rows.len());
    unsafe {
        vals.set_len(rows.len());
    }
    vals.shrink_to_fit();

    // TODO: Remove this unnecessary conversion by using Vec throughout
    let mut cols: Vec<_> = cols.iter().cloned().collect();
    cols.push(rows.len());

    let u = CscMatrix::try_from_csc_data(nrows, ncols, cols, rows, vals).unwrap();
    // TODO: Avoid this transpose
    let l = u.transpose();

    (l, u)
}

fn elimination_tree<T>(m: &CscMatrix<T>) -> Vec<usize> {
    let nrows = m.nrows();
    let mut forest: Vec<_> = iter::repeat(usize::max_value()).take(nrows).collect();
    let mut ancestor: Vec<_> = iter::repeat(usize::max_value()).take(nrows).collect();

    for k in 0..nrows {
        for &irow in m.col(k).row_indices() {
            let mut i = irow;

            while i < k {
                let i_ancestor = ancestor[i];
                ancestor[i] = k;

                if i_ancestor == usize::max_value() {
                    forest[i] = k;
                    break;
                }

                i = i_ancestor;
            }
        }
    }

    forest
}