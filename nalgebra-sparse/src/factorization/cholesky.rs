use crate::{
    convert::utils::CountToOffsetIter,
    cs::{Compression, CsMatrix, CscMatrix},
};
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, RealField};
use std::{borrow::Borrow, iter};
use thiserror::Error;

/// Intermediate struct to hold the offsets and indices of a Cholesky factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CholeskyPattern {
    /// The shape of the matrix
    pub shape: (usize, usize),

    /// The offsets of the sparsity pattern of the Cholesky factor.
    pub offsets: Vec<usize>,

    /// The indices of the sparsity pattern of the Cholesky factor.
    pub indices: Vec<usize>,
}

/// A sparse Cholesky factorization `A = L L^T` of a [`CscMatrix`].
///
/// The factor `L` is a sparse, lower-triangular matrix. See the article on [Wikipedia] for
/// more information.
///
/// The implementation is a port of the `CsCholesky` implementation in `nalgebra`. It is similar
/// to Tim Davis' [`CSparse`]. The current implementation performs no fill-in reduction, and can
/// therefore be expected to produce much too dense Cholesky factors for many matrices.
/// It is therefore not currently recommended to use this implementation for serious projects.
///
/// [`CSparse`]: https://epubs.siam.org/doi/book/10.1137/1.9780898718881
/// [Wikipedia]: https://en.wikipedia.org/wiki/Cholesky_decomposition
// TODO: We should probably implement PartialEq/Eq, but in that case we'd probably need a
// custom implementation, due to the need to exclude the workspace arrays
#[derive(Debug, Clone)]
pub struct CsCholesky<T> {
    l_matrix: CscMatrix<T>,
}

#[derive(Copy, Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
/// Possible errors produced by the Cholesky factorization.
pub enum CholeskyError {
    /// The matrix is not positive definite.
    #[error("The matrix is not positive definite.")]
    NotPositiveDefinite,

    /// The matrix doesn't have nrows == ncols
    #[error("The matrix is not square.")]
    NotSquare,

    /// The matrix and cholesky pattern have different shapes.
    #[error("The matrix and cholesky pattern have different shapes.")]
    ShapeMismatch,

    /// The number of nonzeros in the matrix and Cholesky pattern are not equal.
    #[error("The number of nonzeros in the matrix and Cholesky pattern are not equal.")]
    NonzerosMismatch,
}

impl<T: RealField> CsCholesky<T> {
    /// Computes the Cholesky factorization of the provided matrix.
    ///
    /// The matrix must be symmetric positive definite. Symmetry is not checked, and it is up
    /// to the user to enforce this property.
    ///
    /// # Errors
    ///
    /// Returns an error if the numerical factorization fails. This can occur if the matrix is not
    /// symmetric positive definite.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn factor<MO, MI, D, C>(matrix: &CsMatrix<T, MO, MI, D, C>) -> Result<Self, CholeskyError>
    where
        MO: Borrow<[usize]>,
        MI: Borrow<[usize]>,
        D: Borrow<[T]>,
        C: Compression,
    {
        let (nrows, ncols) = matrix.shape();

        if nrows == ncols {
            let cholesky_pattern = nonzero_pattern(matrix);
            Self::decompose_left_looking(cholesky_pattern, matrix)
        } else {
            Err(CholeskyError::NotSquare)
        }
    }

    pub fn factor_with_pattern<MO, MI, D, C>(
        pattern: CholeskyPattern,
        matrix: &CsMatrix<T, MO, MI, D, C>,
    ) -> Result<Self, CholeskyError>
    where
        MO: Borrow<[usize]>,
        MI: Borrow<[usize]>,
        D: Borrow<[T]>,
        C: Compression,
    {
        if pattern.shape != matrix.shape() {
            return Err(CholeskyError::ShapeMismatch);
        }

        let (nrows, ncols) = matrix.shape();

        if nrows == ncols {
            Self::decompose_left_looking(pattern, matrix)
        } else {
            Err(CholeskyError::NotSquare)
        }
    }

    /// Perform a numerical left-looking cholesky decomposition of a matrix with the same structure as the
    /// one used to initialize `self`, but with different non-zero values provided by `values`.
    fn decompose_left_looking<MO, MI, D, C>(
        pattern: CholeskyPattern,
        matrix: &CsMatrix<T, MO, MI, D, C>,
    ) -> Result<Self, CholeskyError>
    where
        MO: Borrow<[usize]>,
        MI: Borrow<[usize]>,
        D: Borrow<[T]>,
        C: Compression,
    {
        let work_c = Vec::new();
        work_c.copy_from_slice(pattern.offsets.borrow());

        // Fill with MAX so that things hopefully totally fail if values are not overwritten. Might
        // be easier to debug this way
        let work_x = vec![T::zero(); matrix.nmajor()];

        if matrix.nnz() != pattern.indices.len() {
            return Err(CholeskyError::NonzerosMismatch);
        }

        let data = vec![T::zero(); matrix.nnz()];

        for (i, lane) in matrix.iter().enumerate() {
            for (j, val) in lane {
                if j >= i {
                    work_x[j] = val.clone();
                }
            }

            let offset = pattern.offsets[i];

            {
                let (pattern_lane_indices, pattern_lane_data) = if (i + 1) < pattern.offsets.len() {
                    let offset_upper = pattern.offsets[i + 1];
                    let indices = &pattern.indices[offset..offset_upper];
                    let data = &data[offset..offset_upper];

                    (indices, data)
                } else {
                    let indices = &pattern.indices[offset..];
                    let data = &data[offset..];

                    (indices, data)
                };

                for &j in pattern_lane_indices {
                    work_c[j] += 1;

                    if j < i {
                        let factor = -data[work_c[j]].clone();

                        for (&j_t, val) in pattern_lane_indices.iter().zip(pattern_lane_data) {
                            if j_t >= i {
                                work_x[j_t] += val.clone() * factor.clone();
                            }
                        }
                    }
                }
            }

            let diag = work_x[i].clone();

            if diag > T::zero() {
                let denom = diag.sqrt();

                data[offset] = denom.clone();

                let (pattern_lane_indices, pattern_lane_data) = if (i + 1) < pattern.offsets.len() {
                    let offset_upper = pattern.offsets[i + 1];
                    let indices = &pattern.indices[offset..offset_upper];
                    let data = &mut data[offset..offset_upper];

                    (indices, data)
                } else {
                    let indices = &pattern.indices[offset..];
                    let data = &mut data[offset..];

                    (indices, data)
                };

                for (&p, val) in pattern_lane_indices.iter().zip(pattern_lane_data) {
                    *val = work_x[p].clone() / denom.clone();
                    work_x[p] = T::zero();
                }
            } else {
                return Err(CholeskyError::NotPositiveDefinite);
            }
        }

        let CholeskyPattern {
            shape: (nrows, ncols),
            offsets,
            indices,
        } = pattern;

        Ok(Self {
            l_matrix: unsafe {
                CscMatrix::from_parts_unchecked(nrows, ncols, offsets, indices, data)
            },
        })
    }

    /// Returns a reference to the Cholesky factor `L`.
    #[must_use]
    pub fn l(&self) -> &CscMatrix<T> {
        &self.l_matrix
    }

    /// Returns the Cholesky factor `L`.
    pub fn take_l(self) -> CscMatrix<T> {
        self.l_matrix
    }

    /// Returns the cholesky pattern of the current `L` matrix.
    ///
    /// Useful if you want to re-use the sparsity pattern, e.g. if the values in a matrix are
    /// changed but the layout of the non-zeros did not.
    pub fn into_pattern(self) -> CholeskyPattern {
        let shape = self.l_matrix.shape();
        let (offsets, indices, _) = self.l_matrix.disassemble();

        CholeskyPattern {
            shape,
            offsets,
            indices,
        }
    }

    /// Solves the system `A X = B`, where `X` and `B` are dense matrices.
    ///
    /// # Panics
    ///
    /// Panics if `B` is not square.
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<'a>(&'a self, b: impl Into<DMatrixSlice<'a, T>>) -> DMatrix<T> {
        let b = b.into();
        let mut output = b.clone_owned();
        self.solve_mut(&mut output);
        output
    }

    /// Solves the system `AX = B`, where `X` and `B` are dense matrices.
    ///
    /// The result is stored in-place in `b`.
    ///
    /// # Panics
    ///
    /// Panics if `b` is not square.
    pub fn solve_mut<'a>(&'a self, b: impl Into<DMatrixSliceMut<'a, T>>) {
        let expect_msg = "If the Cholesky factorization succeeded,\
            then the triangular solve should never fail";
        // Solve LY = B
        let mut y = b.into();
        spsolve_csc_lower_triangular(Op::NoOp(self.l()), &mut y).expect(expect_msg);

        // Solve L^T X = Y
        let mut x = y;
        spsolve_csc_lower_triangular(Op::Transpose(self.l()), &mut x).expect(expect_msg);
    }
}

/// Computes the pattern of non-zeros for the Cholesky decomposition of the input matrix.
fn nonzero_pattern<T, MO, MI, D, C>(matrix: &CsMatrix<T, MO, MI, D, C>) -> CholeskyPattern
where
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
    C: Compression,
{
    let etree = elimination_tree(matrix);

    let nmajor = matrix.nmajor();

    let mut new_indices = Vec::with_capacity(matrix.nnz());
    let mut marks = vec![false; etree.len()];

    let new_offsets = CountToOffsetIter::new(matrix.iter().enumerate().map(|(i, lane)| {
        marks.fill(false);

        let indices = lane
            .flat_map(|(j, _)| {
                let mut res = Vec::with_capacity(nmajor - i);
                let mut current = Some(j);

                while let Some(curr) = current {
                    if curr > i || marks[curr] {
                        break;
                    }

                    marks[curr] = true;
                    res.push(curr);

                    current = etree[curr];
                }

                res
            })
            .collect::<Vec<_>>();

        let count = indices.len();

        indices.sort_unstable();
        new_indices.append(&mut indices);

        count
    }))
    .collect();

    CholeskyPattern {
        shape: matrix.shape(),
        offsets: new_offsets,
        indices: new_indices,
    }
}

/// Computes the elimination tree of the input matrix.
fn elimination_tree<T, MO, MI, D, C>(matrix: &CsMatrix<T, MO, MI, D, C>) -> Vec<Option<usize>>
where
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
    C: Compression,
{
    let nminor = matrix.nminor();

    let mut forest = iter::repeat(None).take(nminor).collect::<Vec<_>>();
    let mut ancestor = iter::repeat(None).take(nminor).collect::<Vec<_>>();

    for (k, lane) in matrix.iter().enumerate() {
        for (i_minor, _) in lane {
            let mut index = Some(i_minor);

            while let Some(i) = index {
                if i >= k {
                    break;
                }

                let i_ancestor = ancestor[i];
                ancestor[i] = Some(k);

                if i_ancestor.is_none() {
                    forest[i] = Some(k);
                    break;
                }

                index = i_ancestor;
            }
        }
    }

    forest
}
