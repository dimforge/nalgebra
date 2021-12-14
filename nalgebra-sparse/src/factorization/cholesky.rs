use crate::{
    convert::utils::CountToOffsetIter,
    cs::{Compression, CsMatrix, CscMatrix},
    ops::serial::spsolve::*,
};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, Matrix, RealField, Scalar, Storage, StorageMut,
};
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

impl CholeskyPattern {
    fn transpose(&self) -> Self {
        let (nrows, ncols) = self.shape;

        let mut counts = vec![0usize; ncols];
        let mut new_indices = Vec::with_capacity(self.indices.len());

        for i in 0..ncols {
            for j in 0..self.offsets.len() {
                let offset = self.offsets[j];
                let indices = if (j + 1) < self.offsets.len() {
                    let offset_upper = self.offsets[j + 1];
                    &self.indices[offset..offset_upper]
                } else {
                    &self.indices[offset..]
                };

                if let Ok(_) = indices.binary_search_by(|&x| x.cmp(&i)) {
                    counts[i] += 1;
                    new_indices.push(j);
                }
            }
        }

        let new_offsets = CountToOffsetIter::new(counts).collect();

        Self {
            shape: (ncols, nrows),
            offsets: new_offsets,
            indices: new_indices,
        }
    }
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
pub struct CsCholesky<T>
where
    T: Scalar + RealField,
{
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
}

impl<T: Scalar + RealField> CsCholesky<T> {
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
            let lt_pattern = nonzero_pattern(matrix);
            Self::decompose_left_looking(lt_pattern.transpose(), lt_pattern, matrix)
        } else {
            Err(CholeskyError::NotSquare)
        }
    }

    /// Takes in an existing Cholesky pattern and a matrix, and recomputes the factors.
    ///
    /// This is a useful constructor for the CsCholesky type if you're re-using a pattern from a
    /// previous matrix where the values have been updated but the sparsity pattern hasn't changed
    /// at all.
    ///
    /// # Errors
    ///
    /// Returns an error if the numerical factorization fails. This can occur if:
    ///
    /// - The pattern and matrix shapes are not equivalent.
    /// - The pattern and / or matrix are not square.
    /// - The pattern and matrix have an inequal number of nonzeros.
    /// - The matrix is not positive-definite.
    pub fn factor_with_pattern<MO, MI, D, C>(
        l_pattern: CholeskyPattern,
        matrix: &CsMatrix<T, MO, MI, D, C>,
    ) -> Result<Self, CholeskyError>
    where
        MO: Borrow<[usize]>,
        MI: Borrow<[usize]>,
        D: Borrow<[T]>,
        C: Compression,
    {
        if l_pattern.shape != matrix.shape() {
            return Err(CholeskyError::ShapeMismatch);
        }

        let (nrows, ncols) = matrix.shape();

        if nrows == ncols {
            let lt_pattern = l_pattern.transpose();
            Self::decompose_left_looking(l_pattern, lt_pattern, matrix)
        } else {
            Err(CholeskyError::NotSquare)
        }
    }

    /// Perform a numerical left-looking cholesky decomposition of a matrix with the same structure as the
    /// one used to initialize `self`, but with different non-zero values provided by `values`.
    fn decompose_left_looking<MO, MI, D, C>(
        l_pattern: CholeskyPattern,
        u_pattern: CholeskyPattern,
        matrix: &CsMatrix<T, MO, MI, D, C>,
    ) -> Result<Self, CholeskyError>
    where
        MO: Borrow<[usize]>,
        MI: Borrow<[usize]>,
        D: Borrow<[T]>,
        C: Compression,
    {
        let mut work_c = l_pattern.offsets.clone();
        let mut work_x = vec![T::zero(); matrix.nmajor()];

        let mut data = vec![T::zero(); l_pattern.indices.len()];

        for (i, lane) in matrix.iter().enumerate() {
            work_x[i] = T::zero();

            for (j, val) in lane {
                if j >= i {
                    work_x[j] = val.clone();
                }
            }

            {
                let offset = u_pattern.offsets[i];

                let lane_i_indices = if (i + 1) < u_pattern.offsets.len() {
                    let offset_upper = u_pattern.offsets[i + 1];
                    &u_pattern.indices[offset..offset_upper]
                } else {
                    &u_pattern.indices[offset..]
                };

                for &j in lane_i_indices {
                    let factor = -data[work_c[j]].clone();
                    work_c[j] += 1;

                    if j < i {
                        let offset_j = l_pattern.offsets[j];
                        let (lane_j_indices, lane_j_data) = if (j + 1) < l_pattern.offsets.len() {
                            let offset_upper = l_pattern.offsets[j + 1];
                            let indices = &l_pattern.indices[offset_j..offset_upper];
                            let data = &data[offset_j..offset_upper];

                            (indices, data)
                        } else {
                            let indices = &l_pattern.indices[offset_j..];
                            let data = &data[offset_j..];

                            (indices, data)
                        };

                        for (&j_t, val) in lane_j_indices.iter().zip(lane_j_data) {
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

                let offset = l_pattern.offsets[i];
                data[offset] = denom.clone();

                let (pattern_lane_indices, pattern_lane_data) = if (i + 1) < l_pattern.offsets.len()
                {
                    let offset_upper = l_pattern.offsets[i + 1];
                    let indices = &l_pattern.indices[offset..offset_upper];
                    let data = &mut data[offset..offset_upper];

                    (indices, data)
                } else {
                    let indices = &l_pattern.indices[offset..];
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
        } = l_pattern;

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
    /// Panics if `B` is the wrong size i.e. for an N×N matrix `A`, `B` must be some N×M matrix.
    #[must_use]
    pub fn solve<R, C, S>(
        &self,
        b: &Matrix<T, R, C, S>,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>,
        DefaultAllocator: Allocator<T, R, C>,
    {
        let b_clone = b.clone_owned();
        self.solve_mut(b_clone)
    }

    /// Solves the system `AX = B`, where `X` and `B` are dense matrices.
    ///
    /// The result is stored in-place in `b`. We take ownership of `b`, mutate it directly, and
    /// then return the same matrix.
    ///
    /// # Panics
    ///
    /// Panics if `B` is the wrong size i.e. for an N×N matrix `A`, `B` must be some N×M matrix.
    #[must_use]
    pub fn solve_mut<R, C, S>(&self, b: Matrix<T, R, C, S>) -> Matrix<T, R, C, S>
    where
        R: Dim,
        C: Dim,
        S: Storage<T, R, C> + StorageMut<T, R, C>,
    {
        // If the factorization succeeded, then the solve should only fail if the input matrix is
        // of the wrong size. Therefore, we merely unwrap here.

        // Solve LY = B
        let y = spsolve_lower_triangular_csc_dense(self.l_matrix.to_view(), b).unwrap();

        // Solve L^T X = Y
        spsolve_upper_triangular_csr_dense(self.l_matrix.transpose(), y).unwrap()
    }
}

/// Computes the pattern of non-zeros for the Cholesky decomposition of the input matrix.
fn nonzero_pattern<T, MO, MI, D, C>(matrix: &CsMatrix<T, MO, MI, D, C>) -> CholeskyPattern
where
    T: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
    C: Compression,
{
    let etree = elimination_tree(matrix);
    let nmajor = matrix.nmajor();

    let mut counts = vec![0usize; nmajor];
    let mut new_indices = Vec::with_capacity(matrix.nnz());
    let mut marks = vec![false; etree.len()];

    for (i, lane) in matrix.iter().enumerate() {
        marks.fill(false);

        let mut indices = lane
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

        counts[i] += count;
    }

    let new_offsets = CountToOffsetIter::new(counts).collect();

    CholeskyPattern {
        shape: matrix.shape(),
        offsets: new_offsets,
        indices: new_indices,
    }
}

/// Computes the elimination tree of the input matrix.
fn elimination_tree<T, MO, MI, D, C>(matrix: &CsMatrix<T, MO, MI, D, C>) -> Vec<Option<usize>>
where
    T: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
    C: Compression,
{
    let n = matrix.nmajor();

    let mut forest = iter::repeat(None).take(n).collect::<Vec<_>>();
    let mut ancestor = iter::repeat(None).take(n).collect::<Vec<_>>();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cs::{CscMatrix, CsrMatrix},
        proptest::*,
    };
    use matrixcompare::{assert_matrix_eq, prop_assert_matrix_eq};
    use nalgebra::{proptest::matrix, DMatrix, Matrix5, Vector5};
    use proptest::prelude::*;

    /// The tolerance at which we will make value comparisons when performing matrix equality
    /// checks.
    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn cholesky_correct_for_positive_definite_example_1() {
        #[rustfmt::skip]
        let mut a = Matrix5::new(
            40.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 60.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 11.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 50.0, 0.0,
            1.0, 0.0, 0.0, 4.0, 10.0
        );
        a.fill_upper_triangle_with_lower_triangle();

        let csc = CscMatrix::from(&a);

        let chol_a = a.cholesky().unwrap();
        let chol_cs_a = CsCholesky::factor(&csc).unwrap();

        let l = chol_a.unpack();
        let cs_l = chol_cs_a.take_l();

        assert_matrix_eq!(l, cs_l, comp = abs, tol = TOLERANCE);
    }

    #[test]
    fn cholesky_correct_for_positive_definite_example_2() {
        #[rustfmt::skip]
        let mut a = Matrix5::new(
            40.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 60.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 11.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 50.0, 0.0,
            0.0, 0.0, 0.0, 4.0, 10.0
        );
        a.fill_upper_triangle_with_lower_triangle();

        let csc = CscMatrix::from(&a);

        let chol_a = a.cholesky().unwrap();
        let chol_cs_a = CsCholesky::factor(&csc).unwrap();

        let l = chol_a.unpack();
        let cs_l = chol_cs_a.take_l();

        assert_matrix_eq!(l, cs_l, comp = abs, tol = TOLERANCE);
    }

    #[test]
    fn cholesky_correct_for_positive_definite_example_3() {
        let a = Matrix5::from_diagonal(&Vector5::new(40.0, 60.0, 11.0, 50.0, 10.0));

        let csc = CscMatrix::from(&a);

        let chol_a = a.cholesky().unwrap();
        let chol_cs_a = CsCholesky::factor(&csc).unwrap();

        let l = chol_a.unpack();
        let cs_l = chol_cs_a.take_l();

        assert_matrix_eq!(l, cs_l, comp = abs, tol = TOLERANCE);
    }

    #[test]
    fn cholesky_correct_for_positive_definite_example_4() {
        #[rustfmt::skip]
        let mut a = Matrix5::new(
            2.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 2.0
        );
        a.fill_upper_triangle_with_lower_triangle();

        let csc = CscMatrix::from(&a);

        let chol_a = a.cholesky().unwrap();
        let chol_cs_a = CsCholesky::factor(&csc).unwrap();

        let l = chol_a.unpack();
        let cs_l = chol_cs_a.take_l();

        assert_matrix_eq!(l, cs_l, comp = abs, tol = TOLERANCE);
    }

    proptest! {
        #[test]
        fn nonzero_cholesky_pattern_of_identity_matrix_is_same_as_identity(n in 0..100usize) {
            let eye = CsrMatrix::<f32>::identity(n);
            let pattern = nonzero_pattern(&eye);

            let (offsets, indices, _) = eye.cs_data();

            prop_assert_eq!(offsets, pattern.offsets);
            prop_assert_eq!(indices, pattern.indices);
            prop_assert_eq!(indices.len(), eye.nnz());
        }

        #[test]
        fn nonzero_pattern_of_cholesky_agrees_with_dense(matrix in csc_positive_definite()) {
            let dense = DMatrix::from(&matrix);
            let l_dense = dense.cholesky().unwrap().unpack();
            let lt_dense = l_dense.transpose();

            let l_as_csc = CscMatrix::from(&l_dense);
            let lt_as_csc = CscMatrix::from(&lt_dense);

            let (l_offsets, l_indices, _) = l_as_csc.disassemble();
            let (lt_offsets, lt_indices, _) = lt_as_csc.disassemble();

            // nonzero_pattern computes L^T
            let lt_pattern = nonzero_pattern(&matrix);
            let l_pattern = lt_pattern.transpose();

            prop_assert_eq!(l_pattern.offsets, l_offsets);
            prop_assert_eq!(l_pattern.indices, l_indices);

            prop_assert_eq!(lt_pattern.offsets, lt_offsets);
            prop_assert_eq!(lt_pattern.indices, lt_indices);
        }

        #[test]
        fn cholesky_of_csr_identity_matrix_is_identity(n in 0..100usize) {
            let eye = CsrMatrix::<f64>::identity(n);
            let cholesky = CsCholesky::factor(&eye).unwrap();
            let l = cholesky.take_l();

            let reconstructed = l.to_view() * l.transpose();

            prop_assert_matrix_eq!(eye, l, comp = abs, tol = TOLERANCE);
            prop_assert_matrix_eq!(eye, reconstructed, comp = abs, tol = TOLERANCE);
        }

        #[test]
        fn cholesky_of_csc_identity_matrix_is_identity(n in 0..100usize) {
            let eye = CscMatrix::<f64>::identity(n);
            let cholesky = CsCholesky::factor(&eye).unwrap();
            let l = cholesky.take_l();

            let reconstructed = l.to_view() * l.transpose();

            prop_assert_matrix_eq!(eye, reconstructed, comp = abs, tol = TOLERANCE);
        }

        #[test]
        fn csc_cholesky_correct_for_positive_definite_matrices(matrix in csc_positive_definite()) {
            let cholesky = CsCholesky::factor(&matrix).unwrap();
            let l = cholesky.take_l();
            let matrix_reconstructed = l.to_view() * l.transpose();

            prop_assert_matrix_eq!(matrix_reconstructed, matrix, comp = abs, tol = TOLERANCE);

            let is_lower_triangular = l.triplet_iter().all(|(i, j, _)| i <= j);
            prop_assert!(is_lower_triangular);
        }

        #[test]
        fn csr_cholesky_correct_for_positive_definite_matrices(matrix in csr_positive_definite()) {
            let cholesky = CsCholesky::factor(&matrix).unwrap();
            let l = cholesky.take_l();
            let matrix_reconstructed = l.to_view() * l.transpose();

            prop_assert_matrix_eq!(matrix_reconstructed, matrix, comp = abs, tol = TOLERANCE);

            let is_lower_triangular = l.triplet_iter().all(|(i, j, _)| i <= j);
            prop_assert!(is_lower_triangular);
        }

        #[test]
        fn cholesky_solve_positive_definite((matrix, rhs) in csc_positive_definite().prop_flat_map(|csc| {
            let rhs = matrix(value_strategy::<f64>(), csc.nrows(), PROPTEST_MATRIX_DIM);
            (Just(csc), rhs)
        })) {
            let cholesky = CsCholesky::factor(&matrix).unwrap();

            // solve_mut
            {
                let x = cholesky.solve_mut(rhs.clone());
                prop_assert_matrix_eq!(matrix.to_view() * x, rhs, comp = abs, tol = TOLERANCE);
            }

            // solve
            {
                let x = cholesky.solve(&rhs);
                prop_assert_matrix_eq!(matrix.to_view() * x, rhs, comp = abs, tol = TOLERANCE);
            }
        }
    }
}
