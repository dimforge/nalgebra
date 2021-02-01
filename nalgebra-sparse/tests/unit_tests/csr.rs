use nalgebra::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::SparseFormatErrorKind;

use proptest::prelude::*;
use proptest::sample::subsequence;

use crate::common::csr_strategy;

use std::collections::HashSet;

#[test]
fn csr_matrix_valid_data() {
    // Construct matrix from valid data and check that selected methods return results
    // that agree with expectations.

    {
        // A CSR matrix with zero explicitly stored entries
        let offsets = vec![0, 0, 0, 0];
        let indices = vec![];
        let values = Vec::<i32>::new();
        let mut matrix = CsrMatrix::try_from_csr_data(3, 2, offsets, indices, values).unwrap();

        assert_eq!(matrix, CsrMatrix::zeros(3, 2));

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.row_offsets(), &[0, 0, 0, 0]);
        assert_eq!(matrix.col_indices(), &[]);
        assert_eq!(matrix.values(), &[]);

        assert!(matrix.triplet_iter().next().is_none());
        assert!(matrix.triplet_iter_mut().next().is_none());

        assert_eq!(matrix.row(0).ncols(), 2);
        assert_eq!(matrix.row(0).nnz(), 0);
        assert_eq!(matrix.row(0).col_indices(), &[]);
        assert_eq!(matrix.row(0).values(), &[]);
        assert_eq!(matrix.row_mut(0).ncols(), 2);
        assert_eq!(matrix.row_mut(0).nnz(), 0);
        assert_eq!(matrix.row_mut(0).col_indices(), &[]);
        assert_eq!(matrix.row_mut(0).values(), &[]);
        assert_eq!(matrix.row_mut(0).values_mut(), &[]);
        assert_eq!(
            matrix.row_mut(0).cols_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.row(1).ncols(), 2);
        assert_eq!(matrix.row(1).nnz(), 0);
        assert_eq!(matrix.row(1).col_indices(), &[]);
        assert_eq!(matrix.row(1).values(), &[]);
        assert_eq!(matrix.row_mut(1).ncols(), 2);
        assert_eq!(matrix.row_mut(1).nnz(), 0);
        assert_eq!(matrix.row_mut(1).col_indices(), &[]);
        assert_eq!(matrix.row_mut(1).values(), &[]);
        assert_eq!(matrix.row_mut(1).values_mut(), &[]);
        assert_eq!(
            matrix.row_mut(1).cols_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.row(2).ncols(), 2);
        assert_eq!(matrix.row(2).nnz(), 0);
        assert_eq!(matrix.row(2).col_indices(), &[]);
        assert_eq!(matrix.row(2).values(), &[]);
        assert_eq!(matrix.row_mut(2).ncols(), 2);
        assert_eq!(matrix.row_mut(2).nnz(), 0);
        assert_eq!(matrix.row_mut(2).col_indices(), &[]);
        assert_eq!(matrix.row_mut(2).values(), &[]);
        assert_eq!(matrix.row_mut(2).values_mut(), &[]);
        assert_eq!(
            matrix.row_mut(2).cols_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert!(matrix.get_row(3).is_none());
        assert!(matrix.get_row_mut(3).is_none());

        let (offsets, indices, values) = matrix.disassemble();

        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert_eq!(indices, vec![]);
        assert_eq!(values, vec![]);
    }

    {
        // An arbitrary CSR matrix
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let mut matrix =
            CsrMatrix::try_from_csr_data(3, 6, offsets.clone(), indices.clone(), values.clone())
                .unwrap();

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 6);
        assert_eq!(matrix.nnz(), 5);
        assert_eq!(matrix.row_offsets(), &[0, 2, 2, 5]);
        assert_eq!(matrix.col_indices(), &[0, 5, 1, 2, 3]);
        assert_eq!(matrix.values(), &[0, 1, 2, 3, 4]);

        let expected_triplets = vec![(0, 0, 0), (0, 5, 1), (2, 1, 2), (2, 2, 3), (2, 3, 4)];
        assert_eq!(
            matrix
                .triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>(),
            expected_triplets
        );
        assert_eq!(
            matrix
                .triplet_iter_mut()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>(),
            expected_triplets
        );

        assert_eq!(matrix.row(0).ncols(), 6);
        assert_eq!(matrix.row(0).nnz(), 2);
        assert_eq!(matrix.row(0).col_indices(), &[0, 5]);
        assert_eq!(matrix.row(0).values(), &[0, 1]);
        assert_eq!(matrix.row_mut(0).ncols(), 6);
        assert_eq!(matrix.row_mut(0).nnz(), 2);
        assert_eq!(matrix.row_mut(0).col_indices(), &[0, 5]);
        assert_eq!(matrix.row_mut(0).values(), &[0, 1]);
        assert_eq!(matrix.row_mut(0).values_mut(), &[0, 1]);
        assert_eq!(
            matrix.row_mut(0).cols_and_values_mut(),
            ([0, 5].as_ref(), [0, 1].as_mut())
        );

        assert_eq!(matrix.row(1).ncols(), 6);
        assert_eq!(matrix.row(1).nnz(), 0);
        assert_eq!(matrix.row(1).col_indices(), &[]);
        assert_eq!(matrix.row(1).values(), &[]);
        assert_eq!(matrix.row_mut(1).ncols(), 6);
        assert_eq!(matrix.row_mut(1).nnz(), 0);
        assert_eq!(matrix.row_mut(1).col_indices(), &[]);
        assert_eq!(matrix.row_mut(1).values(), &[]);
        assert_eq!(matrix.row_mut(1).values_mut(), &[]);
        assert_eq!(
            matrix.row_mut(1).cols_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.row(2).ncols(), 6);
        assert_eq!(matrix.row(2).nnz(), 3);
        assert_eq!(matrix.row(2).col_indices(), &[1, 2, 3]);
        assert_eq!(matrix.row(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.row_mut(2).ncols(), 6);
        assert_eq!(matrix.row_mut(2).nnz(), 3);
        assert_eq!(matrix.row_mut(2).col_indices(), &[1, 2, 3]);
        assert_eq!(matrix.row_mut(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.row_mut(2).values_mut(), &[2, 3, 4]);
        assert_eq!(
            matrix.row_mut(2).cols_and_values_mut(),
            ([1, 2, 3].as_ref(), [2, 3, 4].as_mut())
        );

        assert!(matrix.get_row(3).is_none());
        assert!(matrix.get_row_mut(3).is_none());

        let (offsets2, indices2, values2) = matrix.disassemble();

        assert_eq!(offsets2, offsets);
        assert_eq!(indices2, indices);
        assert_eq!(values2, values);
    }
}

#[test]
fn csr_matrix_try_from_invalid_csr_data() {
    {
        // Empty offset array (invalid length)
        let matrix = CsrMatrix::try_from_csr_data(0, 0, Vec::new(), Vec::new(), Vec::<u32>::new());
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let offsets = vec![0, 3, 5];
        let indices = vec![0, 1, 2, 3, 5];
        let values = vec![0, 1, 2, 3, 4];

        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid first entry in offsets array
        let offsets = vec![1, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid last entry in offsets array
        let offsets = vec![0, 2, 2, 4];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid length of offsets array
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic offsets
        let offsets = vec![0, 3, 2, 5];
        let indices = vec![0, 1, 2, 3, 4];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic minor indices
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 2, 3, 1, 4];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Minor index out of bounds
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 6, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Duplicate entry
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 2, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }
}

#[test]
fn csr_disassemble_avoids_clone_when_owned() {
    // Test that disassemble avoids cloning the sparsity pattern when it holds the sole reference
    // to the pattern. We do so by checking that the pointer to the data is unchanged.

    let offsets = vec![0, 2, 2, 5];
    let indices = vec![0, 5, 1, 2, 3];
    let values = vec![0, 1, 2, 3, 4];
    let offsets_ptr = offsets.as_ptr();
    let indices_ptr = indices.as_ptr();
    let values_ptr = values.as_ptr();
    let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values).unwrap();

    let (offsets, indices, values) = matrix.disassemble();
    assert_eq!(offsets.as_ptr(), offsets_ptr);
    assert_eq!(indices.as_ptr(), indices_ptr);
    assert_eq!(values.as_ptr(), values_ptr);
}

#[test]
fn csr_matrix_get_index() {
    // TODO: Implement tests for ::get() and index()
}

#[test]
fn csr_matrix_row_iter() {
    // TODO
}

proptest! {
    #[test]
    fn csr_double_transpose_is_identity(csr in csr_strategy()) {
        prop_assert_eq!(csr.transpose().transpose(), csr);
    }

    #[test]
    fn csr_transpose_agrees_with_dense(csr in csr_strategy()) {
        let dense_transpose = DMatrix::from(&csr).transpose();
        let csr_transpose = csr.transpose();
        prop_assert_eq!(dense_transpose, DMatrix::from(&csr_transpose));
        prop_assert_eq!(csr.nnz(), csr_transpose.nnz());
    }

    #[test]
    fn csr_filter(
        (csr, triplet_subset)
        in csr_strategy()
            .prop_flat_map(|matrix| {
                let triplets: Vec<_> = matrix.triplet_iter().cloned_values().collect();
                let subset = subsequence(triplets, 0 ..= matrix.nnz())
                    .prop_map(|triplet_subset| {
                        let set: HashSet<_> = triplet_subset.into_iter().collect();
                        set
                    });
                (Just(matrix), subset)
            }))
    {
        // We generate a CsrMatrix and a HashSet corresponding to a subset of the (i, j, v)
        // values in the matrix, which we use for filtering the matrix entries.
        // The resulting triplets in the filtered matrix must then be exactly equal to
        // the subset.
        let filtered = csr.filter(|i, j, v| triplet_subset.contains(&(i, j, *v)));
        let filtered_triplets: HashSet<_> = filtered
            .triplet_iter()
            .cloned_values()
            .collect();

        prop_assert_eq!(filtered_triplets, triplet_subset);
    }

    #[test]
    fn csr_lower_triangle_agrees_with_dense(csr in csr_strategy()) {
        let csr_lower_triangle = csr.lower_triangle();
        prop_assert_eq!(DMatrix::from(&csr_lower_triangle), DMatrix::from(&csr).lower_triangle());
        prop_assert!(csr_lower_triangle.nnz() <= csr.nnz());
    }

    #[test]
    fn csr_upper_triangle_agrees_with_dense(csr in csr_strategy()) {
        let csr_upper_triangle = csr.upper_triangle();
        prop_assert_eq!(DMatrix::from(&csr_upper_triangle), DMatrix::from(&csr).upper_triangle());
        prop_assert!(csr_upper_triangle.nnz() <= csr.nnz());
    }

    #[test]
    fn csr_diagonal_as_csr(csr in csr_strategy()) {
        let d = csr.diagonal_as_csr();
        let d_entries: HashSet<_> = d.triplet_iter().cloned_values().collect();
        let csr_diagonal_entries: HashSet<_> = csr
            .triplet_iter()
            .cloned_values()
            .filter(|&(i, j, _)| i == j)
            .collect();

        prop_assert_eq!(d_entries, csr_diagonal_entries);
    }

    #[test]
    fn csr_identity(n in 0 ..= 6usize) {
        let csr = CsrMatrix::<i32>::identity(n);
        prop_assert_eq!(csr.nnz(), n);
        prop_assert_eq!(DMatrix::from(&csr), DMatrix::identity(n, n));
    }
}
