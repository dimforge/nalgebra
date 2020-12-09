use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::SparseFormatErrorKind;
use nalgebra::DMatrix;

use proptest::prelude::*;

use crate::common::csc_strategy;

#[test]
fn csc_matrix_valid_data() {
    // Construct matrix from valid data and check that selected methods return results
    // that agree with expectations.

    {
        // A CSC matrix with zero explicitly stored entries
        let offsets = vec![0, 0, 0, 0];
        let indices = vec![];
        let values = Vec::<i32>::new();
        let mut matrix = CscMatrix::try_from_csc_data(2, 3, offsets, indices, values).unwrap();

        assert_eq!(matrix, CscMatrix::new(2, 3));

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.col_offsets(), &[0, 0, 0, 0]);
        assert_eq!(matrix.row_indices(), &[]);
        assert_eq!(matrix.values(), &[]);

        assert!(matrix.triplet_iter().next().is_none());
        assert!(matrix.triplet_iter_mut().next().is_none());

        assert_eq!(matrix.col(0).nrows(), 2);
        assert_eq!(matrix.col(0).nnz(), 0);
        assert_eq!(matrix.col(0).row_indices(), &[]);
        assert_eq!(matrix.col(0).values(), &[]);
        assert_eq!(matrix.col_mut(0).nrows(), 2);
        assert_eq!(matrix.col_mut(0).nnz(), 0);
        assert_eq!(matrix.col_mut(0).row_indices(), &[]);
        assert_eq!(matrix.col_mut(0).values(), &[]);
        assert_eq!(matrix.col_mut(0).values_mut(), &[]);
        assert_eq!(matrix.col_mut(0).rows_and_values_mut(), ([].as_ref(), [].as_mut()));

        assert_eq!(matrix.col(1).nrows(), 2);
        assert_eq!(matrix.col(1).nnz(), 0);
        assert_eq!(matrix.col(1).row_indices(), &[]);
        assert_eq!(matrix.col(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).nrows(), 2);
        assert_eq!(matrix.col_mut(1).nnz(), 0);
        assert_eq!(matrix.col_mut(1).row_indices(), &[]);
        assert_eq!(matrix.col_mut(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).values_mut(), &[]);
        assert_eq!(matrix.col_mut(1).rows_and_values_mut(), ([].as_ref(), [].as_mut()));

        assert_eq!(matrix.col(2).nrows(), 2);
        assert_eq!(matrix.col(2).nnz(), 0);
        assert_eq!(matrix.col(2).row_indices(), &[]);
        assert_eq!(matrix.col(2).values(), &[]);
        assert_eq!(matrix.col_mut(2).nrows(), 2);
        assert_eq!(matrix.col_mut(2).nnz(), 0);
        assert_eq!(matrix.col_mut(2).row_indices(), &[]);
        assert_eq!(matrix.col_mut(2).values(), &[]);
        assert_eq!(matrix.col_mut(2).values_mut(), &[]);
        assert_eq!(matrix.col_mut(2).rows_and_values_mut(), ([].as_ref(), [].as_mut()));

        assert!(matrix.get_col(3).is_none());
        assert!(matrix.get_col_mut(3).is_none());

        let (offsets, indices, values) = matrix.disassemble();

        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert_eq!(indices, vec![]);
        assert_eq!(values, vec![]);
    }

    {
        // An arbitrary CSC matrix
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let mut matrix = CscMatrix::try_from_csc_data(6,
                                                      3,
                                                      offsets.clone(),
                                                      indices.clone(),
                                                      values.clone()).unwrap();

        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(matrix.nnz(), 5);
        assert_eq!(matrix.col_offsets(), &[0, 2, 2, 5]);
        assert_eq!(matrix.row_indices(), &[0, 5, 1, 2, 3]);
        assert_eq!(matrix.values(), &[0, 1, 2, 3, 4]);

        let expected_triplets = vec![(0, 0, 0), (5, 0, 1), (1, 2, 2), (2, 2, 3), (3, 2, 4)];
        assert_eq!(matrix.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>(),
                   expected_triplets);
        assert_eq!(matrix.triplet_iter_mut().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>(),
                   expected_triplets);

        assert_eq!(matrix.col(0).nrows(), 6);
        assert_eq!(matrix.col(0).nnz(), 2);
        assert_eq!(matrix.col(0).row_indices(), &[0, 5]);
        assert_eq!(matrix.col(0).values(), &[0, 1]);
        assert_eq!(matrix.col_mut(0).nrows(), 6);
        assert_eq!(matrix.col_mut(0).nnz(), 2);
        assert_eq!(matrix.col_mut(0).row_indices(), &[0, 5]);
        assert_eq!(matrix.col_mut(0).values(), &[0, 1]);
        assert_eq!(matrix.col_mut(0).values_mut(), &[0, 1]);
        assert_eq!(matrix.col_mut(0).rows_and_values_mut(), ([0, 5].as_ref(), [0, 1].as_mut()));

        assert_eq!(matrix.col(1).nrows(), 6);
        assert_eq!(matrix.col(1).nnz(), 0);
        assert_eq!(matrix.col(1).row_indices(), &[]);
        assert_eq!(matrix.col(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).nrows(), 6);
        assert_eq!(matrix.col_mut(1).nnz(), 0);
        assert_eq!(matrix.col_mut(1).row_indices(), &[]);
        assert_eq!(matrix.col_mut(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).values_mut(), &[]);
        assert_eq!(matrix.col_mut(1).rows_and_values_mut(), ([].as_ref(), [].as_mut()));

        assert_eq!(matrix.col(2).nrows(), 6);
        assert_eq!(matrix.col(2).nnz(), 3);
        assert_eq!(matrix.col(2).row_indices(), &[1, 2, 3]);
        assert_eq!(matrix.col(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.col_mut(2).nrows(), 6);
        assert_eq!(matrix.col_mut(2).nnz(), 3);
        assert_eq!(matrix.col_mut(2).row_indices(), &[1, 2, 3]);
        assert_eq!(matrix.col_mut(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.col_mut(2).values_mut(), &[2, 3, 4]);
        assert_eq!(matrix.col_mut(2).rows_and_values_mut(), ([1, 2, 3].as_ref(), [2, 3, 4].as_mut()));

        assert!(matrix.get_col(3).is_none());
        assert!(matrix.get_col_mut(3).is_none());

        let (offsets2, indices2, values2) = matrix.disassemble();

        assert_eq!(offsets2, offsets);
        assert_eq!(indices2, indices);
        assert_eq!(values2, values);
    }
}

#[test]
fn csc_matrix_try_from_invalid_csc_data() {

    {
        // Empty offset array (invalid length)
        let matrix = CscMatrix::try_from_csc_data(0, 0, Vec::new(), Vec::new(), Vec::<u32>::new());
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Offset array invalid length for arbitrary data
        let offsets = vec![0, 3, 5];
        let indices = vec![0, 1, 2, 3, 5];
        let values = vec![0, 1, 2, 3, 4];

        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Invalid first entry in offsets array
        let offsets = vec![1, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Invalid last entry in offsets array
        let offsets = vec![0, 2, 2, 4];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Invalid length of offsets array
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Nonmonotonic offsets
        let offsets = vec![0, 3, 2, 5];
        let indices = vec![0, 1, 2, 3, 4];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Nonmonotonic minor indices
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 2, 3, 1, 4];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    {
        // Minor index out of bounds
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 6, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::IndexOutOfBounds);
    }

    {
        // Duplicate entry
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 2, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(matrix.unwrap_err().kind(), &SparseFormatErrorKind::DuplicateEntry);
    }

}

#[test]
fn csc_disassemble_avoids_clone_when_owned() {
    // Test that disassemble avoids cloning the sparsity pattern when it holds the sole reference
    // to the pattern. We do so by checking that the pointer to the data is unchanged.

    let offsets = vec![0, 2, 2, 5];
    let indices = vec![0, 5, 1, 2, 3];
    let values = vec![0, 1, 2, 3, 4];
    let offsets_ptr = offsets.as_ptr();
    let indices_ptr = indices.as_ptr();
    let values_ptr = values.as_ptr();
    let matrix = CscMatrix::try_from_csc_data(6,
                                              3,
                                              offsets,
                                              indices,
                                              values).unwrap();

    let (offsets, indices, values) = matrix.disassemble();
    assert_eq!(offsets.as_ptr(), offsets_ptr);
    assert_eq!(indices.as_ptr(), indices_ptr);
    assert_eq!(values.as_ptr(), values_ptr);
}

#[test]
fn csc_matrix_get_index() {
    // TODO: Implement tests for ::get() and index()
}

#[test]
fn csc_matrix_col_iter() {
    // TODO
}

proptest! {
    #[test]
    fn csc_double_transpose_is_identity(csc in csc_strategy()) {
        prop_assert_eq!(csc.transpose().transpose(), csc);
    }

    #[test]
    fn csc_transpose_agrees_with_dense(csc in csc_strategy()) {
        let dense_transpose = DMatrix::from(&csc).transpose();
        let csc_transpose = csc.transpose();
        prop_assert_eq!(dense_transpose, DMatrix::from(&csc_transpose));
        prop_assert_eq!(csc.nnz(), csc_transpose.nnz());
    }
}