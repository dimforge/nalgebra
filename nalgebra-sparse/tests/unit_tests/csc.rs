use nalgebra::DMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::{SparseEntry, SparseEntryMut, SparseFormatErrorKind};

use proptest::prelude::*;
use proptest::sample::subsequence;

use super::test_data_examples::{InvalidCsDataExamples, ValidCsDataExamples};

use crate::assert_panics;
use crate::common::csc_strategy;

use std::collections::HashSet;

#[test]
fn csc_matrix_default() {
    let matrix: CscMatrix<f32> = CscMatrix::default();

    assert_eq!(matrix.nrows(), 0);
    assert_eq!(matrix.ncols(), 0);
    assert_eq!(matrix.nnz(), 0);

    assert_eq!(matrix.values(), &[]);
    assert!(matrix.get_entry(0, 0).is_none());
}

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

        assert_eq!(matrix, CscMatrix::zeros(2, 3));

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
        assert_eq!(
            matrix.col_mut(0).rows_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.col(1).nrows(), 2);
        assert_eq!(matrix.col(1).nnz(), 0);
        assert_eq!(matrix.col(1).row_indices(), &[]);
        assert_eq!(matrix.col(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).nrows(), 2);
        assert_eq!(matrix.col_mut(1).nnz(), 0);
        assert_eq!(matrix.col_mut(1).row_indices(), &[]);
        assert_eq!(matrix.col_mut(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).values_mut(), &[]);
        assert_eq!(
            matrix.col_mut(1).rows_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.col(2).nrows(), 2);
        assert_eq!(matrix.col(2).nnz(), 0);
        assert_eq!(matrix.col(2).row_indices(), &[]);
        assert_eq!(matrix.col(2).values(), &[]);
        assert_eq!(matrix.col_mut(2).nrows(), 2);
        assert_eq!(matrix.col_mut(2).nnz(), 0);
        assert_eq!(matrix.col_mut(2).row_indices(), &[]);
        assert_eq!(matrix.col_mut(2).values(), &[]);
        assert_eq!(matrix.col_mut(2).values_mut(), &[]);
        assert_eq!(
            matrix.col_mut(2).rows_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

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
        let mut matrix =
            CscMatrix::try_from_csc_data(6, 3, offsets.clone(), indices.clone(), values.clone())
                .unwrap();

        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(matrix.nnz(), 5);
        assert_eq!(matrix.col_offsets(), &[0, 2, 2, 5]);
        assert_eq!(matrix.row_indices(), &[0, 5, 1, 2, 3]);
        assert_eq!(matrix.values(), &[0, 1, 2, 3, 4]);

        let expected_triplets = vec![(0, 0, 0), (5, 0, 1), (1, 2, 2), (2, 2, 3), (3, 2, 4)];
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

        assert_eq!(matrix.col(0).nrows(), 6);
        assert_eq!(matrix.col(0).nnz(), 2);
        assert_eq!(matrix.col(0).row_indices(), &[0, 5]);
        assert_eq!(matrix.col(0).values(), &[0, 1]);
        assert_eq!(matrix.col_mut(0).nrows(), 6);
        assert_eq!(matrix.col_mut(0).nnz(), 2);
        assert_eq!(matrix.col_mut(0).row_indices(), &[0, 5]);
        assert_eq!(matrix.col_mut(0).values(), &[0, 1]);
        assert_eq!(matrix.col_mut(0).values_mut(), &[0, 1]);
        assert_eq!(
            matrix.col_mut(0).rows_and_values_mut(),
            ([0, 5].as_ref(), [0, 1].as_mut())
        );

        assert_eq!(matrix.col(1).nrows(), 6);
        assert_eq!(matrix.col(1).nnz(), 0);
        assert_eq!(matrix.col(1).row_indices(), &[]);
        assert_eq!(matrix.col(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).nrows(), 6);
        assert_eq!(matrix.col_mut(1).nnz(), 0);
        assert_eq!(matrix.col_mut(1).row_indices(), &[]);
        assert_eq!(matrix.col_mut(1).values(), &[]);
        assert_eq!(matrix.col_mut(1).values_mut(), &[]);
        assert_eq!(
            matrix.col_mut(1).rows_and_values_mut(),
            ([].as_ref(), [].as_mut())
        );

        assert_eq!(matrix.col(2).nrows(), 6);
        assert_eq!(matrix.col(2).nnz(), 3);
        assert_eq!(matrix.col(2).row_indices(), &[1, 2, 3]);
        assert_eq!(matrix.col(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.col_mut(2).nrows(), 6);
        assert_eq!(matrix.col_mut(2).nnz(), 3);
        assert_eq!(matrix.col_mut(2).row_indices(), &[1, 2, 3]);
        assert_eq!(matrix.col_mut(2).values(), &[2, 3, 4]);
        assert_eq!(matrix.col_mut(2).values_mut(), &[2, 3, 4]);
        assert_eq!(
            matrix.col_mut(2).rows_and_values_mut(),
            ([1, 2, 3].as_ref(), [2, 3, 4].as_mut())
        );

        assert!(matrix.get_col(3).is_none());
        assert!(matrix.get_col_mut(3).is_none());

        let (offsets2, indices2, values2) = matrix.disassemble();

        assert_eq!(offsets2, offsets);
        assert_eq!(indices2, indices);
        assert_eq!(values2, values);
    }
}

#[test]
fn csc_matrix_valid_data_unsorted_column_indices() {
    let valid_data: ValidCsDataExamples = ValidCsDataExamples::new();

    let (offsets, indices, values) = valid_data.valid_unsorted_cs_data;
    let csc = CscMatrix::try_from_unsorted_csc_data(5, 4, offsets, indices, values).unwrap();

    let (offsets2, indices2, values2) = valid_data.valid_cs_data;
    let expected_csc = CscMatrix::try_from_csc_data(5, 4, offsets2, indices2, values2).unwrap();

    assert_eq!(csc, expected_csc);
}

#[test]
fn csc_matrix_try_from_invalid_csc_data() {
    let invalid_data: InvalidCsDataExamples = InvalidCsDataExamples::new();
    {
        // Empty offset array (invalid length)
        let (offsets, indices, values) = invalid_data.empty_offset_array;
        let matrix = CscMatrix::try_from_csc_data(0, 0, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let (offsets, indices, values) =
            invalid_data.offset_array_invalid_length_for_arbitrary_data;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid first entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_first_entry_in_offsets_array;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid last entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_last_entry_in_offsets_array;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid length of offsets array
        let (offsets, indices, values) = invalid_data.invalid_length_of_offsets_array;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic offsets
        let (offsets, indices, values) = invalid_data.nonmonotonic_offsets;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
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
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Minor index out of bounds
        let (offsets, indices, values) = invalid_data.minor_index_out_of_bounds;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Duplicate entry
        let (offsets, indices, values) = invalid_data.duplicate_entry;
        let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }
}

#[test]
fn csc_matrix_try_from_unsorted_invalid_csc_data() {
    let invalid_data: InvalidCsDataExamples = InvalidCsDataExamples::new();
    {
        // Empty offset array (invalid length)
        let (offsets, indices, values) = invalid_data.empty_offset_array;
        let matrix = CscMatrix::try_from_unsorted_csc_data(0, 0, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let (offsets, indices, values) =
            invalid_data.offset_array_invalid_length_for_arbitrary_data;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid first entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_first_entry_in_offsets_array;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid last entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_last_entry_in_offsets_array;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid length of offsets array
        let (offsets, indices, values) = invalid_data.invalid_length_of_offsets_array;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic offsets
        let (offsets, indices, values) = invalid_data.nonmonotonic_offsets;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Major offset out of bounds
        let (offsets, indices, values) = invalid_data.major_offset_out_of_bounds;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Minor index out of bounds
        let (offsets, indices, values) = invalid_data.minor_index_out_of_bounds;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Duplicate entry
        let (offsets, indices, values) = invalid_data.duplicate_entry;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }

    {
        // Duplicate entry in unsorted lane
        let (offsets, indices, values) = invalid_data.duplicate_entry_unsorted;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }

    {
        // Wrong values length
        let (offsets, indices, values) = invalid_data.wrong_values_length;
        let matrix = CscMatrix::try_from_unsorted_csc_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
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
    let matrix = CscMatrix::try_from_csc_data(6, 3, offsets, indices, values).unwrap();

    let (offsets, indices, values) = matrix.disassemble();
    assert_eq!(offsets.as_ptr(), offsets_ptr);
    assert_eq!(indices.as_ptr(), indices_ptr);
    assert_eq!(values.as_ptr(), values_ptr);
}

// Rustfmt makes this test much harder to read by expanding some of the one-liners to 4-liners,
// so for now we skip rustfmt...
#[rustfmt::skip]
#[test]
fn csc_matrix_get_index_entry() {
    // Test .get_entry(_mut) and .index_entry(_mut) methods

    #[rustfmt::skip]
    let dense = DMatrix::from_row_slice(2, 3, &[
        1, 0, 3,
        0, 5, 6
    ]);
    let csc = CscMatrix::from(&dense);

    assert_eq!(csc.get_entry(0, 0), Some(SparseEntry::NonZero(&1)));
    assert_eq!(csc.index_entry(0, 0), SparseEntry::NonZero(&1));
    assert_eq!(csc.get_entry(0, 1), Some(SparseEntry::Zero));
    assert_eq!(csc.index_entry(0, 1), SparseEntry::Zero);
    assert_eq!(csc.get_entry(0, 2), Some(SparseEntry::NonZero(&3)));
    assert_eq!(csc.index_entry(0, 2), SparseEntry::NonZero(&3));
    assert_eq!(csc.get_entry(1, 0), Some(SparseEntry::Zero));
    assert_eq!(csc.index_entry(1, 0), SparseEntry::Zero);
    assert_eq!(csc.get_entry(1, 1), Some(SparseEntry::NonZero(&5)));
    assert_eq!(csc.index_entry(1, 1), SparseEntry::NonZero(&5));
    assert_eq!(csc.get_entry(1, 2), Some(SparseEntry::NonZero(&6)));
    assert_eq!(csc.index_entry(1, 2), SparseEntry::NonZero(&6));

    // Check some out of bounds with .get_entry
    assert_eq!(csc.get_entry(0, 3), None);
    assert_eq!(csc.get_entry(0, 4), None);
    assert_eq!(csc.get_entry(1, 3), None);
    assert_eq!(csc.get_entry(1, 4), None);
    assert_eq!(csc.get_entry(2, 0), None);
    assert_eq!(csc.get_entry(2, 1), None);
    assert_eq!(csc.get_entry(2, 2), None);
    assert_eq!(csc.get_entry(2, 3), None);
    assert_eq!(csc.get_entry(2, 4), None);

    // Check that out of bounds with .index_entry panics
    assert_panics!(csc.index_entry(0, 3));
    assert_panics!(csc.index_entry(0, 4));
    assert_panics!(csc.index_entry(1, 3));
    assert_panics!(csc.index_entry(1, 4));
    assert_panics!(csc.index_entry(2, 0));
    assert_panics!(csc.index_entry(2, 1));
    assert_panics!(csc.index_entry(2, 2));
    assert_panics!(csc.index_entry(2, 3));
    assert_panics!(csc.index_entry(2, 4));

    {
        // Check mutable versions of the above functions
        let mut csc = csc;

        assert_eq!(csc.get_entry_mut(0, 0), Some(SparseEntryMut::NonZero(&mut 1)));
        assert_eq!(csc.index_entry_mut(0, 0), SparseEntryMut::NonZero(&mut 1));
        assert_eq!(csc.get_entry_mut(0, 1), Some(SparseEntryMut::Zero));
        assert_eq!(csc.index_entry_mut(0, 1), SparseEntryMut::Zero);
        assert_eq!(csc.get_entry_mut(0, 2), Some(SparseEntryMut::NonZero(&mut 3)));
        assert_eq!(csc.index_entry_mut(0, 2), SparseEntryMut::NonZero(&mut 3));
        assert_eq!(csc.get_entry_mut(1, 0), Some(SparseEntryMut::Zero));
        assert_eq!(csc.index_entry_mut(1, 0), SparseEntryMut::Zero);
        assert_eq!(csc.get_entry_mut(1, 1), Some(SparseEntryMut::NonZero(&mut 5)));
        assert_eq!(csc.index_entry_mut(1, 1), SparseEntryMut::NonZero(&mut 5));
        assert_eq!(csc.get_entry_mut(1, 2), Some(SparseEntryMut::NonZero(&mut 6)));
        assert_eq!(csc.index_entry_mut(1, 2), SparseEntryMut::NonZero(&mut 6));

        // Check some out of bounds with .get_entry_mut
        assert_eq!(csc.get_entry_mut(0, 3), None);
        assert_eq!(csc.get_entry_mut(0, 4), None);
        assert_eq!(csc.get_entry_mut(1, 3), None);
        assert_eq!(csc.get_entry_mut(1, 4), None);
        assert_eq!(csc.get_entry_mut(2, 0), None);
        assert_eq!(csc.get_entry_mut(2, 1), None);
        assert_eq!(csc.get_entry_mut(2, 2), None);
        assert_eq!(csc.get_entry_mut(2, 3), None);
        assert_eq!(csc.get_entry_mut(2, 4), None);

        // Check that out of bounds with .index_entry_mut panics
        // Note: the cloning is necessary because a mutable reference is not UnwindSafe
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(0, 3); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(0, 4); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(1, 3); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(1, 4); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(2, 0); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(2, 1); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(2, 2); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(2, 3); });
        assert_panics!({ let mut csc = csc.clone(); csc.index_entry_mut(2, 4); });
    }
}

#[test]
fn csc_matrix_col_iter() {
    // Note: this is the transpose of the matrix used for the similar csr_matrix_row_iter test
    // (this way the actual tests are almost identical, due to the transposed relationship
    // between CSR and CSC)
    #[rustfmt::skip]
    let dense = DMatrix::from_row_slice(4, 3, &[
        0, 3, 0,
        1, 0, 4,
        2, 0, 0,
        0, 0, 5,
    ]);
    let csc = CscMatrix::from(&dense);

    // Immutable iterator
    {
        let mut col_iter = csc.col_iter();

        {
            let col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 2);
            assert_eq!(col.row_indices(), &[1, 2]);
            assert_eq!(col.values(), &[1, 2]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(1), Some(SparseEntry::NonZero(&1)));
            assert_eq!(col.get_entry(2), Some(SparseEntry::NonZero(&2)));
            assert_eq!(col.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(4), None);
        }

        {
            let col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 1);
            assert_eq!(col.row_indices(), &[0]);
            assert_eq!(col.values(), &[3]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::NonZero(&3)));
            assert_eq!(col.get_entry(1), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(4), None);
        }

        {
            let col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 2);
            assert_eq!(col.row_indices(), &[1, 3]);
            assert_eq!(col.values(), &[4, 5]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(1), Some(SparseEntry::NonZero(&4)));
            assert_eq!(col.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(3), Some(SparseEntry::NonZero(&5)));
            assert_eq!(col.get_entry(4), None);
        }

        assert!(col_iter.next().is_none());
    }

    // Mutable iterator
    {
        let mut csc = csc;
        let mut col_iter = csc.col_iter_mut();

        {
            let mut col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 2);
            assert_eq!(col.row_indices(), &[1, 2]);
            assert_eq!(col.values(), &[1, 2]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(1), Some(SparseEntry::NonZero(&1)));
            assert_eq!(col.get_entry(2), Some(SparseEntry::NonZero(&2)));
            assert_eq!(col.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(4), None);

            assert_eq!(col.values_mut(), &mut [1, 2]);
            assert_eq!(
                col.rows_and_values_mut(),
                ([1, 2].as_ref(), [1, 2].as_mut())
            );
            assert_eq!(col.get_entry_mut(0), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(1), Some(SparseEntryMut::NonZero(&mut 1)));
            assert_eq!(col.get_entry_mut(2), Some(SparseEntryMut::NonZero(&mut 2)));
            assert_eq!(col.get_entry_mut(3), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(4), None);
        }

        {
            let mut col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 1);
            assert_eq!(col.row_indices(), &[0]);
            assert_eq!(col.values(), &[3]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::NonZero(&3)));
            assert_eq!(col.get_entry(1), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(4), None);

            assert_eq!(col.values_mut(), &mut [3]);
            assert_eq!(col.rows_and_values_mut(), ([0].as_ref(), [3].as_mut()));
            assert_eq!(col.get_entry_mut(0), Some(SparseEntryMut::NonZero(&mut 3)));
            assert_eq!(col.get_entry_mut(1), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(2), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(3), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(4), None);
        }

        {
            let mut col = col_iter.next().unwrap();
            assert_eq!(col.nrows(), 4);
            assert_eq!(col.nnz(), 2);
            assert_eq!(col.row_indices(), &[1, 3]);
            assert_eq!(col.values(), &[4, 5]);
            assert_eq!(col.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(1), Some(SparseEntry::NonZero(&4)));
            assert_eq!(col.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(col.get_entry(3), Some(SparseEntry::NonZero(&5)));
            assert_eq!(col.get_entry(4), None);

            assert_eq!(col.values_mut(), &mut [4, 5]);
            assert_eq!(
                col.rows_and_values_mut(),
                ([1, 3].as_ref(), [4, 5].as_mut())
            );
            assert_eq!(col.get_entry_mut(0), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(1), Some(SparseEntryMut::NonZero(&mut 4)));
            assert_eq!(col.get_entry_mut(2), Some(SparseEntryMut::Zero));
            assert_eq!(col.get_entry_mut(3), Some(SparseEntryMut::NonZero(&mut 5)));
            assert_eq!(col.get_entry_mut(4), None);
        }

        assert!(col_iter.next().is_none());
    }
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

    #[test]
    fn csc_filter(
        (csc, triplet_subset)
        in csc_strategy()
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
        // We generate a CscMatrix and a HashSet corresponding to a subset of the (i, j, v)
        // values in the matrix, which we use for filtering the matrix entries.
        // The resulting triplets in the filtered matrix must then be exactly equal to
        // the subset.
        let filtered = csc.filter(|i, j, v| triplet_subset.contains(&(i, j, *v)));
        let filtered_triplets: HashSet<_> = filtered
            .triplet_iter()
            .cloned_values()
            .collect();

        prop_assert_eq!(filtered_triplets, triplet_subset);
    }

    #[test]
    fn csc_lower_triangle_agrees_with_dense(csc in csc_strategy()) {
        let csc_lower_triangle = csc.lower_triangle();
        prop_assert_eq!(DMatrix::from(&csc_lower_triangle), DMatrix::from(&csc).lower_triangle());
        prop_assert!(csc_lower_triangle.nnz() <= csc.nnz());
    }

    #[test]
    fn csc_upper_triangle_agrees_with_dense(csc in csc_strategy()) {
        let csc_upper_triangle = csc.upper_triangle();
        prop_assert_eq!(DMatrix::from(&csc_upper_triangle), DMatrix::from(&csc).upper_triangle());
        prop_assert!(csc_upper_triangle.nnz() <= csc.nnz());
    }

    #[test]
    fn csc_diagonal_as_csc(csc in csc_strategy()) {
        let d = csc.diagonal_as_csc();
        let d_entries: HashSet<_> = d.triplet_iter().cloned_values().collect();
        let csc_diagonal_entries: HashSet<_> = csc
            .triplet_iter()
            .cloned_values()
            .filter(|&(i, j, _)| i == j)
            .collect();

        prop_assert_eq!(d_entries, csc_diagonal_entries);
    }

    #[test]
    fn csc_identity(n in 0 ..= 6usize) {
        let csc = CscMatrix::<i32>::identity(n);
        prop_assert_eq!(csc.nnz(), n);
        prop_assert_eq!(DMatrix::from(&csc), DMatrix::identity(n, n));
    }
}
