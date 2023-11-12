use nalgebra::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::{SparseEntry, SparseEntryMut, SparseFormatErrorKind};

use proptest::prelude::*;
use proptest::sample::subsequence;

use super::test_data_examples::{InvalidCsDataExamples, ValidCsDataExamples};

use crate::assert_panics;
use crate::common::csr_strategy;

use std::collections::HashSet;

#[test]
fn csr_matrix_default() {
    let matrix: CsrMatrix<f32> = CsrMatrix::default();

    assert_eq!(matrix.nrows(), 0);
    assert_eq!(matrix.ncols(), 0);
    assert_eq!(matrix.nnz(), 0);

    assert_eq!(matrix.values(), &[]);
    assert!(matrix.get_entry(0, 0).is_none());
}

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
fn csr_matrix_valid_data_unsorted_column_indices() {
    let valid_data: ValidCsDataExamples = ValidCsDataExamples::new();

    let (offsets, indices, values) = valid_data.valid_unsorted_cs_data;
    let csr = CsrMatrix::try_from_unsorted_csr_data(4, 5, offsets, indices, values).unwrap();

    let (offsets2, indices2, values2) = valid_data.valid_cs_data;
    let expected_csr = CsrMatrix::try_from_csr_data(4, 5, offsets2, indices2, values2).unwrap();

    assert_eq!(csr, expected_csr);
}

#[test]
fn csr_matrix_try_from_invalid_csr_data() {
    let invalid_data: InvalidCsDataExamples = InvalidCsDataExamples::new();
    {
        // Empty offset array (invalid length)
        let (offsets, indices, values) = invalid_data.empty_offset_array;
        let matrix = CsrMatrix::try_from_csr_data(0, 0, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let (offsets, indices, values) =
            invalid_data.offset_array_invalid_length_for_arbitrary_data;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid first entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_first_entry_in_offsets_array;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid last entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_last_entry_in_offsets_array;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid length of offsets array
        let (offsets, indices, values) = invalid_data.invalid_length_of_offsets_array;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic offsets
        let (offsets, indices, values) = invalid_data.nonmonotonic_offsets;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic minor indices
        let (offsets, indices, values) = invalid_data.nonmonotonic_minor_indices;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Minor index out of bounds
        let (offsets, indices, values) = invalid_data.minor_index_out_of_bounds;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Duplicate entry
        let (offsets, indices, values) = invalid_data.duplicate_entry;
        let matrix = CsrMatrix::try_from_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }
}

#[test]
fn csr_matrix_try_from_unsorted_invalid_csr_data() {
    let invalid_data: InvalidCsDataExamples = InvalidCsDataExamples::new();
    {
        // Empty offset array (invalid length)
        let (offsets, indices, values) = invalid_data.empty_offset_array;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(0, 0, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let (offsets, indices, values) =
            invalid_data.offset_array_invalid_length_for_arbitrary_data;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid first entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_first_entry_in_offsets_array;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid last entry in offsets array
        let (offsets, indices, values) = invalid_data.invalid_last_entry_in_offsets_array;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Invalid length of offsets array
        let (offsets, indices, values) = invalid_data.invalid_length_of_offsets_array;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Nonmonotonic offsets
        let (offsets, indices, values) = invalid_data.nonmonotonic_offsets;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
        );
    }

    {
        // Major offset out of bounds
        let (offsets, indices, values) = invalid_data.major_offset_out_of_bounds;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Minor index out of bounds
        let (offsets, indices, values) = invalid_data.minor_index_out_of_bounds;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::IndexOutOfBounds
        );
    }

    {
        // Duplicate entry
        let (offsets, indices, values) = invalid_data.duplicate_entry;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }

    {
        // Duplicate entry in unsorted lane
        let (offsets, indices, values) = invalid_data.duplicate_entry_unsorted;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(3, 6, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::DuplicateEntry
        );
    }

    {
        // Wrong values length
        let (offsets, indices, values) = invalid_data.wrong_values_length;
        let matrix = CsrMatrix::try_from_unsorted_csr_data(6, 3, offsets, indices, values);
        assert_eq!(
            matrix.unwrap_err().kind(),
            &SparseFormatErrorKind::InvalidStructure
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

// Rustfmt makes this test much harder to read by expanding some of the one-liners to 4-liners,
// so for now we skip rustfmt...
#[rustfmt::skip]
#[test]
fn csr_matrix_get_index_entry() {
    // Test .get_entry(_mut) and .index_entry(_mut) methods

    #[rustfmt::skip]
        let dense = DMatrix::from_row_slice(2, 3, &[
        1, 0, 3,
        0, 5, 6
    ]);
    let csr = CsrMatrix::from(&dense);

    assert_eq!(csr.get_entry(0, 0), Some(SparseEntry::NonZero(&1)));
    assert_eq!(csr.index_entry(0, 0), SparseEntry::NonZero(&1));
    assert_eq!(csr.get_entry(0, 1), Some(SparseEntry::Zero));
    assert_eq!(csr.index_entry(0, 1), SparseEntry::Zero);
    assert_eq!(csr.get_entry(0, 2), Some(SparseEntry::NonZero(&3)));
    assert_eq!(csr.index_entry(0, 2), SparseEntry::NonZero(&3));
    assert_eq!(csr.get_entry(1, 0), Some(SparseEntry::Zero));
    assert_eq!(csr.index_entry(1, 0), SparseEntry::Zero);
    assert_eq!(csr.get_entry(1, 1), Some(SparseEntry::NonZero(&5)));
    assert_eq!(csr.index_entry(1, 1), SparseEntry::NonZero(&5));
    assert_eq!(csr.get_entry(1, 2), Some(SparseEntry::NonZero(&6)));
    assert_eq!(csr.index_entry(1, 2), SparseEntry::NonZero(&6));

    // Check some out of bounds with .get_entry
    assert_eq!(csr.get_entry(0, 3), None);
    assert_eq!(csr.get_entry(0, 4), None);
    assert_eq!(csr.get_entry(1, 3), None);
    assert_eq!(csr.get_entry(1, 4), None);
    assert_eq!(csr.get_entry(2, 0), None);
    assert_eq!(csr.get_entry(2, 1), None);
    assert_eq!(csr.get_entry(2, 2), None);
    assert_eq!(csr.get_entry(2, 3), None);
    assert_eq!(csr.get_entry(2, 4), None);

    // Check that out of bounds with .index_entry panics
    assert_panics!(csr.index_entry(0, 3));
    assert_panics!(csr.index_entry(0, 4));
    assert_panics!(csr.index_entry(1, 3));
    assert_panics!(csr.index_entry(1, 4));
    assert_panics!(csr.index_entry(2, 0));
    assert_panics!(csr.index_entry(2, 1));
    assert_panics!(csr.index_entry(2, 2));
    assert_panics!(csr.index_entry(2, 3));
    assert_panics!(csr.index_entry(2, 4));

    {
        // Check mutable versions of the above functions
        let mut csr = csr;

        assert_eq!(csr.get_entry_mut(0, 0), Some(SparseEntryMut::NonZero(&mut 1)));
        assert_eq!(csr.index_entry_mut(0, 0), SparseEntryMut::NonZero(&mut 1));
        assert_eq!(csr.get_entry_mut(0, 1), Some(SparseEntryMut::Zero));
        assert_eq!(csr.index_entry_mut(0, 1), SparseEntryMut::Zero);
        assert_eq!(csr.get_entry_mut(0, 2), Some(SparseEntryMut::NonZero(&mut 3)));
        assert_eq!(csr.index_entry_mut(0, 2), SparseEntryMut::NonZero(&mut 3));
        assert_eq!(csr.get_entry_mut(1, 0), Some(SparseEntryMut::Zero));
        assert_eq!(csr.index_entry_mut(1, 0), SparseEntryMut::Zero);
        assert_eq!(csr.get_entry_mut(1, 1), Some(SparseEntryMut::NonZero(&mut 5)));
        assert_eq!(csr.index_entry_mut(1, 1), SparseEntryMut::NonZero(&mut 5));
        assert_eq!(csr.get_entry_mut(1, 2), Some(SparseEntryMut::NonZero(&mut 6)));
        assert_eq!(csr.index_entry_mut(1, 2), SparseEntryMut::NonZero(&mut 6));

        // Check some out of bounds with .get_entry_mut
        assert_eq!(csr.get_entry_mut(0, 3), None);
        assert_eq!(csr.get_entry_mut(0, 4), None);
        assert_eq!(csr.get_entry_mut(1, 3), None);
        assert_eq!(csr.get_entry_mut(1, 4), None);
        assert_eq!(csr.get_entry_mut(2, 0), None);
        assert_eq!(csr.get_entry_mut(2, 1), None);
        assert_eq!(csr.get_entry_mut(2, 2), None);
        assert_eq!(csr.get_entry_mut(2, 3), None);
        assert_eq!(csr.get_entry_mut(2, 4), None);

        // Check that out of bounds with .index_entry_mut panics
        // Note: the cloning is necessary because a mutable reference is not UnwindSafe
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(0, 3); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(0, 4); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(1, 3); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(1, 4); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(2, 0); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(2, 1); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(2, 2); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(2, 3); });
        assert_panics!({ let mut csr = csr.clone(); csr.index_entry_mut(2, 4); });
    }
}

#[test]
fn csr_matrix_row_iter() {
    #[rustfmt::skip]
    let dense = DMatrix::from_row_slice(3, 4, &[
        0, 1, 2, 0,
        3, 0, 0, 0,
        0, 4, 0, 5
    ]);
    let csr = CsrMatrix::from(&dense);

    // Immutable iterator
    {
        let mut row_iter = csr.row_iter();

        {
            let row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 2);
            assert_eq!(row.col_indices(), &[1, 2]);
            assert_eq!(row.values(), &[1, 2]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(1), Some(SparseEntry::NonZero(&1)));
            assert_eq!(row.get_entry(2), Some(SparseEntry::NonZero(&2)));
            assert_eq!(row.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(4), None);
        }

        {
            let row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 1);
            assert_eq!(row.col_indices(), &[0]);
            assert_eq!(row.values(), &[3]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::NonZero(&3)));
            assert_eq!(row.get_entry(1), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(4), None);
        }

        {
            let row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 2);
            assert_eq!(row.col_indices(), &[1, 3]);
            assert_eq!(row.values(), &[4, 5]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(1), Some(SparseEntry::NonZero(&4)));
            assert_eq!(row.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(3), Some(SparseEntry::NonZero(&5)));
            assert_eq!(row.get_entry(4), None);
        }

        assert!(row_iter.next().is_none());
    }

    // Mutable iterator
    {
        let mut csr = csr;
        let mut row_iter = csr.row_iter_mut();

        {
            let mut row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 2);
            assert_eq!(row.col_indices(), &[1, 2]);
            assert_eq!(row.values(), &[1, 2]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(1), Some(SparseEntry::NonZero(&1)));
            assert_eq!(row.get_entry(2), Some(SparseEntry::NonZero(&2)));
            assert_eq!(row.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(4), None);

            assert_eq!(row.values_mut(), &mut [1, 2]);
            assert_eq!(
                row.cols_and_values_mut(),
                ([1, 2].as_ref(), [1, 2].as_mut())
            );
            assert_eq!(row.get_entry_mut(0), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(1), Some(SparseEntryMut::NonZero(&mut 1)));
            assert_eq!(row.get_entry_mut(2), Some(SparseEntryMut::NonZero(&mut 2)));
            assert_eq!(row.get_entry_mut(3), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(4), None);
        }

        {
            let mut row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 1);
            assert_eq!(row.col_indices(), &[0]);
            assert_eq!(row.values(), &[3]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::NonZero(&3)));
            assert_eq!(row.get_entry(1), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(3), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(4), None);

            assert_eq!(row.values_mut(), &mut [3]);
            assert_eq!(row.cols_and_values_mut(), ([0].as_ref(), [3].as_mut()));
            assert_eq!(row.get_entry_mut(0), Some(SparseEntryMut::NonZero(&mut 3)));
            assert_eq!(row.get_entry_mut(1), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(2), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(3), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(4), None);
        }

        {
            let mut row = row_iter.next().unwrap();
            assert_eq!(row.ncols(), 4);
            assert_eq!(row.nnz(), 2);
            assert_eq!(row.col_indices(), &[1, 3]);
            assert_eq!(row.values(), &[4, 5]);
            assert_eq!(row.get_entry(0), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(1), Some(SparseEntry::NonZero(&4)));
            assert_eq!(row.get_entry(2), Some(SparseEntry::Zero));
            assert_eq!(row.get_entry(3), Some(SparseEntry::NonZero(&5)));
            assert_eq!(row.get_entry(4), None);

            assert_eq!(row.values_mut(), &mut [4, 5]);
            assert_eq!(
                row.cols_and_values_mut(),
                ([1, 3].as_ref(), [4, 5].as_mut())
            );
            assert_eq!(row.get_entry_mut(0), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(1), Some(SparseEntryMut::NonZero(&mut 4)));
            assert_eq!(row.get_entry_mut(2), Some(SparseEntryMut::Zero));
            assert_eq!(row.get_entry_mut(3), Some(SparseEntryMut::NonZero(&mut 5)));
            assert_eq!(row.get_entry_mut(4), None);
        }

        assert!(row_iter.next().is_none());
    }
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
