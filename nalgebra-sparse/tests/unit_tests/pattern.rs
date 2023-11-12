use nalgebra_sparse::pattern::{SparsityPattern, SparsityPatternFormatError};

#[test]
fn sparsity_pattern_default() {
    // Check that the pattern created with `Default::default()` is equivalent to a zero-sized pattern.
    let pattern = SparsityPattern::default();
    let zero = SparsityPattern::zeros(0, 0);

    assert_eq!(pattern.major_dim(), zero.major_dim());
    assert_eq!(pattern.minor_dim(), zero.minor_dim());
    assert_eq!(pattern.major_offsets(), zero.major_offsets());
    assert_eq!(pattern.minor_indices(), zero.minor_indices());

    assert_eq!(pattern.nnz(), 0);
}

#[test]
fn sparsity_pattern_valid_data() {
    // Construct pattern from valid data and check that selected methods return results
    // that agree with expectations.

    {
        // A pattern with zero explicitly stored entries
        let pattern =
            SparsityPattern::try_from_offsets_and_indices(3, 2, vec![0, 0, 0, 0], Vec::new())
                .unwrap();

        assert_eq!(pattern.major_dim(), 3);
        assert_eq!(pattern.minor_dim(), 2);
        assert_eq!(pattern.nnz(), 0);
        assert_eq!(pattern.major_offsets(), &[0, 0, 0, 0]);
        assert_eq!(pattern.minor_indices(), &[]);
        assert_eq!(pattern.lane(0), &[]);
        assert_eq!(pattern.lane(1), &[]);
        assert_eq!(pattern.lane(2), &[]);
        assert!(pattern.entries().next().is_none());

        assert_eq!(pattern, SparsityPattern::zeros(3, 2));

        let (offsets, indices) = pattern.disassemble();
        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert_eq!(indices, vec![]);
    }

    {
        // Arbitrary pattern
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let pattern =
            SparsityPattern::try_from_offsets_and_indices(3, 6, offsets.clone(), indices.clone())
                .unwrap();

        assert_eq!(pattern.major_dim(), 3);
        assert_eq!(pattern.minor_dim(), 6);
        assert_eq!(pattern.major_offsets(), offsets.as_slice());
        assert_eq!(pattern.minor_indices(), indices.as_slice());
        assert_eq!(pattern.nnz(), 5);
        assert_eq!(pattern.lane(0), &[0, 5]);
        assert_eq!(pattern.lane(1), &[]);
        assert_eq!(pattern.lane(2), &[1, 2, 3]);
        assert_eq!(
            pattern.entries().collect::<Vec<_>>(),
            vec![(0, 0), (0, 5), (2, 1), (2, 2), (2, 3)]
        );

        let (offsets2, indices2) = pattern.disassemble();
        assert_eq!(offsets2, offsets);
        assert_eq!(indices2, indices);
    }
}

#[test]
fn sparsity_pattern_try_from_invalid_data() {
    {
        // Empty offset array (invalid length)
        let pattern = SparsityPattern::try_from_offsets_and_indices(0, 0, Vec::new(), Vec::new());
        assert_eq!(
            pattern,
            Err(SparsityPatternFormatError::InvalidOffsetArrayLength)
        );
    }

    {
        // Offset array invalid length for arbitrary data
        let offsets = vec![0, 3, 5];
        let indices = vec![0, 1, 2, 3, 5];

        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert!(matches!(
            pattern,
            Err(SparsityPatternFormatError::InvalidOffsetArrayLength)
        ));
    }

    {
        // Invalid first entry in offsets array
        let offsets = vec![1, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert!(matches!(
            pattern,
            Err(SparsityPatternFormatError::InvalidOffsetFirstLast)
        ));
    }

    {
        // Invalid last entry in offsets array
        let offsets = vec![0, 2, 2, 4];
        let indices = vec![0, 5, 1, 2, 3];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert!(matches!(
            pattern,
            Err(SparsityPatternFormatError::InvalidOffsetFirstLast)
        ));
    }

    {
        // Invalid length of offsets array
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 5, 1, 2, 3];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert!(matches!(
            pattern,
            Err(SparsityPatternFormatError::InvalidOffsetArrayLength)
        ));
    }

    {
        // Nonmonotonic offsets
        let offsets = vec![0, 3, 2, 5];
        let indices = vec![0, 1, 2, 3, 4];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert_eq!(
            pattern,
            Err(SparsityPatternFormatError::NonmonotonicOffsets)
        );
    }

    {
        // Nonmonotonic minor indices
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 2, 3, 1, 4];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert_eq!(
            pattern,
            Err(SparsityPatternFormatError::NonmonotonicMinorIndices)
        );
    }

    {
        // Minor index out of bounds
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 6, 1, 2, 3];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert_eq!(
            pattern,
            Err(SparsityPatternFormatError::MinorIndexOutOfBounds)
        );
    }

    {
        // Duplicate entry
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 2, 2, 3];
        let pattern = SparsityPattern::try_from_offsets_and_indices(3, 6, offsets, indices);
        assert_eq!(pattern, Err(SparsityPatternFormatError::DuplicateEntry));
    }
}
