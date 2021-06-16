use crate::assert_panics;
use nalgebra::DMatrix;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::SparseFormatErrorKind;

#[test]
fn coo_construction_for_valid_data() {
    // Test that construction with try_from_triplets succeeds, that the state of the
    // matrix afterwards is as expected, and that the dense representation matches expectations.

    {
        // Zero matrix
        let coo =
            CooMatrix::<i32>::try_from_triplets(3, 2, Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.ncols(), 2);
        assert!(coo.triplet_iter().next().is_none());
        assert!(coo.row_indices().is_empty());
        assert!(coo.col_indices().is_empty());
        assert!(coo.values().is_empty());

        assert_eq!(DMatrix::from(&coo), DMatrix::repeat(3, 2, 0));
    }

    {
        // Arbitrary matrix, no duplicates
        let i = vec![0, 1, 0, 0, 2];
        let j = vec![0, 2, 1, 3, 3];
        let v = vec![2, 3, 7, 3, 1];
        let coo =
            CooMatrix::<i32>::try_from_triplets(3, 5, i.clone(), j.clone(), v.clone()).unwrap();
        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.ncols(), 5);

        assert_eq!(i.as_slice(), coo.row_indices());
        assert_eq!(j.as_slice(), coo.col_indices());
        assert_eq!(v.as_slice(), coo.values());

        let expected_triplets: Vec<_> = i
            .iter()
            .zip(&j)
            .zip(&v)
            .map(|((i, j), v)| (*i, *j, *v))
            .collect();
        let actual_triplets: Vec<_> = coo.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
        assert_eq!(actual_triplets, expected_triplets);

        #[rustfmt::skip]
        let expected_dense = DMatrix::from_row_slice(3, 5, &[
            2, 7, 0, 3, 0,
            0, 0, 3, 0, 0,
            0, 0, 0, 1, 0
        ]);
        assert_eq!(DMatrix::from(&coo), expected_dense);
    }

    {
        // Arbitrary matrix, with duplicates
        let i = vec![0, 1, 0, 0, 0, 0, 2, 1];
        let j = vec![0, 2, 0, 1, 0, 3, 3, 2];
        let v = vec![2, 3, 4, 7, 1, 3, 1, 5];
        let coo =
            CooMatrix::<i32>::try_from_triplets(3, 5, i.clone(), j.clone(), v.clone()).unwrap();
        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.ncols(), 5);

        assert_eq!(i.as_slice(), coo.row_indices());
        assert_eq!(j.as_slice(), coo.col_indices());
        assert_eq!(v.as_slice(), coo.values());

        let expected_triplets: Vec<_> = i
            .iter()
            .zip(&j)
            .zip(&v)
            .map(|((i, j), v)| (*i, *j, *v))
            .collect();
        let actual_triplets: Vec<_> = coo.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
        assert_eq!(actual_triplets, expected_triplets);

        #[rustfmt::skip]
            let expected_dense = DMatrix::from_row_slice(3, 5, &[
            7, 7, 0, 3, 0,
            0, 0, 8, 0, 0,
            0, 0, 0, 1, 0
        ]);
        assert_eq!(DMatrix::from(&coo), expected_dense);
    }
}

#[test]
fn coo_try_from_triplets_reports_out_of_bounds_indices() {
    {
        // 0x0 matrix
        let result = CooMatrix::<i32>::try_from_triplets(0, 0, vec![0], vec![0], vec![2]);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }

    {
        // 1x1 matrix, row out of bounds
        let result = CooMatrix::<i32>::try_from_triplets(1, 1, vec![1], vec![0], vec![2]);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }

    {
        // 1x1 matrix, col out of bounds
        let result = CooMatrix::<i32>::try_from_triplets(1, 1, vec![0], vec![1], vec![2]);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }

    {
        // 1x1 matrix, row and col out of bounds
        let result = CooMatrix::<i32>::try_from_triplets(1, 1, vec![1], vec![1], vec![2]);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }

    {
        // Arbitrary matrix, row out of bounds
        let i = vec![0, 1, 0, 3, 2];
        let j = vec![0, 2, 1, 3, 3];
        let v = vec![2, 3, 7, 3, 1];
        let result = CooMatrix::<i32>::try_from_triplets(3, 5, i, j, v);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }

    {
        // Arbitrary matrix, col out of bounds
        let i = vec![0, 1, 0, 0, 2];
        let j = vec![0, 2, 1, 5, 3];
        let v = vec![2, 3, 7, 3, 1];
        let result = CooMatrix::<i32>::try_from_triplets(3, 5, i, j, v);
        assert!(matches!(
            result.unwrap_err().kind(),
            SparseFormatErrorKind::IndexOutOfBounds
        ));
    }
}

#[test]
fn coo_try_from_triplets_panics_on_mismatched_vectors() {
    // Check that try_from_triplets panics when the triplet vectors have different lengths
    macro_rules! assert_errs {
        ($result:expr) => {
            assert!(matches!(
                $result.unwrap_err().kind(),
                SparseFormatErrorKind::InvalidStructure
            ))
        };
    }

    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1, 2],
        vec![0],
        vec![0]
    ));
    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1],
        vec![0, 0],
        vec![0]
    ));
    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1],
        vec![0],
        vec![0, 1]
    ));
    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1, 2],
        vec![0, 1],
        vec![0]
    ));
    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1],
        vec![0, 1],
        vec![0, 1]
    ));
    assert_errs!(CooMatrix::<i32>::try_from_triplets(
        3,
        5,
        vec![1, 1],
        vec![0],
        vec![0, 1]
    ));
}

#[test]
fn coo_push_valid_entries() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    assert_eq!(coo.triplet_iter().collect::<Vec<_>>(), vec![(0, 0, &1)]);

    coo.push(0, 0, 2);
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![(0, 0, &1), (0, 0, &2)]
    );

    coo.push(2, 2, 3);
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![(0, 0, &1), (0, 0, &2), (2, 2, &3)]
    );
}

#[test]
fn coo_push_out_of_bounds_entries() {
    {
        // 0x0 matrix
        let coo = CooMatrix::new(0, 0);
        assert_panics!(coo.clone().push(0, 0, 1));
    }

    {
        // 0x1 matrix
        assert_panics!(CooMatrix::new(0, 1).push(0, 0, 1));
    }

    {
        // 1x0 matrix
        assert_panics!(CooMatrix::new(1, 0).push(0, 0, 1));
    }

    {
        // Arbitrary matrix dimensions
        let coo = CooMatrix::new(3, 2);
        assert_panics!(coo.clone().push(3, 0, 1));
        assert_panics!(coo.clone().push(2, 2, 1));
        assert_panics!(coo.clone().push(3, 2, 1));
    }
}

#[test]
fn coo_push_matrix_valid_entries() {
    let mut coo = CooMatrix::new(3, 3);

    // Works with static
    {
        // new is row-major...
        let inserted = nalgebra::SMatrix::<i32, 2, 2>::new(1, 2, 3, 4);
        coo.push_matrix(1, 1, &inserted);

        // insert happens column-major, so expect transposition when read this way
        assert_eq!(
            coo.triplet_iter().collect::<Vec<_>>(),
            vec![(1, 1, &1), (2, 1, &3), (1, 2, &2), (2, 2, &4)]
        );
    }

    // Works with owned dynamic
    {
        let inserted = nalgebra::DMatrix::<i32>::repeat(1, 2, 5);
        coo.push_matrix(0, 0, &inserted);

        assert_eq!(
            coo.triplet_iter().collect::<Vec<_>>(),
            vec![
                (1, 1, &1),
                (2, 1, &3),
                (1, 2, &2),
                (2, 2, &4),
                (0, 0, &5),
                (0, 1, &5)
            ]
        );
    }

    // Works with sliced
    {
        let source = nalgebra::SMatrix::<i32, 2, 2>::new(6, 7, 8, 9);
        let sliced = source.fixed_slice::<2, 1>(0, 0);
        coo.push_matrix(1, 0, &sliced);

        assert_eq!(
            coo.triplet_iter().collect::<Vec<_>>(),
            vec![
                (1, 1, &1),
                (2, 1, &3),
                (1, 2, &2),
                (2, 2, &4),
                (0, 0, &5),
                (0, 1, &5),
                (1, 0, &6),
                (2, 0, &8)
            ]
        );
    }
}

#[test]
fn coo_push_matrix_out_of_bounds_entries() {
    // 0x0
    {
        let inserted = nalgebra::SMatrix::<i32, 1, 1>::new(1);
        assert_panics!(CooMatrix::new(0, 0).push_matrix(0, 0, &inserted));
    }
    // 0x1
    {
        let inserted = nalgebra::SMatrix::<i32, 1, 1>::new(1);
        assert_panics!(CooMatrix::new(1, 0).push_matrix(0, 0, &inserted));
    }
    // 1x0
    {
        let inserted = nalgebra::SMatrix::<i32, 1, 1>::new(1);
        assert_panics!(CooMatrix::new(0, 1).push_matrix(0, 0, &inserted));
    }

    // 3x3 exceeds col-dim
    {
        let inserted = nalgebra::SMatrix::<i32, 1, 2>::repeat(1);
        assert_panics!(CooMatrix::new(3, 3).push_matrix(0, 2, &inserted));
    }
    // 3x3 exceeds row-dim
    {
        let inserted = nalgebra::SMatrix::<i32, 2, 1>::repeat(1);
        assert_panics!(CooMatrix::new(3, 3).push_matrix(2, 0, &inserted));
    }
    // 3x3 exceeds row-dim and row-dim
    {
        let inserted = nalgebra::SMatrix::<i32, 2, 2>::repeat(1);
        assert_panics!(CooMatrix::new(3, 3).push_matrix(2, 2, &inserted));
    }
}
