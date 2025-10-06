use crate::assert_panics;
use nalgebra::DMatrix;
use nalgebra_sparse::SparseFormatErrorKind;
use nalgebra_sparse::coo::CooMatrix;

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
fn coo_triplets_iter_mut() {
    // Arbitrary matrix, with duplicates
    let i = vec![0, 1, 0, 0, 0, 0, 2, 1];
    let j = vec![0, 2, 0, 1, 0, 3, 3, 2];
    let v = vec![2, 3, 4, 7, 1, 3, 1, 5];
    let mut coo =
        CooMatrix::<i32>::try_from_triplets(3, 5, i.clone(), j.clone(), v.clone()).unwrap();

    let actual_triplets: Vec<_> = coo.triplet_iter_mut().map(|(i, j, v)| (i, j, *v)).collect();

    let expected_triplets: Vec<_> = i
        .iter()
        .zip(&j)
        .zip(&v)
        .map(|((i, j), v)| (*i, *j, *v))
        .collect();
    assert_eq!(expected_triplets, actual_triplets);

    for (_i, _j, v) in coo.triplet_iter_mut() {
        *v += *v;
    }

    let actual_triplets: Vec<_> = coo.triplet_iter_mut().map(|(i, j, v)| (i, j, *v)).collect();
    let v = vec![4, 6, 8, 14, 2, 6, 2, 10];
    let expected_triplets: Vec<_> = i
        .iter()
        .zip(&j)
        .zip(&v)
        .map(|((i, j), v)| (*i, *j, *v))
        .collect();
    assert_eq!(expected_triplets, actual_triplets);
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
fn coo_try_from_triplets_iter() {
    // Check that try_from_triplets_iter panics when the triplet vectors have different lengths
    macro_rules! assert_errs {
        ($result:expr) => {
            assert!(matches!(
                $result.unwrap_err().kind(),
                SparseFormatErrorKind::IndexOutOfBounds
            ))
        };
    }

    assert_errs!(CooMatrix::<f32>::try_from_triplets_iter(
        3,
        5,
        vec![(0, 6, 3.0)].into_iter(),
    ));
    assert!(
        CooMatrix::<f32>::try_from_triplets_iter(
            3,
            5,
            vec![(0, 3, 3.0), (1, 2, 2.0), (0, 3, 1.0),].into_iter(),
        )
        .is_ok()
    );
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
fn coo_clear_triplets_valid_entries() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 0, 2);
    coo.push(2, 2, 3);
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![(0, 0, &1), (0, 0, &2), (2, 2, &3)]
    );
    coo.clear_triplets();
    assert_eq!(coo.triplet_iter().collect::<Vec<_>>(), vec![]);
    // making sure everything works after clearing
    coo.push(0, 0, 1);
    coo.push(0, 0, 2);
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
        let view = source.fixed_view::<2, 1>(0, 0);
        coo.push_matrix(1, 0, &view);

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

#[test]
fn coo_remove_row_valid() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 1, 2);
    coo.push(0, 2, 3);

    let mut removed_coo = coo.remove_row(0);
    assert_eq!(removed_coo.nrows(), 2);
    assert_eq!(removed_coo.ncols(), 3);
    assert_eq!(removed_coo.nnz(), 0);

    // makes sure resulting COO matrix still works. This will push to the new
    // matrices row 0.
    removed_coo.push(0, 0, 1);
    removed_coo.push(0, 1, 2);
    removed_coo.push(0, 2, 3);
    assert_eq!(removed_coo.nnz(), 3);

    assert_panics!(removed_coo.clone().push(2, 0, 4));

    // makes sure original matrix is untouched.
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![(0, 0, &1), (0, 1, &2), (0, 2, &3)]
    );
}

#[test]
fn coo_remove_row_out_of_bounds() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 1, 2);
    coo.push(0, 2, 3);

    // Push past col-dim
    assert_panics!(coo.clone().remove_row(3));
}

#[test]
fn coo_remove_column_valid() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 1, 2);
    coo.push(0, 2, 3);

    let mut removed_coo = coo.remove_column(1);
    assert_eq!(removed_coo.ncols(), 2);
    assert_eq!(removed_coo.nrows(), 3);
    assert_eq!(removed_coo.nnz(), 2);

    // makes sure resulting COO matrix still works.
    removed_coo.push(0, 1, 2);
    removed_coo.push(2, 1, 4);
    assert_eq!(removed_coo.nnz(), 4);

    assert_panics!(removed_coo.clone().push(0, 2, 4));

    // makes sure original matrix is untouched.
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![(0, 0, &1), (0, 1, &2), (0, 2, &3)]
    );
}

#[test]
fn coo_remove_column_out_of_bounds() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 1, 2);
    coo.push(0, 2, 3);

    // Push past col-dim
    assert_panics!(coo.clone().remove_column(3));
}

#[test]
fn coo_remove_row_column_valid() {
    let mut coo = CooMatrix::new(3, 3);

    coo.push(0, 0, 1);
    coo.push(0, 1, 2);
    coo.push(0, 2, 3);
    coo.push(1, 0, 4);
    coo.push(1, 1, 5);
    coo.push(1, 2, 6);
    coo.push(2, 0, 7);
    coo.push(2, 1, 8);
    coo.push(2, 2, 9);

    let mut removed_coo = coo.remove_row_column(1, 1);
    assert_eq!(removed_coo.ncols(), 2);
    assert_eq!(removed_coo.nrows(), 2);
    assert_eq!(removed_coo.nnz(), 4);

    // makes sure resulting COO matrix still works.
    removed_coo.push(0, 0, 1);
    removed_coo.push(0, 1, 0);
    removed_coo.push(1, 0, 0);
    removed_coo.push(1, 1, 4);
    assert_eq!(removed_coo.nnz(), 8);

    assert_panics!(removed_coo.clone().push(0, 2, 4));
    assert_panics!(removed_coo.clone().push(2, 1, 4));

    // makes sure original matrix is untouched.
    assert_eq!(
        coo.triplet_iter().collect::<Vec<_>>(),
        vec![
            (0, 0, &1),
            (0, 1, &2),
            (0, 2, &3),
            (1, 0, &4),
            (1, 1, &5),
            (1, 2, &6),
            (2, 0, &7),
            (2, 1, &8),
            (2, 2, &9)
        ]
    );
}
