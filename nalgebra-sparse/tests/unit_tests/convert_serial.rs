use crate::common::csc_strategy;
use nalgebra::DMatrix;
use nalgebra::proptest::matrix;
use nalgebra_sparse::convert::serial::{
    convert_coo_csc, convert_coo_csr, convert_coo_dense, convert_csc_coo, convert_csc_csr,
    convert_csc_dense, convert_csr_coo, convert_csr_csc, convert_csr_dense, convert_dense_coo,
    convert_dense_csc, convert_dense_csr,
};
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::proptest::{coo_no_duplicates, coo_with_duplicates, csc, csr};
use proptest::prelude::*;

#[test]
fn test_convert_dense_coo() {
    // No duplicates
    {
        #[rustfmt::skip]
        let entries = &[1, 0, 3,
                        0, 5, 0];
        // The COO representation of a dense matrix is not unique.
        // Here we implicitly test that the coo matrix is indeed constructed from column-major
        // iteration of the dense matrix.
        let dense = DMatrix::from_row_slice(2, 3, entries);
        let coo = CooMatrix::try_from_triplets(2, 3, vec![0, 1, 0], vec![0, 1, 2], vec![1, 5, 3])
            .unwrap();

        assert_eq!(CooMatrix::from(&dense), coo);
        assert_eq!(DMatrix::from(&coo), dense);
    }

    // Duplicates
    // No duplicates
    {
        #[rustfmt::skip]
        let entries = &[1, 0, 3,
            0, 5, 0];
        // The COO representation of a dense matrix is not unique.
        // Here we implicitly test that the coo matrix is indeed constructed from column-major
        // iteration of the dense matrix.
        let dense = DMatrix::from_row_slice(2, 3, entries);
        let coo_no_dup =
            CooMatrix::try_from_triplets(2, 3, vec![0, 1, 0], vec![0, 1, 2], vec![1, 5, 3])
                .unwrap();
        let coo_dup = CooMatrix::try_from_triplets(
            2,
            3,
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 1],
            vec![1, -2, 3, 7],
        )
        .unwrap();

        assert_eq!(CooMatrix::from(&dense), coo_no_dup);
        assert_eq!(DMatrix::from(&coo_dup), dense);
    }
}

#[test]
fn test_convert_coo_csr() {
    // No duplicates
    {
        let coo = {
            let mut coo = CooMatrix::new(3, 4);
            coo.push(1, 3, 4);
            coo.push(0, 1, 2);
            coo.push(2, 0, 1);
            coo.push(2, 3, 2);
            coo.push(2, 2, 1);
            coo
        };

        let expected_csr = CsrMatrix::try_from_csr_data(
            3,
            4,
            vec![0, 1, 2, 5],
            vec![1, 3, 0, 2, 3],
            vec![2, 4, 1, 1, 2],
        )
        .unwrap();

        assert_eq!(convert_coo_csr(&coo), expected_csr);
    }

    // Duplicates
    {
        let coo = {
            let mut coo = CooMatrix::new(3, 4);
            coo.push(1, 3, 4);
            coo.push(2, 3, 2);
            coo.push(0, 1, 2);
            coo.push(2, 0, 1);
            coo.push(2, 3, 2);
            coo.push(0, 1, 3);
            coo.push(2, 2, 1);
            coo
        };

        let expected_csr = CsrMatrix::try_from_csr_data(
            3,
            4,
            vec![0, 1, 2, 5],
            vec![1, 3, 0, 2, 3],
            vec![5, 4, 1, 1, 4],
        )
        .unwrap();

        assert_eq!(convert_coo_csr(&coo), expected_csr);
    }
}

#[test]
fn test_convert_csr_coo() {
    let csr = CsrMatrix::try_from_csr_data(
        3,
        4,
        vec![0, 1, 2, 5],
        vec![1, 3, 0, 2, 3],
        vec![5, 4, 1, 1, 4],
    )
    .unwrap();

    let expected_coo = CooMatrix::try_from_triplets(
        3,
        4,
        vec![0, 1, 2, 2, 2],
        vec![1, 3, 0, 2, 3],
        vec![5, 4, 1, 1, 4],
    )
    .unwrap();

    assert_eq!(convert_csr_coo(&csr), expected_coo);
}

#[test]
fn test_convert_coo_csc() {
    // No duplicates
    {
        let coo = {
            let mut coo = CooMatrix::new(3, 4);
            coo.push(1, 3, 4);
            coo.push(0, 1, 2);
            coo.push(2, 0, 1);
            coo.push(2, 3, 2);
            coo.push(2, 2, 1);
            coo
        };

        let expected_csc = CscMatrix::try_from_csc_data(
            3,
            4,
            vec![0, 1, 2, 3, 5],
            vec![2, 0, 2, 1, 2],
            vec![1, 2, 1, 4, 2],
        )
        .unwrap();

        assert_eq!(convert_coo_csc(&coo), expected_csc);
    }

    // Duplicates
    {
        let coo = {
            let mut coo = CooMatrix::new(3, 4);
            coo.push(1, 3, 4);
            coo.push(2, 3, 2);
            coo.push(0, 1, 2);
            coo.push(2, 0, 1);
            coo.push(2, 3, 2);
            coo.push(0, 1, 3);
            coo.push(2, 2, 1);
            coo
        };

        let expected_csc = CscMatrix::try_from_csc_data(
            3,
            4,
            vec![0, 1, 2, 3, 5],
            vec![2, 0, 2, 1, 2],
            vec![1, 5, 1, 4, 4],
        )
        .unwrap();

        assert_eq!(convert_coo_csc(&coo), expected_csc);
    }
}

#[test]
fn test_convert_csc_coo() {
    let csc = CscMatrix::try_from_csc_data(
        3,
        4,
        vec![0, 1, 2, 3, 5],
        vec![2, 0, 2, 1, 2],
        vec![1, 2, 1, 4, 2],
    )
    .unwrap();

    let expected_coo = CooMatrix::try_from_triplets(
        3,
        4,
        vec![2, 0, 2, 1, 2],
        vec![0, 1, 2, 3, 3],
        vec![1, 2, 1, 4, 2],
    )
    .unwrap();

    assert_eq!(convert_csc_coo(&csc), expected_coo);
}

#[test]
fn test_convert_csr_csc_bidirectional() {
    let csr = CsrMatrix::try_from_csr_data(
        3,
        4,
        vec![0, 3, 4, 6],
        vec![1, 2, 3, 0, 1, 3],
        vec![5, 3, 2, 2, 1, 4],
    )
    .unwrap();

    let csc = CscMatrix::try_from_csc_data(
        3,
        4,
        vec![0, 1, 3, 4, 6],
        vec![1, 0, 2, 0, 0, 2],
        vec![2, 5, 1, 3, 2, 4],
    )
    .unwrap();

    assert_eq!(convert_csr_csc(&csr), csc);
    assert_eq!(convert_csc_csr(&csc), csr);
}

#[test]
fn test_convert_csr_dense_bidirectional() {
    let csr = CsrMatrix::try_from_csr_data(
        3,
        4,
        vec![0, 3, 4, 6],
        vec![1, 2, 3, 0, 1, 3],
        vec![5, 3, 2, 2, 1, 4],
    )
    .unwrap();

    #[rustfmt::skip]
    let dense = DMatrix::from_row_slice(3, 4, &[
        0, 5, 3, 2,
        2, 0, 0, 0,
        0, 1, 0, 4
    ]);

    assert_eq!(convert_csr_dense(&csr), dense);
    assert_eq!(convert_dense_csr(&dense), csr);
}

#[test]
fn test_convert_csc_dense_bidirectional() {
    let csc = CscMatrix::try_from_csc_data(
        3,
        4,
        vec![0, 1, 3, 4, 6],
        vec![1, 0, 2, 0, 0, 2],
        vec![2, 5, 1, 3, 2, 4],
    )
    .unwrap();

    #[rustfmt::skip]
    let dense = DMatrix::from_row_slice(3, 4, &[
        0, 5, 3, 2,
        2, 0, 0, 0,
        0, 1, 0, 4
    ]);

    assert_eq!(convert_csc_dense(&csc), dense);
    assert_eq!(convert_dense_csc(&dense), csc);
}

fn coo_strategy() -> impl Strategy<Value = CooMatrix<i32>> {
    coo_with_duplicates(-5..=5, 0..=6usize, 0..=6usize, 40, 2)
}

fn coo_no_duplicates_strategy() -> impl Strategy<Value = CooMatrix<i32>> {
    coo_no_duplicates(-5..=5, 0..=6usize, 0..=6usize, 40)
}

fn csr_strategy() -> impl Strategy<Value = CsrMatrix<i32>> {
    csr(-5..=5, 0..=6usize, 0..=6usize, 40)
}

/// Avoid generating explicit zero values so that it is possible to reason about sparsity patterns
fn non_zero_csr_strategy() -> impl Strategy<Value = CsrMatrix<i32>> {
    csr(1..=5, 0..=6usize, 0..=6usize, 40)
}

/// Avoid generating explicit zero values so that it is possible to reason about sparsity patterns
fn non_zero_csc_strategy() -> impl Strategy<Value = CscMatrix<i32>> {
    csc(1..=5, 0..=6usize, 0..=6usize, 40)
}

fn dense_strategy() -> impl Strategy<Value = DMatrix<i32>> {
    matrix(-5..=5, 0..=6, 0..=6)
}

proptest! {

    #[test]
    fn convert_dense_coo_roundtrip(dense in matrix(-5 ..= 5, 0 ..=6, 0..=6)) {
        let coo = convert_dense_coo(&dense);
        let dense2 = convert_coo_dense(&coo);
        prop_assert_eq!(&dense, &dense2);
    }

    #[test]
    fn convert_coo_dense_coo_roundtrip(coo in coo_strategy()) {
        // We cannot compare the result of the roundtrip coo -> dense -> coo directly for
        // two reasons:
        //  1. the COO matrices will generally have different ordering of elements
        //  2. explicitly stored zero entries in the original matrix will be discarded
        //     when converting back to COO
        // Therefore we instead compare the results of converting the COO matrix
        // at the end of the roundtrip with its dense representation
        let dense = convert_coo_dense(&coo);
        let coo2 = convert_dense_coo(&dense);
        let dense2 = convert_coo_dense(&coo2);
        prop_assert_eq!(dense, dense2);
    }

    #[test]
    fn coo_from_dense_roundtrip(dense in dense_strategy()) {
        prop_assert_eq!(&dense, &DMatrix::from(&CooMatrix::from(&dense)));
    }

    #[test]
    fn convert_coo_csr_agrees_with_csr_dense(coo in coo_strategy()) {
        let coo_dense = convert_coo_dense(&coo);
        let csr = convert_coo_csr(&coo);
        let csr_dense = convert_csr_dense(&csr);
        prop_assert_eq!(csr_dense, coo_dense);

        // It might be that COO matrices have a higher nnz due to duplicates,
        // so we can only check that the CSR matrix has no more than the original COO matrix
        prop_assert!(csr.nnz() <= coo.nnz());
    }

    #[test]
    fn convert_coo_csr_nnz(coo in coo_no_duplicates_strategy()) {
        // Check that the NNZ are equal when converting from a CooMatrix without
        // duplicates to a CSR matrix
        let csr = convert_coo_csr(&coo);
        prop_assert_eq!(csr.nnz(), coo.nnz());
    }

    #[test]
    fn convert_csr_coo_roundtrip(csr in csr_strategy()) {
        let coo = convert_csr_coo(&csr);
        let csr2 = convert_coo_csr(&coo);
        prop_assert_eq!(csr2, csr);
    }

    #[test]
    fn coo_from_csr_roundtrip(csr in csr_strategy()) {
        prop_assert_eq!(&csr, &CsrMatrix::from(&CooMatrix::from(&csr)));
    }

    #[test]
    fn csr_from_dense_roundtrip(dense in dense_strategy()) {
        prop_assert_eq!(&dense, &DMatrix::from(&CsrMatrix::from(&dense)));
    }

    #[test]
    fn convert_csr_dense_roundtrip(csr in non_zero_csr_strategy()) {
        // Since we only generate CSR matrices with non-zero values, we know that the
        // number of explicitly stored entries when converting CSR->Dense->CSR should be
        // unchanged, so that we can verify that the result is the same as the input
        let dense = convert_csr_dense(&csr);
        let csr2 = convert_dense_csr(&dense);
        prop_assert_eq!(csr2, csr);
    }

    #[test]
    fn convert_csc_coo_roundtrip(csc in csc_strategy()) {
        let coo = convert_csc_coo(&csc);
        let csc2 = convert_coo_csc(&coo);
        prop_assert_eq!(csc2, csc);
    }

    #[test]
    fn coo_from_csc_roundtrip(csc in csc_strategy()) {
        prop_assert_eq!(&csc, &CscMatrix::from(&CooMatrix::from(&csc)));
    }

    #[test]
    fn convert_csc_dense_roundtrip(csc in non_zero_csc_strategy()) {
        // Since we only generate CSC matrices with non-zero values, we know that the
        // number of explicitly stored entries when converting CSC->Dense->CSC should be
        // unchanged, so that we can verify that the result is the same as the input
        let dense = convert_csc_dense(&csc);
        let csc2 = convert_dense_csc(&dense);
        prop_assert_eq!(csc2, csc);
    }

    #[test]
    fn csc_from_dense_roundtrip(dense in dense_strategy()) {
        prop_assert_eq!(&dense, &DMatrix::from(&CscMatrix::from(&dense)));
    }

    #[test]
    fn convert_coo_csc_agrees_with_csc_dense(coo in coo_strategy()) {
        let coo_dense = convert_coo_dense(&coo);
        let csc = convert_coo_csc(&coo);
        let csc_dense = convert_csc_dense(&csc);
        prop_assert_eq!(csc_dense, coo_dense);

        // It might be that COO matrices have a higher nnz due to duplicates,
        // so we can only check that the CSR matrix has no more than the original COO matrix
        prop_assert!(csc.nnz() <= coo.nnz());
    }

    #[test]
    fn convert_coo_csc_nnz(coo in coo_no_duplicates_strategy()) {
        // Check that the NNZ are equal when converting from a CooMatrix without
        // duplicates to a CSR matrix
        let csc = convert_coo_csc(&coo);
        prop_assert_eq!(csc.nnz(), coo.nnz());
    }

    #[test]
    fn convert_csc_csr_roundtrip(csc in csc_strategy()) {
        let csr = convert_csc_csr(&csc);
        let csc2 = convert_csr_csc(&csr);
        prop_assert_eq!(csc2, csc);
    }

    #[test]
    fn convert_csr_csc_roundtrip(csr in csr_strategy()) {
        let csc = convert_csr_csc(&csr);
        let csr2 = convert_csc_csr(&csc);
        prop_assert_eq!(csr2, csr);
    }

    #[test]
    fn csc_from_csr_roundtrip(csr in csr_strategy()) {
        prop_assert_eq!(&csr, &CsrMatrix::from(&CscMatrix::from(&csr)));
    }

    #[test]
    fn csr_from_csc_roundtrip(csc in csc_strategy()) {
        prop_assert_eq!(&csc, &CscMatrix::from(&CsrMatrix::from(&csc)));
    }
}
