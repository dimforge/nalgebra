use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::convert::serial::{convert_coo_dense, convert_dense_coo};
use nalgebra_sparse::proptest::coo_with_duplicates;
use nalgebra::proptest::matrix;
use proptest::prelude::*;
use nalgebra::DMatrix;

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
        let coo_no_dup = CooMatrix::try_from_triplets(2, 3,
                                                      vec![0, 1, 0],
                                                      vec![0, 1, 2],
                                                      vec![1, 5, 3])
            .unwrap();
        let coo_dup = CooMatrix::try_from_triplets(2, 3,
                                                   vec![0, 1, 0, 1],
                                                   vec![0, 1, 2, 1],
                                                   vec![1, -2, 3, 7])
            .unwrap();

        assert_eq!(CooMatrix::from(&dense), coo_no_dup);
        assert_eq!(DMatrix::from(&coo_dup), dense);
    }
}

fn coo_strategy() -> impl Strategy<Value=CooMatrix<i32>> {
    coo_with_duplicates(-5 ..= 5, 0..=6usize, 0..=6usize, 40, 2)
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
    fn from_dense_coo_roundtrip(dense in matrix(-5..=5, 0..=6, 0..=6)) {
        prop_assert_eq!(&dense, &DMatrix::from(&CooMatrix::from(&dense)));
    }
}