#![cfg(feature = "serde-serialize")]
//! Serialization tests
#[cfg(any(not(feature = "proptest-support"), not(feature = "compare")))]
compile_error!("Tests must be run with features `proptest-support` and `compare`");

#[macro_use]
pub mod common;

use nalgebra_sparse::csr::CsrMatrix;

use proptest::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::csr_strategy;

fn json_roundtrip<T: Serialize + for<'a> Deserialize<'a>>(csr: &CsrMatrix<T>) -> CsrMatrix<T> {
    let serialized = serde_json::to_string(csr).unwrap();
    let deserialized: CsrMatrix<T> = serde_json::from_str(&serialized).unwrap();
    deserialized
}

#[test]
fn csr_roundtrip() {
    {
        // A CSR matrix with zero explicitly stored entries
        let offsets = vec![0, 0, 0, 0];
        let indices = vec![];
        let values = Vec::<i32>::new();
        let matrix = CsrMatrix::try_from_csr_data(3, 2, offsets, indices, values).unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }

    {
        // An arbitrary CSR matrix
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix =
            CsrMatrix::try_from_csr_data(3, 6, offsets.clone(), indices.clone(), values.clone())
                .unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }
}

#[test]
fn invalid_csr_deserialize() {
    // Valid matrix: {"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,8,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3,1,1],"values":[0,1,2,3,4]}"#).is_err());
    // The following actually panics ('range end index 10 out of range for slice of length 5', nalgebra-sparse\src\pattern.rs:156:38)
    //assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,10,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
}

proptest! {
    #[test]
    fn csr_roundtrip_proptest(csr in csr_strategy()) {
        prop_assert_eq!(json_roundtrip(&csr), csr);
    }
}
