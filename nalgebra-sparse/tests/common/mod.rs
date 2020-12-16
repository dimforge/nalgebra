use proptest::strategy::Strategy;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::proptest::{csr, csc};
use nalgebra_sparse::csc::CscMatrix;
use std::ops::RangeInclusive;

#[macro_export]
macro_rules! assert_panics {
    ($e:expr) => {{
        use std::panic::{catch_unwind};
        use std::stringify;
        let expr_string = stringify!($e);

        // Note: We cannot manipulate the panic hook here, because it is global and the test
        // suite is run in parallel, which leads to race conditions in the sense
        // that some regular tests that panic might not output anything anymore.
        // Unfortunately this means that output is still printed to stdout if
        // we run cargo test -- --nocapture. But Cargo does not forward this if the test
        // binary is not run with nocapture, so it is somewhat acceptable nonetheless.

        let result = catch_unwind(|| $e);
        if result.is_ok() {
            panic!("assert_panics!({}) failed: the expression did not panic.", expr_string);
        }
    }};
}

pub const PROPTEST_MATRIX_DIM: RangeInclusive<usize> = 0..=6;
pub const PROPTEST_MAX_NNZ: usize = 40;
pub const PROPTEST_I32_VALUE_STRATEGY: RangeInclusive<i32> = -5 ..= 5;

pub fn csr_strategy() -> impl Strategy<Value=CsrMatrix<i32>> {
    csr(-5 ..= 5, PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ)
}

pub fn csc_strategy() -> impl Strategy<Value=CscMatrix<i32>> {
    csc(-5 ..= 5, PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ)
}
