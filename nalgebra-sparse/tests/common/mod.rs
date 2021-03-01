use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::proptest::{csc, csr};
use proptest::strategy::Strategy;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::ops::RangeInclusive;

#[macro_export]
macro_rules! assert_panics {
    ($e:expr) => {{
        use std::panic::catch_unwind;
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
            panic!(
                "assert_panics!({}) failed: the expression did not panic.",
                expr_string
            );
        }
    }};
}

pub const PROPTEST_MATRIX_DIM: RangeInclusive<usize> = 0..=6;
pub const PROPTEST_MAX_NNZ: usize = 40;
pub const PROPTEST_I32_VALUE_STRATEGY: RangeInclusive<i32> = -5..=5;

pub fn value_strategy<T>() -> RangeInclusive<T>
where
    T: TryFrom<i32>,
    T::Error: Debug,
{
    let (start, end) = (
        PROPTEST_I32_VALUE_STRATEGY.start(),
        PROPTEST_I32_VALUE_STRATEGY.end(),
    );
    T::try_from(*start).unwrap()..=T::try_from(*end).unwrap()
}

pub fn non_zero_i32_value_strategy() -> impl Strategy<Value = i32> {
    let (start, end) = (
        PROPTEST_I32_VALUE_STRATEGY.start(),
        PROPTEST_I32_VALUE_STRATEGY.end(),
    );
    assert!(start < &0);
    assert!(end > &0);
    // Note: we don't use RangeInclusive for the second range, because then we'd have different
    // types, which would require boxing
    (*start..0).prop_union(1..*end + 1)
}

pub fn csr_strategy() -> impl Strategy<Value = CsrMatrix<i32>> {
    csr(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

pub fn csc_strategy() -> impl Strategy<Value = CscMatrix<i32>> {
    csc(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}
