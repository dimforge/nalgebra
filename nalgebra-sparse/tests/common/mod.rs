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