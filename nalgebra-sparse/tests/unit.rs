//! Unit tests
#[cfg(not(all(feature = "proptest-support", feature = "compare", feature = "io",)))]
compile_error!(
    "Please enable the `proptest-support`, `compare` and `io` features in order to compile and run the tests.
     Example: `cargo test -p nalgebra-sparse --features proptest-support,compare,io`"
);

mod unit_tests;

#[macro_use]
pub mod common;
