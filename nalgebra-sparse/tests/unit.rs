//! Unit tests
#[cfg(any(not(feature = "proptest-support"), not(feature = "compare")))]
compile_error!("Tests must be run with features `proptest-support` and `compare`");

mod unit_tests;

#[macro_use]
pub mod common;
