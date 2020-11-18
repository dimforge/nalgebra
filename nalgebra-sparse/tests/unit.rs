//! Unit tests
#[cfg(not(feature = "proptest-support"))]
compile_error!("Tests must be run with feature proptest-support");

mod unit_tests;

#[macro_use]
pub mod common;