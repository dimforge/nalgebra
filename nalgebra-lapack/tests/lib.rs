#[macro_use]
extern crate approx;
#[cfg(not(feature = "proptest-support"))]
compile_error!("Tests must be run with `proptest-support`");

extern crate nalgebra as na;
extern crate nalgebra_lapack as nl;

mod linalg;
#[path = "../../tests/proptest/mod.rs"]
mod proptest;
