#[cfg(any(
    not(feature = "debug"),
    not(feature = "compare"),
    not(feature = "rand")
))]
compile_error!(
    "Please enable the `debug`, `compare`, and `rand` features in order to compile and run the tests.
     Example: `cargo test --features debug,compare,rand`"
);

#[cfg(feature = "abomonation-serialize")]
extern crate abomonation;
#[macro_use]
extern crate approx;
extern crate nalgebra as na;
extern crate num_traits as num;
extern crate rand_package as rand;

mod core;
mod geometry;
mod linalg;

#[cfg(feature = "proptest-support")]
mod proptest;

//#[cfg(feature = "sparse")]
//mod sparse;
