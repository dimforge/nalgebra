#[macro_use]
extern crate approx;
extern crate nalgebra as na;
extern crate num_traits as num;
#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;

mod core;
mod geometry;
mod linalg;
//#[cfg(feature = "sparse")]
//mod sparse;
