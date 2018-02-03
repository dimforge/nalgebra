#[cfg(feature = "abomonation-serialize")]
extern crate abomonation;
extern crate alga;
#[macro_use]
extern crate approx;
#[cfg(feature = "mint")]
extern crate mint;
extern crate nalgebra as na;
extern crate num_traits as num;
#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;
extern crate rand;
extern crate serde_json;

mod core;
mod linalg;
mod geometry;
