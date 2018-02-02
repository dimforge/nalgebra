#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;
#[macro_use]
extern crate approx;
extern crate num_traits as num;
extern crate serde_json;
#[cfg(feature = "abomonation-serialize")]
extern crate abomonation;
#[cfg(feature = "mint")]
extern crate mint;
extern crate rand;
extern crate alga;
extern crate nalgebra as na;


mod core;
// mod linalg;
// mod geometry;
