
#![allow(dead_code)]



extern crate num_traits as num;
#[macro_use]
extern crate approx;
extern crate alga;
extern crate nalgebra as na;

pub use aliases::*;
pub use constructors::*;
pub use common::*;
pub use geometric::*;
pub use matrix::*;
pub use traits::*;
pub use trigonometric::*;
pub use vector_relational::*;
pub use exponential::*;

pub use gtx::*;
pub use gtc::*;
pub use ext::*;

mod aliases;
mod constructors;
mod common;
mod matrix;
mod geometric;
mod traits;
mod trigonometric;
mod vector_relational;
mod exponential;
//mod integer;
//mod packing;

pub mod ext;
pub mod gtc;
pub mod gtx;