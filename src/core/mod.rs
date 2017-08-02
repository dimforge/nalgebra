//! [Reexported at the root of this crate.] Data structures for vector and matrix computations.

pub mod dimension;
pub mod constraint;
pub mod allocator;
pub mod storage;
pub mod coordinates;
mod ops;
mod blas;
pub mod iter;
pub mod default_allocator;

mod scalar;
mod matrix;
mod construction;
mod properties;
mod alias;
mod matrix_alga;
mod conversion;
mod matrix_slice;
mod matrix_array;
mod matrix_vec;
mod cg;
mod unit;
mod componentwise;
mod edition;

#[doc(hidden)]
pub mod helper;

pub use self::scalar::*;
pub use self::matrix::*;
pub use self::unit::*;

pub use self::dimension::*;
pub use self::default_allocator::*;

pub use self::alias::*;
pub use self::matrix_slice::*;
pub use self::matrix_array::*;
pub use self::matrix_vec::*;
