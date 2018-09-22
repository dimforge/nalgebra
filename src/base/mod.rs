//! [Reexported at the root of this crate.] Data structures for vector and matrix computations.

pub mod allocator;
mod blas;
pub mod constraint;
pub mod coordinates;
pub mod default_allocator;
pub mod dimension;
pub mod iter;
mod ops;
pub mod storage;

mod alias;
mod alias_slice;
mod swizzle;
mod cg;
mod componentwise;
mod construction;
mod construction_slice;
mod conversion;
mod edition;
mod matrix;
mod matrix_alga;
mod matrix_array;
mod matrix_slice;
#[cfg(any(feature = "std", feature = "alloc"))]
mod matrix_vec;
mod properties;
mod scalar;
mod unit;

#[doc(hidden)]
pub mod helper;

pub use self::matrix::*;
pub use self::scalar::*;
pub use self::unit::*;

pub use self::default_allocator::*;
pub use self::dimension::*;

pub use self::alias::*;
pub use self::alias_slice::*;
pub use self::matrix_array::*;
pub use self::matrix_slice::*;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::matrix_vec::*;
