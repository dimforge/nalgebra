//! [Reexported at the root of this crate.] Data structures for vector and matrix computations.

pub mod allocator;
mod blas;
pub mod constraint;
pub mod coordinates;
pub mod default_allocator;
pub mod dimension;
pub mod iter;
mod ops;
pub use ops::{SimpleAdd, SimpleSub};
pub mod storage;

mod alias;
mod alias_slice;
mod array_storage;
mod cg;
mod componentwise;
mod construction;
mod construction_slice;
mod conversion;
mod edition;
pub mod indexing;
mod matrix;
#[cfg(feature = "alga")]
mod matrix_alga;
mod matrix_simba;
mod matrix_slice;
mod norm;
mod properties;
mod scalar;
mod statistics;
mod swizzle;
mod unit;
#[cfg(any(feature = "std", feature = "alloc"))]
mod vec_storage;

#[doc(hidden)]
pub mod helper;

pub use self::matrix::*;
pub use self::norm::*;
pub use self::scalar::*;
pub use self::unit::*;

pub use self::default_allocator::*;
pub use self::dimension::*;

pub use self::alias::*;
pub use self::alias_slice::*;
pub use self::array_storage::*;
pub use self::matrix_slice::*;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::vec_storage::*;
