mod coo;
mod csr;
mod pattern;

pub mod ops;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use pattern::{SparsityPattern};

/// Iterator types for matrices.
///
/// Most users will not need to interface with these types directly. Instead, refer to the
/// iterator methods for the respective matrix formats.
pub mod iter {
    // Iterators are best implemented in the same modules as the matrices they iterate over,
    // since they are so closely tied to their respective implementations. However,
    // in the crate's public API we move them into a separate `iter` module in order to avoid
    // cluttering the docs with iterator types that most users will never need to explicitly
    // know about.
    pub use crate::pattern::SparsityPatternIter;
    pub use crate::csr::{CsrTripletIter, CsrTripletIterMut};
}

use std::error::Error;
use std::fmt;

#[derive(Debug)]
#[non_exhaustive]
pub enum SparseFormatError {
    /// Indicates that the index data associated with the format contains at least one index
    /// out of bounds.
    IndexOutOfBounds(Box<dyn Error>),

    /// Indicates that the provided data contains at least one duplicate entry, and the
    /// current format does not support duplicate entries.
    DuplicateEntry(Box<dyn Error>),

    /// Indicates that the provided data for the format does not conform to the high-level
    /// structure of the format.
    ///
    /// For example, the arrays defining the format data might have incompatible sizes.
    InvalidStructure(Box<dyn Error>),
}

impl fmt::Display for SparseFormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds(err) => err.fmt(f),
            Self::DuplicateEntry(err) => err.fmt(f),
            Self::InvalidStructure(err) => err.fmt(f)
        }
    }
}

impl Error for SparseFormatError {}
