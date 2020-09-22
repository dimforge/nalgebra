//! Sparse matrices and algorithms for nalgebra.
//!
//! TODO: Docs
//!
//!
//! ### Planned functionality
//!
//! Below we list desired functionality. This further needs to be refined into what is needed
//! for an initial contribution, and what can be added in future contributions.
//!
//! - Sparsity pattern type. Functionality:
//!     - [x] Access to offsets, indices as slices.
//!     - [x] Return number of nnz
//!     - [x] Access a given lane as a slice of minor indices
//!     - [x] Construct from valid offset + index data
//!     - [ ] Construct from unsorted (but otherwise valid) offset + index data
//!     - [x] Iterate over entries (i, j) in the pattern
//! - CSR matrix type. Functionality:
//!     - [x] Access to CSR data as slices.
//!     - [x] Return number of nnz
//!     - [x] Access a given row, which gives convenient access to the data associated
//!       with a particular row
//!     - [x] Construct from valid CSR data
//!     - [ ] Construct from unsorted CSR data
//!     - [x] Iterate over entries (i, j, v) in the matrix (+mutable).
//!     - [x] Iterate over rows in the matrix (+ mutable).
//!
//! - CSC matrix type. Functionality:
//!     - Same as CSR, but with columns instead of rows.
//! - COO matrix type. Functionality:
//!     - [x] Construct new "empty" COO matrix
//!     - [x] Construct from triplet arrays.
//!     - [x] Push new triplets to the matrix.
//!     - [x] Iterate over triplets.
//!     - [x] "Disassemble" the COO matrix into its underlying triplet arrays.
//! - Format conversion:
//!     - [x] COO -> Dense
//!     - [ ] CSR -> Dense
//!     - [ ] CSC -> Dense
//!     - [ ] COO -> CSR
//!     - [ ] COO -> CSC
//!     - [ ] CSR -> CSC
//!     - [ ] CSC -> CSR
//!     - [ ] CSR -> COO
//!     - [ ] CSC -> COO
//! - Arithmetic. In general arithmetic is only implemented between instances of the same format,
//!   or between dense and instances of a given format. For example, we do not implement
//!   CSR * CSC, only CSR * CSR and CSC * CSC.
//!     - CSR:
//!         - [ ] Dense = CSR * Dense (the other way around is not particularly useful)
//!         - [ ] CSR = CSR * CSR
//!         - [ ] CSR = CSR +- CSR
//!         - [ ] CSR +=/-= CSR
//!     - COO:
//!         - [ ] Dense = COO * Dense (sometimes useful for very sparse matrices)
//!     - CSC:
//!         - Same as CSR
//! - Cholesky factorization (port existing factorization from nalgebra's sparse module)
//!
//!
#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(missing_docs)]

mod coo;
mod csr;
mod pattern;

pub mod ops;

pub use coo::CooMatrix;
pub use csr::{CsrMatrix, CsrRow, CsrRowMut};
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

/// Errors produced by functions that expect well-formed sparse format data.
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
