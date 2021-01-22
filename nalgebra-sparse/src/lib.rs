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
//!     - [x] "Disassemble" the sparsity pattern into the raw index data arrays.
//! - CSR matrix type. Functionality:
//!     - [x] Access to CSR data as slices.
//!     - [x] Return number of nnz
//!     - [x] Access a given row, which gives convenient access to the data associated
//!       with a particular row
//!     - [x] Construct from valid CSR data
//!     - [ ] Construct from unsorted CSR data
//!     - [x] Iterate over entries (i, j, v) in the matrix (+mutable).
//!     - [x] Iterate over rows in the matrix (+ mutable).
//!     - [x] "Disassemble" the CSR matrix into the raw CSR data arrays.
//!
//! - CSC matrix type. Functionality:
//!     - [x] Access to CSC data as slices.
//!     - [x] Return number of nnz
//!     - [x] Access a given column, which gives convenient access to the data associated
//!       with a particular column
//!     - [x] Construct from valid CSC data
//!     - [ ] Construct from unsorted CSC data
//!     - [x] Iterate over entries (i, j, v) in the matrix (+mutable).
//!     - [x] Iterate over rows in the matrix (+ mutable).
//!     - [x] "Disassemble" the CSC matrix into the raw CSC data arrays.
//! - COO matrix type. Functionality:
//!     - [x] Construct new "empty" COO matrix
//!     - [x] Construct from triplet arrays.
//!     - [x] Push new triplets to the matrix.
//!     - [x] Iterate over triplets.
//!     - [x] "Disassemble" the COO matrix into its underlying triplet arrays.
//! - Format conversion:
//!     - [x] COO -> Dense
//!     - [x] CSR -> Dense
//!     - [x] CSC -> Dense
//!     - [x] COO -> CSR
//!     - [x] COO -> CSC
//!     - [x] CSR -> CSC
//!     - [x] CSC -> CSR
//!     - [x] CSR -> COO
//!     - [x] CSC -> COO
//!     - [x] Dense -> COO
//!     - [x] Dense -> CSR
//!     - [x] Dense -> CSC
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
//! TODO: Write docs on the following:
//!
//! - Overall design ("easy API" vs. "expert" API etc.)
//! - Conversions (From, explicit "expert" API etc.)
//! - Matrix ops design
//! - Proptest and matrixcompare integrations
#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(missing_docs)]

pub mod coo;
pub mod csc;
pub mod csr;
pub mod pattern;
pub mod ops;
pub mod convert;
pub mod factorization;

pub(crate) mod cs;

#[cfg(feature = "proptest-support")]
pub mod proptest;

#[cfg(feature = "compare")]
mod matrixcompare;

use std::error::Error;
use std::fmt;
use num_traits::Zero;

/// Errors produced by functions that expect well-formed sparse format data.
#[derive(Debug)]
pub struct SparseFormatError {
    kind: SparseFormatErrorKind,
    // Currently we only use an underlying error for generating the `Display` impl
    error: Box<dyn Error>
}

impl SparseFormatError {
    /// The type of error.
    pub fn kind(&self) -> &SparseFormatErrorKind {
        &self.kind
    }

    pub(crate) fn from_kind_and_error(kind: SparseFormatErrorKind, error: Box<dyn Error>) -> Self {
        Self {
            kind,
            error
        }
    }

    /// Helper functionality for more conveniently creating errors.
    pub(crate) fn from_kind_and_msg(kind: SparseFormatErrorKind, msg: &'static str) -> Self {
        Self::from_kind_and_error(kind, Box::<dyn Error>::from(msg))
    }
}

/// The type of format error described by a [SparseFormatError](struct.SparseFormatError.html).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseFormatErrorKind {
    /// Indicates that the index data associated with the format contains at least one index
    /// out of bounds.
    IndexOutOfBounds,

    /// Indicates that the provided data contains at least one duplicate entry, and the
    /// current format does not support duplicate entries.
    DuplicateEntry,

    /// Indicates that the provided data for the format does not conform to the high-level
    /// structure of the format.
    ///
    /// For example, the arrays defining the format data might have incompatible sizes.
    InvalidStructure,
}

impl fmt::Display for SparseFormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl Error for SparseFormatError {}

/// An entry in a sparse matrix.
///
/// Sparse matrices do not store all their entries explicitly. Therefore, entry (i, j) in the matrix
/// can either be a reference to an explicitly stored element, or it is implicitly zero.
#[derive(Debug, PartialEq, Eq)]
pub enum SparseEntry<'a, T> {
    /// The entry is a reference to an explicitly stored element.
    ///
    /// Note that the naming here is a misnomer: The element can still be zero, even though it
    /// is explicitly stored (a so-called "explicit zero").
    NonZero(&'a T),
    /// The entry is implicitly zero, i.e. it is not explicitly stored.
    Zero
}

impl<'a, T: Clone + Zero> SparseEntry<'a, T> {
    /// Returns the value represented by this entry.
    ///
    /// Either clones the underlying reference or returns zero if the entry is not explicitly
    /// stored.
    pub fn to_value(self) -> T {
        match self {
            SparseEntry::NonZero(value) => value.clone(),
            SparseEntry::Zero => T::zero()
        }
    }
}

/// A mutable entry in a sparse matrix.
///
/// See also `SparseEntry`.
#[derive(Debug, PartialEq, Eq)]
pub enum SparseEntryMut<'a, T> {
    /// The entry is a mutable reference to an explicitly stored element.
    ///
    /// Note that the naming here is a misnomer: The element can still be zero, even though it
    /// is explicitly stored (a so-called "explicit zero").
    NonZero(&'a mut T),
    /// The entry is implicitly zero i.e. it is not explicitly stored.
    Zero
}

impl<'a, T: Clone + Zero> SparseEntryMut<'a, T> {
    /// Returns the value represented by this entry.
    ///
    /// Either clones the underlying reference or returns zero if the entry is not explicitly
    /// stored.
    pub fn to_value(self) -> T {
        match self {
            SparseEntryMut::NonZero(value) => value.clone(),
            SparseEntryMut::Zero => T::zero()
        }
    }
}