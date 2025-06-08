//! Sparse matrices and algorithms for [nalgebra](https://www.nalgebra.rs).
//!
//! This crate extends `nalgebra` with sparse matrix formats and operations on sparse matrices.
//!
//! ## Goals
//! The long-term goals for this crate are listed below.
//!
//! - Provide proven sparse matrix formats in an easy-to-use and idiomatic Rust API that
//!   naturally integrates with `nalgebra`.
//! - Provide additional expert-level APIs for fine-grained control over operations.
//! - Integrate well with external sparse matrix libraries.
//! - Provide native Rust high-performance routines, including parallel matrix operations.
//!
//! ## Highlighted current features
//!
//! - [CSR](csr::CsrMatrix), [CSC](csc::CscMatrix) and [COO](coo::CooMatrix) formats, and
//!   [conversions](`convert`) between them.
//! - Common arithmetic operations are implemented. See the [`ops`] module.
//! - Sparsity patterns in CSR and CSC matrices are explicitly represented by the
//!   [SparsityPattern](pattern::SparsityPattern) type, which encodes the invariants of the
//!   associated index data structures.
//! - [Matrix market format support](`io`) when the `io` feature is enabled.
//! - [proptest strategies](`proptest`) for sparse matrices when the feature
//!   `proptest-support` is enabled.
//! - [matrixcompare support](https://crates.io/crates/matrixcompare) for effortless
//!   (approximate) comparison of matrices in test code (requires the `compare` feature).
//!
//! ## Current state
//!
//! The library is in an early, but usable state. The API has been designed to be extensible,
//! but breaking changes will be necessary to implement several planned features. While it is
//! backed by an extensive test suite, it has yet to be thoroughly battle-tested in real
//! applications. Moreover, the focus so far has been on correctness and API design, with little
//! focus on performance. Future improvements will include incremental performance enhancements.
//!
//! Current limitations:
//!
//! - Limited or no availability of sparse system solvers.
//! - Limited support for complex numbers. Currently only arithmetic operations that do not
//!   rely on particular properties of complex numbers, such as e.g. conjugation, are
//!   supported.
//! - No integration with external libraries.
//!
//! # Usage
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! nalgebra_sparse = "0.1"
//! ```
//!
//! # Supported matrix formats
//!
//! | Format                  | Notes                                        |
//! | ------------------------|--------------------------------------------- |
//! | [COO](`coo::CooMatrix`) | Well-suited for matrix construction. <br /> Ill-suited for algebraic operations. |
//! | [CSR](`csr::CsrMatrix`) | Immutable sparsity pattern, suitable for algebraic operations. <br /> Fast row access. |
//! | [CSC](`csc::CscMatrix`) | Immutable sparsity pattern, suitable for algebraic operations. <br /> Fast column access. |
//!
//! What format is best to use depends on the application. The most common use case for sparse
//! matrices in science is the solution of sparse linear systems. Here we can differentiate between
//! two common cases:
//!
//! - Direct solvers. Typically, direct solvers take their input in CSR or CSC format.
//! - Iterative solvers. Many iterative solvers require only matrix-vector products,
//!   for which the CSR or CSC formats are suitable.
//!
//! The [COO](coo::CooMatrix) format is primarily intended for matrix construction.
//! A common pattern is to use COO for construction, before converting to CSR or CSC for use
//! in a direct solver or for computing matrix-vector products in an iterative solver.
//! Some high-performance applications might also directly manipulate the CSR and/or CSC
//! formats.
//!
//! # Example: COO -> CSR -> matrix-vector product
//!
//! ```
//! use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
//! use nalgebra::{DMatrix, DVector};
//! use matrixcompare::assert_matrix_eq;
//!
//! // The dense representation of the matrix
//! let dense = DMatrix::from_row_slice(3, 3,
//!     &[1.0, 0.0, 3.0,
//!       2.0, 0.0, 1.3,
//!       0.0, 0.0, 4.1]);
//!
//! // Build the equivalent COO representation. We only add the non-zero values
//! let mut coo = CooMatrix::new(3, 3);
//! // We can add elements in any order. For clarity, we do so in row-major order here.
//! coo.push(0, 0, 1.0);
//! coo.push(0, 2, 3.0);
//! coo.push(1, 0, 2.0);
//! coo.push(1, 2, 1.3);
//! coo.push(2, 2, 4.1);
//!
//! // ... or add entire dense matrices like so:
//! // coo.push_matrix(0, 0, &dense);
//!
//! // The simplest way to construct a CSR matrix is to first construct a COO matrix, and
//! // then convert it to CSR. The `From` trait is implemented for conversions between different
//! // sparse matrix types.
//! // Alternatively, we can construct a matrix directly from the CSR data.
//! // See the docs for CsrMatrix for how to do that.
//! let csr = CsrMatrix::from(&coo);
//!
//! // Let's check that the CSR matrix and the dense matrix represent the same matrix.
//! // We can use macros from the `matrixcompare` crate to easily do this, despite the fact that
//! // we're comparing across two different matrix formats. Note that these macros are only really
//! // appropriate for writing tests, however.
//! assert_matrix_eq!(csr, dense);
//!
//! let x = DVector::from_column_slice(&[1.3, -4.0, 3.5]);
//!
//! // Compute the matrix-vector product y = A * x. We don't need to specify the type here,
//! // but let's just do it to make sure we get what we expect
//! let y: DVector<_> = &csr * &x;
//!
//! // Verify the result with a small element-wise absolute tolerance
//! let y_expected = DVector::from_column_slice(&[11.8, 7.15, 14.35]);
//! assert_matrix_eq!(y, y_expected, comp = abs, tol = 1e-9);
//!
//! // The above expression is simple, and gives easy to read code, but if we're doing this in a
//! // loop, we'll have to keep allocating new vectors. If we determine that this is a bottleneck,
//! // then we can resort to the lower level APIs for more control over the operations
//! {
//!     use nalgebra_sparse::ops::{Op, serial::spmm_csr_dense};
//!     let mut y = y;
//!     // Compute y <- 0.0 * y + 1.0 * csr * dense. We store the result directly in `y`, without
//!     // any intermediate allocations
//!     spmm_csr_dense(0.0, &mut y, 1.0, Op::NoOp(&csr), Op::NoOp(&x));
//!     assert_matrix_eq!(y, y_expected, comp = abs, tol = 1e-9);
//! }
//! ```
#![deny(
    nonstandard_style,
    unused,
    missing_docs,
    rust_2018_idioms,
    rust_2024_compatibility,
    future_incompatible,
    missing_copy_implementations
)]

pub extern crate nalgebra as na;
#[macro_use]
#[cfg(feature = "io")]
extern crate pest_derive;

pub mod convert;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod factorization;
#[cfg(feature = "io")]
pub mod io;
pub mod ops;
pub mod pattern;

pub(crate) mod cs;
pub(crate) mod utils;

#[cfg(feature = "proptest-support")]
pub mod proptest;

#[cfg(feature = "compare")]
mod matrixcompare;

use num_traits::Zero;
use std::error::Error;
use std::fmt;

pub use self::coo::CooMatrix;
pub use self::csc::CscMatrix;
pub use self::csr::CsrMatrix;

/// Errors produced by functions that expect well-formed sparse format data.
#[derive(Debug)]
pub struct SparseFormatError {
    kind: SparseFormatErrorKind,
    // Currently we only use an underlying error for generating the `Display` impl
    error: Box<dyn Error>,
}

impl SparseFormatError {
    /// The type of error.
    #[must_use]
    pub fn kind(&self) -> &SparseFormatErrorKind {
        &self.kind
    }

    pub(crate) fn from_kind_and_error(kind: SparseFormatErrorKind, error: Box<dyn Error>) -> Self {
        Self { kind, error }
    }

    /// Helper functionality for more conveniently creating errors.
    pub(crate) fn from_kind_and_msg(kind: SparseFormatErrorKind, msg: &'static str) -> Self {
        Self::from_kind_and_error(kind, Box::<dyn Error>::from(msg))
    }
}

/// The type of format error described by a [`SparseFormatError`].
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    Zero,
}

impl<'a, T: Clone + Zero> SparseEntry<'a, T> {
    /// Returns the value represented by this entry.
    ///
    /// Either clones the underlying reference or returns zero if the entry is not explicitly
    /// stored.
    pub fn into_value(self) -> T {
        match self {
            SparseEntry::NonZero(value) => value.clone(),
            SparseEntry::Zero => T::zero(),
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
    Zero,
}

impl<'a, T: Clone + Zero> SparseEntryMut<'a, T> {
    /// Returns the value represented by this entry.
    ///
    /// Either clones the underlying reference or returns zero if the entry is not explicitly
    /// stored.
    pub fn into_value(self) -> T {
        match self {
            SparseEntryMut::NonZero(value) => value.clone(),
            SparseEntryMut::Zero => T::zero(),
        }
    }
}
