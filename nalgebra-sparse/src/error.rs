//! Error types used across the library

use thiserror::Error;

/// Errors produced by constructors or functions that expect to check sparse format data
/// correctness.
#[derive(Debug, Error)]
#[error("{error}")]
pub struct SparseFormatError {
    kind: SparseFormatErrorKind,
    // Currently we only use an underlying error for generating the `Display` impl
    error: Box<dyn std::error::Error>,
}

impl SparseFormatError {
    /// The type of error.
    #[must_use]
    pub fn kind(&self) -> &SparseFormatErrorKind {
        &self.kind
    }

    pub(crate) fn from_kind_and_error(
        kind: SparseFormatErrorKind,
        error: Box<dyn std::error::Error>,
    ) -> Self {
        Self { kind, error }
    }

    /// Helper functionality for more conveniently creating errors.
    pub(crate) fn from_kind_and_msg(kind: SparseFormatErrorKind, msg: &'static str) -> Self {
        Self::from_kind_and_error(kind, Box::<dyn std::error::Error>::from(msg))
    }
}

/// The type of format error described by a [SparseFormatError](struct.SparseFormatError.html).
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

/// Error type for `SparsityPattern` format errors.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum SparsityPatternFormatError {
    /// Indicates an invalid number of offsets.
    ///
    /// The number of offsets must be equal to `major_dim`.
    #[error("Length of offset array is not equal to `major_dim`.")]
    InvalidOffsetArrayLength,

    /// The first entry of the offsets was not equal to zero.
    #[error("The first entry of the offsets must be exactly zero.")]
    InvalidFirstOffset,

    /// Indicates that the major offsets are not monotonically increasing.
    #[error("The offsets do not monotonically increase.")]
    NonmonotonicOffsets,

    /// One or more minor indices are out of bounds.
    #[error("A minor index is out of bounds.")]
    MinorIndexOutOfBounds,

    /// Data and indices do not share the same size.
    #[error("Data and indices do not share the same size (number of non-zeros not equal).")]
    DataAndIndicesSizeMismatch,

    /// One or more duplicate entries were detected.
    ///
    /// Two entries are considered duplicates if they are part of the same major lane and have
    /// the same minor index.
    #[error("Input data contains duplicate entries.")]
    DuplicateEntry,

    /// Indicates that minor indices are not monotonically increasing within each lane.
    #[error("Minor axis indices do not monotonically increase across their respective lanes.")]
    NonmonotonicMinorIndices,
}

impl From<SparsityPatternFormatError> for SparseFormatError {
    fn from(err: SparsityPatternFormatError) -> Self {
        use SparsityPatternFormatError::*;

        match err {
            InvalidOffsetArrayLength
            | InvalidFirstOffset
            | NonmonotonicOffsets
            | NonmonotonicMinorIndices
            | DataAndIndicesSizeMismatch => SparseFormatError::from_kind_and_error(
                SparseFormatErrorKind::InvalidStructure,
                Box::from(err),
            ),
            MinorIndexOutOfBounds => SparseFormatError::from_kind_and_error(
                SparseFormatErrorKind::IndexOutOfBounds,
                Box::from(err),
            ),
            DuplicateEntry => SparseFormatError::from_kind_and_error(
                #[allow(unused_qualifications)]
                SparseFormatErrorKind::DuplicateEntry,
                Box::from(err),
            ),
        }
    }
}
