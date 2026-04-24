//! Serial sparse matrix arithmetic routines.
//!
//! All routines are single-threaded.
//!
//! Some operations have the `prealloc` suffix. This means that they expect that the sparsity
//! pattern of the output matrix has already been pre-allocated: that is, the pattern of the result
//! of the operation fits entirely in the output pattern. In the future, there will also be
//! some operations which will be able to dynamically adapt the output pattern to fit the
//! result, but these have yet to be implemented.

macro_rules! assert_compatible_spmm_dims {
    ($c:expr, $a:expr, $b:expr) => {{
        use crate::ops::Op::{NoOp, Transpose};
        match (&$a, &$b) {
            (&NoOp(ref a), &NoOp(ref b)) => {
                assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!(a.ncols(), b.nrows(), "A.ncols() != B.nrows()");
            }
            (&Transpose(ref a), &NoOp(ref b)) => {
                assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!(a.nrows(), b.nrows(), "A.nrows() != B.nrows()");
            }
            (&NoOp(ref a), &Transpose(ref b)) => {
                assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!(a.ncols(), b.ncols(), "A.ncols() != B.ncols()");
            }
            (&Transpose(ref a), &Transpose(ref b)) => {
                assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!(a.nrows(), b.ncols(), "A.nrows() != B.ncols()");
            }
        }
    }};
}

macro_rules! assert_compatible_spadd_dims {
    ($c:expr, $a:expr) => {
        use crate::ops::Op;
        match $a {
            Op::NoOp(a) => {
                assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), a.ncols(), "C.ncols() != A.ncols()");
            }
            Op::Transpose(a) => {
                assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), a.nrows(), "C.ncols() != A.nrows()");
            }
        }
    };
}

mod cs;
mod csc;
mod csr;
mod pattern;

pub use csc::*;
pub use csr::*;
pub use pattern::*;
use std::fmt;
use std::fmt::Formatter;

/// A description of the error that occurred during an arithmetic operation.
#[derive(Clone, Debug)]
pub struct OperationError {
    error_kind: OperationErrorKind,
    message: String,
}

/// The different kinds of operation errors that may occur.
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
pub enum OperationErrorKind {
    /// Indicates that one or more sparsity patterns involved in the operation violate the
    /// expectations of the routine.
    ///
    /// For example, this could indicate that the sparsity pattern of the output is not able to
    /// contain the result of the operation.
    InvalidPattern,

    /// Indicates that a matrix is singular when it is expected to be invertible.
    Singular,
}

impl OperationError {
    fn from_kind_and_message(error_type: OperationErrorKind, message: String) -> Self {
        Self {
            error_kind: error_type,
            message,
        }
    }

    /// The operation error kind.
    #[must_use]
    pub fn kind(&self) -> &OperationErrorKind {
        &self.error_kind
    }

    /// The underlying error message.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse matrix operation error: ")?;
        match self.kind() {
            OperationErrorKind::InvalidPattern => {
                write!(f, "InvalidPattern")?;
            }
            OperationErrorKind::Singular => {
                write!(f, "Singular")?;
            }
        }
        write!(f, " Message: {}", self.message)
    }
}

impl std::error::Error for OperationError {}
