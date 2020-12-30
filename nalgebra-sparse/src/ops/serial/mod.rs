//! TODO

#[macro_use]
macro_rules! assert_compatible_spmm_dims {
    ($c:expr, $a:expr, $b:expr) => {
        {
            use crate::ops::Op::{NoOp, Transpose};
            match (&$a, &$b) {
                (NoOp(ref a), NoOp(ref b)) => {
                    assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                    assert_eq!($c.ncols(), b.ncols(), "C.ncols() != B.ncols()");
                    assert_eq!(a.ncols(), b.nrows(), "A.ncols() != B.nrows()");
                },
                (Transpose(ref a), NoOp(ref b)) => {
                    assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                    assert_eq!($c.ncols(), b.ncols(), "C.ncols() != B.ncols()");
                    assert_eq!(a.nrows(), b.nrows(), "A.nrows() != B.nrows()");
                },
                (NoOp(ref a), Transpose(ref b)) => {
                    assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                    assert_eq!($c.ncols(), b.nrows(), "C.ncols() != B.nrows()");
                    assert_eq!(a.ncols(), b.ncols(), "A.ncols() != B.ncols()");
                },
                (Transpose(ref a), Transpose(ref b)) => {
                    assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                    assert_eq!($c.ncols(), b.nrows(), "C.ncols() != B.nrows()");
                    assert_eq!(a.nrows(), b.ncols(), "A.nrows() != B.ncols()");
                }
            }
        }
    }
}

#[macro_use]
macro_rules! assert_compatible_spadd_dims {
    ($c:expr, $a:expr) => {
        use crate::ops::Op;
        match $a {
            Op::NoOp(a) => {
                assert_eq!($c.nrows(), a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), a.ncols(), "C.ncols() != A.ncols()");
            },
            Op::Transpose(a) => {
                assert_eq!($c.nrows(), a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), a.nrows(), "C.ncols() != A.nrows()");
            }
        }

    }
}

mod csc;
mod csr;
mod pattern;
mod cs;

pub use csc::*;
pub use csr::*;
pub use pattern::*;

/// TODO
#[derive(Clone, Debug)]
pub struct OperationError {
    error_type: OperationErrorType,
    message: String
}

/// TODO
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum OperationErrorType {
    /// TODO
    InvalidPattern,
}

impl OperationError {
    /// TODO
    pub fn from_type_and_message(error_type: OperationErrorType, message: String) -> Self {
        Self { error_type, message }
    }
}