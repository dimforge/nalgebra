//! TODO

#[macro_use]
macro_rules! assert_compatible_spmm_dims {
    ($c:expr, $a:expr, $b:expr, $trans_a:expr, $trans_b:expr) => {
        use crate::ops::Transpose;
        match ($trans_a, $trans_b) {
            (Transpose(false), Transpose(false)) => {
                assert_eq!($c.nrows(), $a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), $b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!($a.ncols(), $b.nrows(), "A.ncols() != B.nrows()");
            },
            (Transpose(true), Transpose(false)) => {
                assert_eq!($c.nrows(), $a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), $b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!($a.nrows(), $b.nrows(), "A.nrows() != B.nrows()");
            },
            (Transpose(false), Transpose(true)) => {
                assert_eq!($c.nrows(), $a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), $b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!($a.ncols(), $b.ncols(), "A.ncols() != B.ncols()");
            },
            (Transpose(true), Transpose(true)) => {
                assert_eq!($c.nrows(), $a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), $b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!($a.nrows(), $b.ncols(), "A.nrows() != B.ncols()");
            }
        }

    }
}

mod coo;
mod csr;
mod pattern;

pub use coo::*;
pub use csr::*;
pub use pattern::*;