//! TODO

#[macro_use]
macro_rules! assert_compatible_spmm_dims {
    ($c:expr, $a:expr, $b:expr, $trans_a:expr, $trans_b:expr) => {
        use crate::ops::Transposition::{Transpose, NoTranspose};
        match ($trans_a, $trans_b) {
            (NoTranspose, NoTranspose) => {
                assert_eq!($c.nrows(), $a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), $b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!($a.ncols(), $b.nrows(), "A.ncols() != B.nrows()");
            },
            (Transpose, NoTranspose) => {
                assert_eq!($c.nrows(), $a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), $b.ncols(), "C.ncols() != B.ncols()");
                assert_eq!($a.nrows(), $b.nrows(), "A.nrows() != B.nrows()");
            },
            (NoTranspose, Transpose) => {
                assert_eq!($c.nrows(), $a.nrows(), "C.nrows() != A.nrows()");
                assert_eq!($c.ncols(), $b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!($a.ncols(), $b.ncols(), "A.ncols() != B.ncols()");
            },
            (Transpose, Transpose) => {
                assert_eq!($c.nrows(), $a.ncols(), "C.nrows() != A.ncols()");
                assert_eq!($c.ncols(), $b.nrows(), "C.ncols() != B.nrows()");
                assert_eq!($a.nrows(), $b.ncols(), "A.nrows() != B.ncols()");
            }
        }

    }
}

mod coo;
mod csr;

pub use coo::*;
pub use csr::*;