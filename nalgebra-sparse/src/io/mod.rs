//! Parsers for various matrix formats.
//!
//! Use mm(or MM) to represent matrix market.

pub use self::matrix_market::{
    load_coo_from_mm_file, load_coo_from_mm_str, MMError, MMErrorKind, MMType,
};
mod matrix_market;
