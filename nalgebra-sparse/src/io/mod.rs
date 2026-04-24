//! Functionality for importing and exporting sparse matrices to and from files.
//!
//! **Available only when the `io` feature is enabled.**
//!
//! The following formats are currently supported:
//!
//! | Format                                          |  Import    |   Export   |
//! | ------------------------------------------------|------------|------------|
//! | [Matrix market](#matrix-market-format)          |  Yes       |    Yes     |
//!
//! [Matrix market]: https://math.nist.gov/MatrixMarket/formats.html
//!
//! ## Matrix Market format
//!
//! The Matrix Market format is a simple ASCII-based file format for sparse matrices, and was initially developed for
//! the [NIST Matrix Market](https://math.nist.gov/MatrixMarket/), a repository of example sparse matrices.
//! In later years it has largely been superseded by the
//! [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) (formerly University of Florida Sparse Matrix Collection),
//! which also uses the Matrix Market file format.
//!
//! We currently offer functionality for importing a Matrix market file to an instance of a
//! [CooMatrix](crate::CooMatrix) through the function [load_coo_from_matrix_market_file],
//! as well as functionality for writing various sparse matrices to the matrix market format
//! through [save_to_matrix_market_file]. It is also possible to load
//! a matrix stored as a string in the matrix market format with the function
//! [load_coo_from_matrix_market_str], or similarly write to a string with
//! [save_to_matrix_market_str].
//!
//! Our implementation is based on the [format description](https://math.nist.gov/MatrixMarket/formats.html)
//! on the Matrix Market website and the
//! [following NIST whitepaper](https://math.nist.gov/MatrixMarket/reports/MMformat.ps):
//!
//! > Boisvert, Ronald F., Roldan Pozo, and Karin A. Remington.<br/>
//! > "*The Matrix Market Exchange Formats: Initial Design.*" (1996).

pub use self::matrix_market::{
    MatrixMarketError, MatrixMarketErrorKind, MatrixMarketExport, MatrixMarketScalar,
    load_coo_from_matrix_market_file, load_coo_from_matrix_market_str, save_to_matrix_market,
    save_to_matrix_market_file, save_to_matrix_market_str,
};
mod matrix_market;
