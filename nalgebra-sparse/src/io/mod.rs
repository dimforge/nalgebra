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
//! [CooMatrix](crate::CooMatrix) through the function [load_coo_from_matrix_market_file]. It is also possible to load
//! a matrix stored in the matrix market format with the function [load_coo_from_matrix_market_str].
//!
//! Export is currently not implemented, but [planned](https://github.com/dimforge/nalgebra/issues/1037).
//!
//! Our implementation is based on the [format description](https://math.nist.gov/MatrixMarket/formats.html)
//! on the Matrix Market website and the
//! [following NIST whitepaper](https://math.nist.gov/MatrixMarket/reports/MMformat.ps):
//!
//! > Boisvert, Ronald F., Roldan Pozo, and Karin A. Remington.<br/>
//! > "*The Matrix Market Exchange Formats: Initial Design.*" (1996).

pub use self::matrix_market::{
    load_coo_from_matrix_market_file, load_coo_from_matrix_market_str, write_to_matrix_market_file,
    write_to_matrix_market_str, MatrixMarketError, MatrixMarketErrorKind, MatrixMarketExport,
    MatrixMarketScalar,
};
mod matrix_market;
