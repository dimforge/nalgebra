//! Parsers for various matrix formats.
//!
//! ## Matrix Market
//! See the [website](https://math.nist.gov/MatrixMarket/formats.html) or the [paper](https://www.researchgate.net/publication/2630533_The_Matrix_Market_Exchange_Formats_Initial_Design) for more details about matrix market.

pub use self::matrix_market::{
    load_coo_from_matrix_market_file, load_coo_from_matrix_market_str, MatrixMarketError,
    MatrixMarketErrorKind, MatrixMarketScalar,
};
mod matrix_market;
