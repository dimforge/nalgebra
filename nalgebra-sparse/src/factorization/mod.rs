//! Matrix factorization for sparse matrices.
//!
//! Currently, the only factorization provided here is the [`CscCholesky`] factorization.
mod cholesky;

pub use cholesky::*;
