//! Matrix factorization for sparse matrices.
//!
//! Currently, the only factorization provided here is the [`CscCholesky`] factorization.
mod cholesky;
mod lu;

pub use cholesky::*;
pub use lu::*;
