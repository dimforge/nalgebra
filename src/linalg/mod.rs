//! [Reexported at the root of this crate.] Factorization of real matrices.

pub mod balancing;
mod bidiagonal;
mod cholesky;
mod determinant;
mod full_piv_lu;
pub mod givens;
mod hessenberg;
pub mod householder;
mod inverse;
mod lu;
mod permutation_sequence;
mod qr;
mod schur;
mod solve;
mod svd;
mod symmetric_eigen;
mod symmetric_tridiagonal;
mod convolution;

//// FIXME: Not complete enough for publishing.
//// This handles only cases where each eigenvalue has multiplicity one.
// mod eigen;
pub use {
    self::bidiagonal::*,
    self::cholesky::*,
    self::full_piv_lu::*,
    self::hessenberg::*,
    self::lu::*,
    self::permutation_sequence::*,
    self::qr::*,
    self::schur::*,
    self::svd::*,
    self::symmetric_eigen::*,
    self::symmetric_tridiagonal::*,
    self::convolution::*
};
