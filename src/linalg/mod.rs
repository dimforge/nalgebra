//! [Reexported at the root of this crate.] Factorization of real matrices.

pub mod balancing;
mod bidiagonal;
mod cholesky;
mod convolution;
mod determinant;
mod exp;
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

//// FIXME: Not complete enough for publishing.
//// This handles only cases where each eigenvalue has multiplicity one.
// mod eigen;

pub use self::bidiagonal::*;
pub use self::cholesky::*;
pub use self::convolution::*;
pub use self::exp::*;
pub use self::full_piv_lu::*;
pub use self::hessenberg::*;
pub use self::lu::*;
pub use self::permutation_sequence::*;
pub use self::qr::*;
pub use self::schur::*;
pub use self::svd::*;
pub use self::symmetric_eigen::*;
pub use self::symmetric_tridiagonal::*;
