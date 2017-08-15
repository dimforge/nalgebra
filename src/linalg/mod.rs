//! [Reexported at the root of this crate.] Factorization of real matrices.

mod solve;
mod determinant;
mod inverse;
pub mod householder;
pub mod givens;
pub mod balancing;
mod permutation_sequence;
mod qr;
mod hessenberg;
mod bidiagonal;
mod symmetric_tridiagonal;
mod cholesky;
mod lu;
mod full_piv_lu;
mod schur;
mod svd;
mod symmetric_eigen;

//// FIXME: Not complete enough for publishing.
//// This handles only cases where each eigenvalue has multiplicity one.
// mod eigen;

pub use self::permutation_sequence::*;
pub use self::qr::*;
pub use self::hessenberg::*;
pub use self::bidiagonal::*;
pub use self::cholesky::*;
pub use self::lu::*;
pub use self::full_piv_lu::*;
pub use self::schur::*;
pub use self::svd::*;
pub use self::symmetric_tridiagonal::*;
pub use self::symmetric_eigen::*;
