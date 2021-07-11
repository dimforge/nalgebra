//! Routines for converting between sparse matrix formats.
//!
//! Most users should instead use the provided `From` implementations to convert between matrix
//! formats. Note that `From` implementations may not be available between all combinations of
//! sparse matrices.
//!
//! The following example illustrates how to convert between matrix formats with the `From`
//! implementations.
//!
//! ```
//! use nalgebra_sparse::{csr::CsrMatrix, csc::CscMatrix, coo::CooMatrix};
//! use nalgebra::DMatrix;
//!
//! // Conversion from dense
//! let dense = DMatrix::<f64>::identity(9, 8);
//! let csr = CsrMatrix::from(&dense);
//! let csc = CscMatrix::from(&dense);
//! let coo = CooMatrix::from(&dense);
//!
//! // CSR <-> CSC
//! let _ = CsrMatrix::from(&csc);
//! let _ = CscMatrix::from(&csr);
//!
//! // CSR <-> COO
//! let _ = CooMatrix::from(&csr);
//! let _ = CsrMatrix::from(&coo);
//!
//! // CSC <-> COO
//! let _ = CooMatrix::from(&csc);
//! let _ = CscMatrix::from(&coo);
//! ```
//!
//! The routines available here are able to provide more specialized APIs, giving
//! more control over the conversion process. The routines are organized by backends.
//! Currently, only the [`serial`] backend is available.
//! In the future, backends that offer parallel routines may become available.

pub mod serial;

mod impl_std_ops;
