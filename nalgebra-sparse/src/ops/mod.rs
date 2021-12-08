//! Sparse matrix arithmetic operations.
//!
//! This module contains a number of routines for sparse matrix arithmetic. These routines are
//! primarily intended for "expert usage". Most users should prefer to use standard
//! `std::ops` operations for simple and readable code when possible. The routines provided here
//! offer more control over allocation, and allow fusing some low-level operations for higher
//! performance.
//!
//! The available operations are organized by backend. Currently, only the [`serial`] backend
//! is available. In the future, backends that expose parallel operations may become available.
//! All `std::ops` implementations will remain single-threaded and powered by the
//! `serial` backend.
//!
//! Many routines are able to implicitly transpose matrices involved in the operation.
//! For example, the routine [`spadd_csr_prealloc`](serial::spadd_csr_prealloc) performs the
//! operation `C <- beta * C + alpha * op(A)`. Here `op(A)` indicates that the matrix `A` can
//! either be used as-is or transposed. The notation `op(A)` is represented in code by the
//! [`Op`] enum.
//!
//! # Available `std::ops` implementations
//!
//! ## Binary operators
//!
//! The below table summarizes the currently supported binary operators between matrices.
//! In general, binary operators between sparse matrices are only supported if both matrices
//! are stored in the same format. All supported binary operators are implemented for
//! all four combinations of values and references.
//!
//! <table>
//!     <tr>
//!         <th>LHS (down) \ RHS (right)</th>
//!         <th>COO</th>
//!         <th>CSR</th>
//!         <th>CSC</th>
//!         <th>Dense</th>
//!     </tr>
//!     <tr>
//!         <th>COO</th>
//!         <td></td>
//!         <td></td>
//!         <td></td>
//!         <td></td>
//!     </tr>
//!     <tr>
//!         <th>CSR</th>
//!         <td></td>
//!         <td>+ - *</td>
//!         <td></td>
//!         <td>*</td>
//!     </tr>
//!     <tr>
//!         <th>CSC</th>
//!         <td></td>
//!         <td></td>
//!         <td>+ - *</td>
//!         <td>*</td>
//!     </tr>
//!     <tr>
//!         <th>Dense</th>
//!         <td></td>
//!         <td></td>
//!         <td></td>
//!         <td>+ - *</td>
//!     </tr>
//! </table>
//!
//! As can be seen from the table, only `CSR * Dense` and `CSC * Dense` are supported.
//! The other way around, i.e. `Dense * CSR` and `Dense * CSC` are not implemented.
//!
//! Additionally, [CsrMatrix](`crate::cs::CsrMatrix`) and [CooMatrix](`crate::coo::CooMatrix`)
//! support multiplication with scalars, in addition to division by a scalar.
//! Note that only `Matrix * Scalar` works in a generic context, although `Scalar * Matrix`
//! has been implemented for many of the built-in arithmetic types. This is due to a fundamental
//! restriction of the Rust type system. Therefore, in generic code you will need to always place
//! the matrix on the left-hand side of the multiplication.
//!
//! ## Unary operators
//!
//! The following table lists currently supported unary operators.
//!
//! | Format   | AddAssign\<Matrix\> | MulAssign\<Matrix\> | MulAssign\<Scalar\> | Neg    |
//! | -------- | -----------------   | -----------------   | ------------------- | ------ |
//! | COO      |                     |                     |                     |        |
//! | CSR      |                     |                     | x                   | x      |
//! | CSC      |                     |                     | x                   | x      |
//! |
//! # Example usage
//!
//! For example, consider the case where you want to compute the expression
//! `C <- 3.0 * C + 2.0 * A^T * B`, where `A`, `B`, `C` are matrices and `A^T` is the transpose
//! of `A`. The simplest way to write this is:
//!
//! ```
//! # use nalgebra_sparse::cs::CsrMatrix;
//! let a = CsrMatrix::<f32>::identity(10);
//! let b = CsrMatrix::<f32>::identity(10);
//! let c = CsrMatrix::<f32>::identity(10);
//! //let d = 3.0 * c + (2.0 * a.transpose() * b);
//! let d = a.transpose() * b.transpose();
//! ```
//! This is simple and straightforward to read, and therefore the recommended way to implement
//! it. However, if you have determined that this is a performance bottleneck of your application,
//! it may be possible to speed things up. First, let's see what's going on here. The `std`
//! operations are evaluated eagerly. This means that the following steps take place:
//!
//! 1. Evaluate `let c_temp = 3.0 * c`. This requires scaling all values of the matrix.
//! 2. Evaluate `let a_t = a.transpose()` into a new temporary matrix.
//! 3. Evaluate `let a_t_b = a_t * b` into a new temporary matrix.
//! 4. Evaluate `let a_t_b_scaled = 2.0 * a_t_b`. This requires scaling all values of the matrix.
//! 5. Evaluate `c = c_temp + a_t_b_scaled`.
//!
//! An alternative way to implement this expression (here using CSR matrices) is:
//!
//! ```
//! # use nalgebra_sparse::cs::CsrMatrix;
//! use nalgebra_sparse::ops::{serial::spmm_csr_csr};
//!
//! let a = CsrMatrix::identity(10);
//! let b = CsrMatrix::identity(10);
//!
//! // Evaluate the expression `c <- a^T * b
//! let c = spmm_csr_csr(a.transpose(), b)
//!     .expect("We assume that the patterns of A and B are able to accommodate the result.");
//! ```
//! Compared to the simpler example, this snippet is harder to read, but it calls a single
//! computational kernel that avoids many of the intermediate steps listed out before. Therefore
//! directly calling kernels may sometimes lead to better performance. However, this should
//! always be verified by performance profiling!

mod impl_std_ops;
pub mod serial;
