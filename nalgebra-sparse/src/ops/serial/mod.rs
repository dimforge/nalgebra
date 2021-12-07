//! Serial sparse matrix arithmetic routines.
//!
//! All routines are single-threaded.
//!
//! Some operations have the `prealloc` suffix. This means that they expect that the sparsity
//! pattern of the output matrix has already been pre-allocated: that is, the pattern of the result
//! of the operation fits entirely in the output pattern. In the future, there will also be
//! some operations which will be able to dynamically adapt the output pattern to fit the
//! result, but these have yet to be implemented.

pub mod scalar;
pub mod spadd;
pub mod spmm;
pub(crate) mod spsolve;
pub mod spsub;
