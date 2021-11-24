//! Serial sparse matrix arithmetic routines.
//!
//! All routines are single-threaded.
//!
//! Some operations have the `prealloc` suffix. This means that they expect that the sparsity
//! pattern of the output matrix has already been pre-allocated: that is, the pattern of the result
//! of the operation fits entirely in the output pattern. In the future, there will also be
//! some operations which will be able to dynamically adapt the output pattern to fit the
//! result, but these have yet to be implemented.

mod spmm;

pub use spmm::{
    spmm_csc_csc, spmm_csc_csr, spmm_csc_dense, spmm_csr_csc, spmm_csr_csr, spmm_csr_dense,
    spmm_dense_csc, spmm_dense_csr,
};
