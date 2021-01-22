//! Sparse matrix arithmetic operations.
//!
//! TODO: Explain that users should prefer to use std ops unless they need to get more performance
//!
//! The available operations are organized by backend. Currently, only the [`serial`] backend
//! is available. In the future, backends that expose parallel operations may become available.
//!
//! Many routines are able to implicitly transpose matrices involved in the operation.
//! For example, the routine [`spadd_csr_prealloc`](serial::spadd_csr_prealloc) performs the
//! operation `C <- beta * C + alpha * op(A)`. Here `op(A)` indicates that the matrix `A` can
//! either be used as-is or transposed. The notation `op(A)` is represented in code by the
//! [`Op`] enum.

mod impl_std_ops;
pub mod serial;

/// Determines whether a matrix should be transposed in a given operation.
///
/// See the [module-level documentation](crate::ops) for the purpose of this enum.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op<T> {
    /// Indicates that the matrix should be used as-is.
    NoOp(T),
    /// Indicates that the matrix should be transposed.
    Transpose(T),
}

impl<T> Op<T> {
    /// Returns a reference to the inner value that the operation applies to.
    pub fn inner_ref(&self) -> &T {
        self.as_ref().into_inner()
    }

    /// Returns an `Op` applied to a reference of the inner value.
    pub fn as_ref(&self) -> Op<&T> {
        match self {
            Op::NoOp(obj) => Op::NoOp(&obj),
            Op::Transpose(obj) => Op::Transpose(&obj)
        }
    }

    /// Converts the underlying data type.
    pub fn convert<U>(self) -> Op<U>
        where T: Into<U>
    {
        self.map_same_op(T::into)
    }

    /// Transforms the inner value with the provided function, but preserves the operation.
    pub fn map_same_op<U, F: FnOnce(T) -> U>(self, f: F) -> Op<U> {
        match self {
            Op::NoOp(obj) => Op::NoOp(f(obj)),
            Op::Transpose(obj) => Op::Transpose(f(obj))
        }
    }

    /// Consumes the `Op` and returns the inner value.
    pub fn into_inner(self) -> T {
        match self {
            Op::NoOp(obj) | Op::Transpose(obj) => obj,
        }
    }

    /// Applies the transpose operation.
    ///
    /// This operation follows the usual semantics of transposition. In particular, double
    /// transposition is equivalent to no transposition.
    pub fn transposed(self) -> Self {
        match self {
            Op::NoOp(obj) => Op::Transpose(obj),
            Op::Transpose(obj) => Op::NoOp(obj)
        }
    }
}

impl<T> From<T> for Op<T> {
    fn from(obj: T) -> Self {
        Self::NoOp(obj)
    }
}

