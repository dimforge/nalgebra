use traits::row::Row;
use traits::col::Col;
use traits::rlmul::{RMul, LMul};

/// Trait of matrix. A matrix must have lines and columns.
pub trait Mat<R, C> : Row<R> + Col<C> + RMul<R> + LMul<C> { }

impl<M: Row<R> + Col<C> + RMul<R> + LMul<C>, R, C> Mat<R, C> for M;
