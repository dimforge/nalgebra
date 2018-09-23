use na::{Scalar, DefaultAllocator};

use traits::{Alloc, Dimension};
use aliases::{TVec, TMat};

/// The `index`-th column of the matrix `m`.
pub fn column<N: Scalar, R: Dimension, C: Dimension>(m: &TMat<N, R, C>, index: usize) -> TVec<N, R>
    where DefaultAllocator: Alloc<N, R, C> {
    m.column(index).into_owned()
}

/// Sets to `x` the `index`-th column of the matrix `m`.
pub fn set_column<N: Scalar, R: Dimension, C: Dimension>(m: &TMat<N, R, C>, index: usize, x: &TVec<N, R>) -> TMat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = m.clone();
    res.set_column(index, x);
    res
}

/// The `index`-th row of the matrix `m`.
pub fn row<N: Scalar, R: Dimension, C: Dimension>(m: &TMat<N, R, C>, index: usize) -> TVec<N, C>
    where DefaultAllocator: Alloc<N, R, C> {
    m.row(index).into_owned().transpose()
}

/// Sets to `x` the `index`-th row of the matrix `m`.
pub fn set_row<N: Scalar, R: Dimension, C: Dimension>(m: &TMat<N, R, C>, index: usize, x: &TVec<N, C>) -> TMat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = m.clone();
    res.set_row(index, &x.transpose());
    res
}
