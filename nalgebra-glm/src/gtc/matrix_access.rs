use na::{DefaultAllocator, Scalar};

use aliases::{TMat, TVec};
use traits::{Alloc, Dimension};

/// The `index`-th column of the matrix `m`.
///
/// # See also:
///
/// * [`row`](fn.row.html)
/// * [`set_column`](fn.set_column.html)
/// * [`set_row`](fn.set_row.html)
pub fn column<N: Scalar, R: Dimension, C: Dimension>(
    m: &TMat<N, R, C>,
    index: usize,
) -> TVec<N, R>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    m.column(index).into_owned()
}

/// Sets to `x` the `index`-th column of the matrix `m`.
///
/// # See also:
///
/// * [`column`](fn.column.html)
/// * [`row`](fn.row.html)
/// * [`set_row`](fn.set_row.html)
pub fn set_column<N: Scalar, R: Dimension, C: Dimension>(
    m: &TMat<N, R, C>,
    index: usize,
    x: &TVec<N, R>,
) -> TMat<N, R, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = m.clone();
    res.set_column(index, x);
    res
}

/// The `index`-th row of the matrix `m`.
///
/// # See also:
///
/// * [`column`](fn.column.html)
/// * [`set_column`](fn.set_column.html)
/// * [`set_row`](fn.set_row.html)
pub fn row<N: Scalar, R: Dimension, C: Dimension>(m: &TMat<N, R, C>, index: usize) -> TVec<N, C>
where DefaultAllocator: Alloc<N, R, C> {
    m.row(index).into_owned().transpose()
}

/// Sets to `x` the `index`-th row of the matrix `m`.
///
/// # See also:
///
/// * [`column`](fn.column.html)
/// * [`row`](fn.row.html)
/// * [`set_column`](fn.set_column.html)
pub fn set_row<N: Scalar, R: Dimension, C: Dimension>(
    m: &TMat<N, R, C>,
    index: usize,
    x: &TVec<N, C>,
) -> TMat<N, R, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = m.clone();
    res.set_row(index, &x.transpose());
    res
}
