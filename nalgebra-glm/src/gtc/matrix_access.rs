use na::Scalar;

use crate::aliases::{TMat, TVec};

/// The `index`-th column of the matrix `m`.
///
/// # See also:
///
/// * [`row()`]
/// * [`set_column()`]
/// * [`set_row()`]
pub fn column<T: Scalar, const R: usize, const C: usize>(
    m: &TMat<T, R, C>,
    index: usize,
) -> TVec<T, R> {
    m.column(index).into_owned()
}

/// Sets to `x` the `index`-th column of the matrix `m`.
///
/// # See also:
///
/// * [`column()`]
/// * [`row()`]
/// * [`set_row()`]
pub fn set_column<T: Scalar, const R: usize, const C: usize>(
    m: &TMat<T, R, C>,
    index: usize,
    x: &TVec<T, R>,
) -> TMat<T, R, C> {
    let mut res = m.clone();
    res.set_column(index, x);
    res
}

/// The `index`-th row of the matrix `m`.
///
/// # See also:
///
/// * [`column()`]
/// * [`set_column()`]
/// * [`set_row()`]
pub fn row<T: Scalar, const R: usize, const C: usize>(
    m: &TMat<T, R, C>,
    index: usize,
) -> TVec<T, C> {
    m.row(index).into_owned().transpose()
}

/// Sets to `x` the `index`-th row of the matrix `m`.
///
/// # See also:
///
/// * [`column()`]
/// * [`row()`]
/// * [`set_column()`]
pub fn set_row<T: Scalar, const R: usize, const C: usize>(
    m: &TMat<T, R, C>,
    index: usize,
    x: &TVec<T, C>,
) -> TMat<T, R, C> {
    let mut res = m.clone();
    res.set_row(index, &x.transpose());
    res
}
