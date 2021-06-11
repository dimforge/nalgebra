use crate::aliases::{TMat, TVec};
use crate::traits::Number;

/// Perform a component-wise equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn equal_columns<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
) -> TVec<bool, C> {
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C {
        res[i] = x.column(i) == y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
    epsilon: T,
) -> TVec<bool, C> {
    equal_columns_eps_vec(x, y, &TVec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison on each matrix column `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps_vec<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
    epsilon: &TVec<T, C>,
) -> TVec<bool, C> {
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C {
        res[i] = (x.column(i) - y.column(i)).abs() < TVec::<_, R>::repeat(epsilon[i])
    }

    res
}

/// Perform a component-wise not-equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn not_equal_columns<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
) -> TVec<bool, C> {
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C {
        res[i] = x.column(i) != y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
    epsilon: T,
) -> TVec<bool, C> {
    not_equal_columns_eps_vec(x, y, &TVec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison of `|x - y| >= epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps_vec<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
    epsilon: &TVec<T, C>,
) -> TVec<bool, C> {
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C {
        res[i] = (x.column(i) - y.column(i)).abs() >= TVec::<_, R>::repeat(epsilon[i])
    }

    res
}
