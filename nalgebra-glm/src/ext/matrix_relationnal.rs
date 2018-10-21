use na::DefaultAllocator;

use aliases::{TMat, TVec};
use traits::{Alloc, Dimension, Number};

/// Perform a component-wise equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn equal_columns<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) == y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
    epsilon: N,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    equal_columns_eps_vec(x, y, &TVec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison on each matrix column `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps_vec<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
    epsilon: &TVec<N, C>,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() < TVec::<_, R>::repeat(epsilon[i])
    }

    res
}

/// Perform a component-wise not-equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn not_equal_columns<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) != y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
    epsilon: N,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    not_equal_columns_eps_vec(x, y, &TVec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison of `|x - y| >= epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps_vec<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
    epsilon: &TVec<N, C>,
) -> TVec<bool, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    let mut res = TVec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() >= TVec::<_, R>::repeat(epsilon[i])
    }

    res
}
