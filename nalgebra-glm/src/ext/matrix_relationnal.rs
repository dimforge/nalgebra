use na::DefaultAllocator;

use aliases::{Vec, Mat};
use traits::{Alloc, Number, Dimension};

/// Perform a component-wise equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn equal_columns<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) == y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    equal_columns_eps_vec(x, y, &Vec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is satisfied.
pub fn equal_columns_eps_vec<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() < Vec::<_, R>::repeat(epsilon[i])
    }

    res
}

/// Perform a component-wise not-equal-to comparison of two matrices.
///
/// Return a boolean vector which components value is True if this expression is satisfied per column of the matrices.
pub fn not_equal_columns<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) != y.column(i)
    }

    res
}

/// Returns the component-wise comparison of `|x - y| < epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    not_equal_columns_eps_vec(x, y, &Vec::<_, C>::repeat(epsilon))
}

/// Returns the component-wise comparison of `|x - y| >= epsilon`.
///
/// True if this expression is not satisfied.
pub fn not_equal_columns_eps_vec<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() >= Vec::<_, R>::repeat(epsilon[i])
    }

    res
}
