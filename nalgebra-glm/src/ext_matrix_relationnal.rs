use na::DefaultAllocator;

use aliases::{Vec, Mat};
use traits::{Alloc, Number, Dimension};


pub fn equal<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) == y.column(i)
    }

    res
}

pub fn equal_eps<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    equal_eps_vec(x, y, &Vec::<_, C>::repeat(epsilon))
}

pub fn equal_eps_vec<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() < Vec::<_, R>::repeat(epsilon[i])
    }

    res
}

pub fn not_equal<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = x.column(i) != y.column(i)
    }

    res
}

pub fn not_equal_eps<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    not_equal_eps_vec(x, y, &Vec::<_, C>::repeat(epsilon))
}

pub fn not_equal_eps_vec<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = Vec::<_, C>::repeat(false);

    for i in 0..C::dim() {
        res[i] = (x.column(i) - y.column(i)).abs() >= Vec::<_, R>::repeat(epsilon[i])
    }

    res
}
