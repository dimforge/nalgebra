use na::{Scalar, DefaultAllocator};

use traits::{Alloc, Dimension};
use aliases::{Vec, Mat};

pub fn column<N: Scalar, R: Dimension, C: Dimension>(m: &Mat<N, R, C>, index: usize) -> Vec<N, R>
    where DefaultAllocator: Alloc<N, R, C> {
    m.column(index).into_owned()
}

pub fn set_column<N: Scalar, R: Dimension, C: Dimension>(m: &Mat<N, R, C>, index: usize, x: &Vec<N, R>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = m.clone();
    res.set_column(index, x);
    res
}

pub fn row<N: Scalar, R: Dimension, C: Dimension>(m: &Mat<N, R, C>, index: usize) -> Vec<N, C>
    where DefaultAllocator: Alloc<N, R, C> {
    m.row(index).into_owned().transpose()
}

pub fn set_row<N: Scalar, R: Dimension, C: Dimension>(m: &Mat<N, R, C>, index: usize, x: &Vec<N, C>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    let mut res = m.clone();
    res.set_row(index, &x.transpose());
    res
}
