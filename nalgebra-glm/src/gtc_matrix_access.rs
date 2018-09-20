use na::{Scalar, DimName, DefaultAllocator};

use traits::Alloc;
use aliases::{Vec, Mat};

pub fn column<N: Scalar, R: DimName, C: DimName>(m: &Mat<N, R, C>, index: usize) -> Vec<N, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn column2<N: Scalar, R: DimName, C: DimName>(m: &Mat<N, R, C>, index: usize, x: &Vec<N, C>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn row<N: Scalar, R: DimName, C: DimName>(m: &Mat<N, R, C>, index: usize) -> Vec<N, R>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn row2<N: Scalar, R: DimName, C: DimName>(m: &Mat<N, R, C>, index: usize, x: &Vec<N, R>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}
