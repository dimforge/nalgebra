use num::Num;
use traits::{Alloc, Number};
use na::{Scalar, Real, DimName, DefaultAllocator, U1};
use na::allocator::Allocator;

use aliases::{Mat, Vec};


//pub fn determinant<N: Real, D: DimName>(m: &Mat<N, D, D>) -> N
//    where DefaultAllocator: Allocator<N, D, D> {
//    m.determinant()
//}

pub fn inverse<N: Real, D: DimName>(m: &Mat<N, D, D>) -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    m.clone().try_inverse().unwrap_or(Mat::<N, D, D>::zeros())
}

pub fn matrix_comp_mult<N: Number, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    x.component_mul(y)
}

pub fn outer_product<N: Number, R: DimName, C: DimName>(c: &Vec<N, R>, r: &Vec<N, C>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
    c * r.transpose()
}

pub fn transpose<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>) -> Mat<N, C, R>
    where DefaultAllocator: Alloc<N, R, C> {
    x.transpose()
}