use na::{Scalar, DimName, DefaultAllocator};

use aliases::{Vec, Mat};
use traits::Alloc;


pub fn equal<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn equal_eps<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>,epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn equal_eps_vec<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn not_equal<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn not_equal_eps<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>,epsilon: N) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}

pub fn not_equal_eps_vec<N: Scalar, R: DimName, C: DimName>(x: &Mat<N, R, C>, y: &Mat<N, R, C>, epsilon: &Vec<N, C>) -> Vec<bool, C>
    where DefaultAllocator: Alloc<N, R, C> {
    unimplemented!()
}
