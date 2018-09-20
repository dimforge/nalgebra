use na::{Scalar, DimName, DefaultAllocator};

use traits::Alloc;
use aliases::Mat;

pub fn affineInverse<N: Scalar, D: DimName>(m: &Mat<N, D, D>) -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    unimplemented!()
}

pub fn inverseTranspose<N: Scalar, D: DimName>(m: &Mat<N, D, D>) -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    unimplemented!()
}