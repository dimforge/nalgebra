use na::{DimName, Scalar, DefaultAllocator};

use traits::Alloc;
use aliases::Vec;

pub fn epsilonEqual<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn epsilonEqual2<T>(x: T, y: T, epsilon: T) -> bool {
    unimplemented!()
}

pub fn epsilonNotEqual<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn epsilonNotEqual2<T>(x: T, y: T, epsilon: T) -> bool {
    unimplemented!()
}
