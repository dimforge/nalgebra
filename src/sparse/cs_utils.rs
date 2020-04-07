use crate::allocator::Allocator;
use crate::{DefaultAllocator, Dim, VectorN};

pub fn cumsum<D: Dim>(a: &mut VectorN<usize, D>, b: &mut VectorN<usize, D>) -> usize
where
    DefaultAllocator: Allocator<usize, D>,
{
    assert!(a.len() == b.len());
    let mut sum = 0;

    for i in 0..a.len() {
        b[i] = sum;
        sum += a[i];
        a[i] = b[i];
    }

    sum
}
