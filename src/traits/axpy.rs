use alga::general::Field;

use core::{Scalar, Matrix};
use core::dimension::{Dim, DimName, U1};
use core::storage::StorageMut;

use geometry::PointBase;

/// Operation that combines scalar multiplication and vector addition.
pub trait Axpy<A> {
    /// Computes `self = a * x + self`.
    fn axpy(&mut self, a: A, x: &Self);
}

impl<N, R: Dim, C: Dim, S> Axpy<N> for Matrix<N, R, C, S>
where N: Scalar + Field,
      S: StorageMut<N, R, C> {
    #[inline]
    fn axpy(&mut self, a: N, x: &Self) {
        for (me, x) in self.iter_mut().zip(x.iter()) {
            *me += *x * a;
        }
    }
}


impl<N, D: DimName, S> Axpy<N> for PointBase<N, D, S>
where N: Scalar + Field,
      S: StorageMut<N, D, U1> {
    #[inline]
    fn axpy(&mut self, a: N, x: &Self) {
        for (me, x) in self.coords.iter_mut().zip(x.coords.iter()) {
            *me += *x * a;
        }
    }
}

// FIXME: implemente Axpy with matrices and transforms.
