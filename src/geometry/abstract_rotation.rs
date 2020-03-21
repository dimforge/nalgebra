use crate::allocator::Allocator;
use crate::geometry::{Rotation, UnitComplex, UnitQuaternion};
use crate::{DefaultAllocator, DimName, Point, RealField, Scalar, VectorN, U2, U3};

use simba::scalar::ClosedMul;

pub trait AbstractRotation<N: Scalar, D: DimName>: PartialEq + ClosedMul + Clone {
    fn identity() -> Self;
    fn inverse(&self) -> Self;
    fn inverse_mut(&mut self);
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where DefaultAllocator: Allocator<N, D>;
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where DefaultAllocator: Allocator<N, D>;
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where DefaultAllocator: Allocator<N, D>;
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where DefaultAllocator: Allocator<N, D>;
}

impl<N: RealField, D: DimName> AbstractRotation<N, D> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }

    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where DefaultAllocator: Allocator<N, D> {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where DefaultAllocator: Allocator<N, D> {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where DefaultAllocator: Allocator<N, D> {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where DefaultAllocator: Allocator<N, D> {
        self.inverse_transform_point(p)
    }
}

impl<N: RealField> AbstractRotation<N, U3> for UnitQuaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }

    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, U3>) -> VectorN<N, U3> {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, U3>) -> Point<N, U3> {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, U3>) -> VectorN<N, U3> {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, U3>) -> Point<N, U3> {
        self.inverse_transform_point(p)
    }
}

impl<N: RealField> AbstractRotation<N, U2> for UnitComplex<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }

    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, U2>) -> VectorN<N, U2> {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, U2>) -> Point<N, U2> {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, U2>) -> VectorN<N, U2> {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, U2>) -> Point<N, U2> {
        self.inverse_transform_point(p)
    }
}
