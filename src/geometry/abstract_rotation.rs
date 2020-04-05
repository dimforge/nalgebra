use crate::allocator::Allocator;
use crate::geometry::{Rotation, UnitComplex, UnitQuaternion};
use crate::{DefaultAllocator, DimName, Point, Scalar, SimdRealField, VectorN, U2, U3};

use simba::scalar::ClosedMul;

/// Trait implemented by rotations that can be used inside of an `Isometry` or `Similarity`.
pub trait AbstractRotation<N: Scalar, D: DimName>: PartialEq + ClosedMul + Clone {
    /// The rotation identity.
    fn identity() -> Self;
    /// The rotation inverse.
    fn inverse(&self) -> Self;
    /// Change `self` to its inverse.
    fn inverse_mut(&mut self);
    /// Apply the rotation to the given vector.
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>;
    /// Apply the rotation to the given point.
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where
        DefaultAllocator: Allocator<N, D>;
    /// Apply the inverse rotation to the given vector.
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>;
    /// Apply the inverse rotation to the given point.
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where
        DefaultAllocator: Allocator<N, D>;
}

impl<N: SimdRealField, D: DimName> AbstractRotation<N, D> for Rotation<N, D>
where
    N::Element: SimdRealField,
    DefaultAllocator: Allocator<N, D, D>,
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
    where
        DefaultAllocator: Allocator<N, D>,
    {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        self.inverse_transform_point(p)
    }
}

impl<N: SimdRealField> AbstractRotation<N, U3> for UnitQuaternion<N>
where
    N::Element: SimdRealField,
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

impl<N: SimdRealField> AbstractRotation<N, U2> for UnitComplex<N>
where
    N::Element: SimdRealField,
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
