use crate::geometry::{Rotation, UnitComplex, UnitQuaternion};
use crate::{CVectorN, Const, Point, Scalar, SimdRealField, Unit, VectorN};

use simba::scalar::ClosedMul;

/// Trait implemented by rotations that can be used inside of an `Isometry` or `Similarity`.
pub trait AbstractRotation<N: Scalar, const D: usize>: PartialEq + ClosedMul + Clone {
    /// The rotation identity.
    fn identity() -> Self;
    /// The rotation inverse.
    fn inverse(&self) -> Self;
    /// Change `self` to its inverse.
    fn inverse_mut(&mut self);
    /// Apply the rotation to the given vector.
    fn transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D>;
    // where
    //     DefaultAllocator: Allocator<N, D>;
    /// Apply the rotation to the given point.
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>;
    // where
    //     DefaultAllocator: Allocator<N, D>;
    /// Apply the inverse rotation to the given vector.
    fn inverse_transform_vector(&self, v: &VectorN<N, Const<D>>) -> VectorN<N, Const<D>>;
    // where
    //     DefaultAllocator: Allocator<N, D>;
    /// Apply the inverse rotation to the given unit vector.
    fn inverse_transform_unit_vector(&self, v: &Unit<CVectorN<N, D>>) -> Unit<CVectorN<N, D>>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        Unit::new_unchecked(self.inverse_transform_vector(&**v))
    }
    /// Apply the inverse rotation to the given point.
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>;
    // where
    //     DefaultAllocator: Allocator<N, D>;
}

impl<N: SimdRealField, const D: usize> AbstractRotation<N, D> for Rotation<N, D>
where
    N::Element: SimdRealField,
    // DefaultAllocator: Allocator<N, D, D>,
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
    fn transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, D>) -> Point<N, D>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_unit_vector(&self, v: &Unit<CVectorN<N, D>>) -> Unit<CVectorN<N, D>>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        self.inverse_transform_unit_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, D>) -> Point<N, D>
// where
    //     DefaultAllocator: Allocator<N, D>,
    {
        self.inverse_transform_point(p)
    }
}

impl<N: SimdRealField> AbstractRotation<N, 3> for UnitQuaternion<N>
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
    fn transform_vector(&self, v: &CVectorN<N, 3>) -> CVectorN<N, 3> {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, 3>) -> Point<N, 3> {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &CVectorN<N, 3>) -> CVectorN<N, 3> {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, 3>) -> Point<N, 3> {
        self.inverse_transform_point(p)
    }
}

impl<N: SimdRealField> AbstractRotation<N, 2> for UnitComplex<N>
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
    fn transform_vector(&self, v: &CVectorN<N, 2>) -> CVectorN<N, 2> {
        self * v
    }

    #[inline]
    fn transform_point(&self, p: &Point<N, 2>) -> Point<N, 2> {
        self * p
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &CVectorN<N, 2>) -> CVectorN<N, 2> {
        self.inverse_transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point<N, 2>) -> Point<N, 2> {
        self.inverse_transform_point(p)
    }
}
