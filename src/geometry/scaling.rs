use std::ops::{
    Mul,
    MulAssign,
    Div,
    DivAssign
};

use crate::{ClosedDiv, ClosedMul, DefaultAllocator, DimName, OVector, allocator::Allocator, Scalar, OPoint, Const};

/// A scaling
pub type Scaling<T, const D: usize> = OScaling<T, Const<D>>;

/// An owned scaling represents a non-uniform scale transformation
pub struct OScaling<T: Scalar, D: DimName>(OVector<T, D>) where DefaultAllocator: Allocator<T, D>;

impl<T: Scalar> OScaling<T, Const<1_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T) -> OScaling<T, Const<1_usize>>
    {
        return OScaling(OVector::<T, Const<1_usize>>::new(x));
    }
}

impl<T: Scalar> OScaling<T, Const<2_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T) -> OScaling<T, Const<2_usize>>
    {
        return OScaling(OVector::<T, Const<2_usize>>::new(x, y));
    }
}

impl<T: Scalar> OScaling<T, Const<3_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T) -> OScaling<T, Const<3_usize>>
    {
        return OScaling(OVector::<T, Const<3_usize>>::new(x, y, z));
    }
}

impl<T: Scalar> OScaling<T, Const<4_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T) -> OScaling<T, Const<4_usize>>
    {
        return OScaling(OVector::<T, Const<4_usize>>::new(x, y, z, w));
    }
}

impl<T: Scalar> OScaling<T, Const<5_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T, a: T) -> OScaling<T, Const<5_usize>>
    {
        return OScaling(OVector::<T, Const<5_usize>>::new(x, y, z, w, a));
    }
}

impl<T: Scalar> OScaling<T, Const<6_usize>>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> OScaling<T, Const<6_usize>>
    {
        return OScaling(OVector::<T, Const<6_usize>>::new(x, y, z, w, a, b));
    }
}

impl<T, D: DimName> Mul<OVector<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedMul, DefaultAllocator: Allocator<T, D>
{
    type Output = OScaling<T, D>;

    fn mul(self, rhs: OVector<T, D>) -> Self::Output
    {
        return OScaling::<T, D>(self.0.component_mul(&rhs));
    }
}

impl<T, D: DimName> MulAssign<OVector<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedMul, DefaultAllocator: Allocator<T, D>
{
    fn mul_assign(&mut self, rhs: OVector<T, D>)
    {
        self.0.component_mul_assign(&rhs);
    }
}

impl<T, D: DimName> Div<OVector<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedDiv, DefaultAllocator: Allocator<T, D>
{
    type Output = OScaling<T, D>;

    fn div(self, rhs: OVector<T, D>) -> Self::Output
    {
        return OScaling::<T, D>(self.0.component_div(&rhs));
    }
}

impl<T, D: DimName> DivAssign<OVector<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedDiv, DefaultAllocator: Allocator<T, D>
{
    fn div_assign(&mut self, rhs: OVector<T, D>)
    {
        self.0.component_div_assign(&rhs);
    }
}

impl<T, D: DimName> Mul<OScaling<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedMul, DefaultAllocator: Allocator<T, D>
{
    type Output = OScaling<T, D>;

    fn mul(self, rhs: OScaling<T, D>) -> Self::Output
    {
        return OScaling::<T, D>(self.0.component_mul(&rhs.0));
    }
}

impl<T, D: DimName> MulAssign<OScaling<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedMul, DefaultAllocator: Allocator<T, D>
{
    fn mul_assign(&mut self, rhs: OScaling<T, D>)
    {
        self.0.component_mul_assign(&rhs.0);
    }
}

impl<T, D: DimName> Div<OScaling<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedDiv, DefaultAllocator: Allocator<T, D>
{
    type Output = OScaling<T, D>;

    fn div(self, rhs: OScaling<T, D>) -> Self::Output
    {
        return OScaling::<T, D>(self.0.component_div(&rhs.0));
    }
}

impl<T, D: DimName> DivAssign<OScaling<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedDiv, DefaultAllocator: Allocator<T, D>
{
    fn div_assign(&mut self, rhs: OScaling<T, D>)
    {
        self.0.component_div_assign(&rhs.0);
    }
}

impl<T, D: DimName> Mul<OPoint<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedMul, DefaultAllocator: Allocator<T, D>
{
    type Output = OPoint<T, D>;

    fn mul(self, rhs: OPoint<T, D>) -> Self::Output
    {
        return OPoint::from(self.0.component_mul(&rhs.coords));
    }
}

impl<T, D: DimName> Div<OPoint<T, D>> for OScaling<T, D>
    where T: Scalar + ClosedDiv, DefaultAllocator: Allocator<T, D>
{
    type Output = OPoint<T, D>;

    fn div(self, rhs: OPoint<T, D>) -> Self::Output
    {
        return OPoint::from(self.0.component_div(&rhs.coords));
    }
}

impl<T, D: DimName> From<OVector<T, D>> for OScaling<T, D>
    where T: Scalar, DefaultAllocator: Allocator<T, D>
{
    fn from(other: OVector<T, D>) -> Self
    {
        return OScaling::<T, D>(other);
    }
}

impl<T, D: DimName> Into<OVector<T, D>> for OScaling<T, D>
    where T: Scalar, DefaultAllocator: Allocator<T, D>
{
    fn into(self) -> OVector<T, D>
    {
        return self.0;
    }
}
