use std::ops::{
    Mul,
    MulAssign,
    Div,
    DivAssign
};

use crate::Scaling;
use crate::{ClosedDiv, ClosedMul, SVector, Scalar, Point};

impl<T, const D: usize> Mul<SVector<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedMul
{
    type Output = Scaling<T, D>;

    fn mul(self, rhs: SVector<T, D>) -> Self::Output
    {
        return Scaling::<T, D>(self.0.component_mul(&rhs));
    }
}

impl<T, const D: usize> MulAssign<SVector<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedMul
{
    fn mul_assign(&mut self, rhs: SVector<T, D>)
    {
        self.0.component_mul_assign(&rhs);
    }
}

impl<T, const D: usize> Div<SVector<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedDiv
{
    type Output = Scaling<T, D>;

    fn div(self, rhs: SVector<T, D>) -> Self::Output
    {
        return Scaling::<T, D>(self.0.component_div(&rhs));
    }
}

impl<T, const D: usize> DivAssign<SVector<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedDiv
{
    fn div_assign(&mut self, rhs: SVector<T, D>)
    {
        self.0.component_div_assign(&rhs);
    }
}

impl<T, const D: usize> Mul<Scaling<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedMul
{
    type Output = Scaling<T, D>;

    fn mul(self, rhs: Scaling<T, D>) -> Self::Output
    {
        return Scaling::<T, D>(self.0.component_mul(&rhs.0));
    }
}

impl<T, const D: usize> MulAssign<Scaling<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedMul
{
    fn mul_assign(&mut self, rhs: Scaling<T, D>)
    {
        self.0.component_mul_assign(&rhs.0);
    }
}

impl<T, const D: usize> Div<Scaling<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedDiv
{
    type Output = Scaling<T, D>;

    fn div(self, rhs: Scaling<T, D>) -> Self::Output
    {
        return Scaling::<T, D>(self.0.component_div(&rhs.0));
    }
}

impl<T, const D: usize> DivAssign<Scaling<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedDiv
{
    fn div_assign(&mut self, rhs: Scaling<T, D>)
    {
        self.0.component_div_assign(&rhs.0);
    }
}

impl<T, const D: usize> Mul<Point<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedMul
{
    type Output = Point<T, D>;

    fn mul(self, rhs: Point<T, D>) -> Self::Output
    {
        return Point::from(self.0.component_mul(&rhs.coords));
    }
}

impl<T, const D: usize> Div<Point<T, D>> for Scaling<T, D>
    where T: Scalar + ClosedDiv
{
    type Output = Point<T, D>;

    fn div(self, rhs: Point<T, D>) -> Self::Output
    {
        return Point::from(self.0.component_div(&rhs.coords));
    }
}
