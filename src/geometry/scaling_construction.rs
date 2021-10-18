use crate::Scaling;
use crate::{SVector, Scalar};

impl<T: Scalar> Scaling<T, 1>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T) -> Scaling<T, 1>
    {
        return Scaling(SVector::<T, 1>::new(x));
    }
}

impl<T: Scalar> Scaling<T, 2>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T) -> Scaling<T, 2>
    {
        return Scaling(SVector::<T, 2>::new(x, y));
    }
}

impl<T: Scalar> Scaling<T, 3>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T) -> Scaling<T, 3>
    {
        return Scaling(SVector::<T, 3>::new(x, y, z));
    }
}

impl<T: Scalar> Scaling<T, 4>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T) -> Scaling<T, 4>
    {
        return Scaling(SVector::<T, 4>::new(x, y, z, w));
    }
}

impl<T: Scalar> Scaling<T, 5>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T, a: T) -> Scaling<T, 5>
    {
        return Scaling(SVector::<T, 5>::new(x, y, z, w, a));
    }
}

impl<T: Scalar> Scaling<T, 6>
{
    /// Initializes this scaling from its components.
    pub fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Scaling<T, 6>
    {
        return Scaling(SVector::<T, 6>::new(x, y, z, w, a, b));
    }
}
