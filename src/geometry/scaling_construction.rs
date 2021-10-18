use crate::OScaling;
use crate::{OVector, Scalar, Const};

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
