/// Traits of objects which can be put in homogeneous coordinates.
pub trait ToHomogeneous<U>
{
    /// Gets the homogeneous coordinates version of this object.
    fn to_homogeneous(&self) -> U;
}

/// Traits of objects which can be build from an homogeneous coordinate representation.
pub trait FromHomogeneous<U>
{
    /// Builds an object with its homogeneous coordinate version. Note this it is not required for
    /// `from` to be the iverse of `to_homogeneous`. Typically, `from` will remove some informations
    /// unrecoverable by `to_homogeneous`.
    fn from(&U) -> Self;
}
