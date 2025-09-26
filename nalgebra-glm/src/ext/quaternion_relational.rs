use crate::RealNumber;
use crate::aliases::{Qua, TVec};

/// Component-wise equality comparison between two quaternions.
pub fn quat_equal<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::equal(&x.coords, &y.coords)
}

/// Component-wise approximate equality comparison between two quaternions.
pub fn quat_equal_eps<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, epsilon: T) -> TVec<bool, 4> {
    crate::equal_eps(&x.coords, &y.coords, epsilon)
}

/// Component-wise non-equality comparison between two quaternions.
pub fn quat_not_equal<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::not_equal(&x.coords, &y.coords)
}

/// Component-wise approximate non-equality comparison between two quaternions.
pub fn quat_not_equal_eps<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, epsilon: T) -> TVec<bool, 4> {
    crate::not_equal_eps(&x.coords, &y.coords, epsilon)
}
