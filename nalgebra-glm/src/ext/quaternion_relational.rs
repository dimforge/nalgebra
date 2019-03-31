use na::{RealField, U4};

use crate::aliases::{Qua, TVec};

/// Component-wise equality comparison between two quaternions.
pub fn quat_equal<N: RealField>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    crate::equal(&x.coords, &y.coords)
}

/// Component-wise approximate equality comparison between two quaternions.
pub fn quat_equal_eps<N: RealField>(x: &Qua<N>, y: &Qua<N>, epsilon: N) -> TVec<bool, U4> {
    crate::equal_eps(&x.coords, &y.coords, epsilon)
}

/// Component-wise non-equality comparison between two quaternions.
pub fn quat_not_equal<N: RealField>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    crate::not_equal(&x.coords, &y.coords)
}

/// Component-wise approximate non-equality comparison between two quaternions.
pub fn quat_not_equal_eps<N: RealField>(x: &Qua<N>, y: &Qua<N>, epsilon: N) -> TVec<bool, U4> {
    crate::not_equal_eps(&x.coords, &y.coords, epsilon)
}
