use na::{Real, U4};


use aliases::{Qua, Vec};

/// Component-wise equality comparison between two quaternions.
pub fn equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::equal(&x.coords, &y.coords)
}

/// Component-wise approximate equality comparison between two quaternions.
pub fn equal_eps<N: Real>(x: &Qua<N>, y: &Qua<N>, epsilon: N) -> Vec<bool, U4> {
    ::equal_eps(&x.coords, &y.coords, epsilon)
}

/// Component-wise non-equality comparison between two quaternions.
pub fn not_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::not_equal(&x.coords, &y.coords)
}

/// Component-wise approximate non-equality comparison between two quaternions.
pub fn not_equal_eps<N: Real>(x: &Qua<N>, y: &Qua<N>, epsilon: N) -> Vec<bool, U4> {
    ::not_equal_eps(&x.coords, &y.coords, epsilon)
}
