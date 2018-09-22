use approx::AbsDiffEq;
use na::Real;

/// Default epsilon value used for apporximate comparison.
pub fn epsilon<N: AbsDiffEq<Epsilon = N>>() -> N {
    N::default_epsilon()
}

/// The value of PI.
pub fn pi<N: Real>() -> N {
    N::pi()
}
