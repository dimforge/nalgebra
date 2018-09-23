use approx::AbsDiffEq;
use na::Real;

/// Default epsilon value used for approximate comparison.
pub fn epsilon<N: AbsDiffEq<Epsilon = N>>() -> N {
    N::default_epsilon()
}

/// The value of PI.
pub fn pi<N: Real>() -> N {
    N::pi()
}
