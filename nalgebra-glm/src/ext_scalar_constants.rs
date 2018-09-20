use approx::AbsDiffEq;
use na::Real;

pub fn epsilon<N: AbsDiffEq<Epsilon = N>>() -> N {
    N::default_epsilon()
}

pub fn pi<N: Real>() -> N {
    N::pi()
}
