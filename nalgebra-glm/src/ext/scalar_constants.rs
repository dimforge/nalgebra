use approx::AbsDiffEq;
use na::Real;

/// Default epsilon value used for approximate comparison.
pub fn epsilon<N: AbsDiffEq<Epsilon = N>>() -> N {
    N::default_epsilon()
}

/// The value of PI.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn pi<N: Real>() -> N {
    N::pi()
}
