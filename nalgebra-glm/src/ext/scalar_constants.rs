use crate::RealNumber;
use approx::AbsDiffEq;

/// Default epsilon value used for approximate comparison.
pub fn epsilon<T: AbsDiffEq<Epsilon = T>>() -> T {
    T::default_epsilon()
}

/// The value of PI.
///
/// # See also:
///
/// * [`four_over_pi()`](crate::four_over_pi)
/// * [`half_pi()`](crate::half_pi)
/// * [`one_over_pi()`](crate::one_over_pi)
/// * [`one_over_two_pi()`](crate::one_over_two_pi)
/// * [`quarter_pi()`](crate::quarter_pi)
/// * [`root_half_pi()`](crate::root_half_pi)
/// * [`root_pi()`](crate::root_pi)
/// * [`root_two_pi()`](crate::root_two_pi)
/// * [`three_over_two_pi()`](crate::three_over_two_pi)
/// * [`two_over_pi()`](crate::two_over_pi)
/// * [`two_over_root_pi()`](crate::two_over_root_pi)
/// * [`two_pi()`](crate::two_pi)
pub fn pi<T: RealNumber>() -> T {
    T::pi()
}
