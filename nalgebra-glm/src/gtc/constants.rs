use crate::RealNumber;

/// The Euler constant.
///
/// This is a shorthand alias for [`euler()`].
pub fn e<T: RealNumber>() -> T {
    T::e()
}

/// The Euler constant.
pub fn euler<T: RealNumber>() -> T {
    T::e()
}

/// Returns `4 / pi`.
///
/// # See also:
///
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn four_over_pi<T: RealNumber>() -> T {
    na::convert::<_, T>(4.0) / T::pi()
}

/// Returns the golden ratio.
pub fn golden_ratio<T: RealNumber>() -> T {
    (T::one() + root_five()) / na::convert(2.0)
}

/// Returns `pi / 2`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn half_pi<T: RealNumber>() -> T {
    T::frac_pi_2()
}

/// Returns `ln(ln(2))`.
///
/// # See also:
///
/// * [`ln_ten()`]
/// * [`ln_two()`]
pub fn ln_ln_two<T: RealNumber>() -> T {
    T::ln_2().ln()
}

/// Returns `ln(10)`.
///
/// # See also:
///
/// * [`ln_ln_two()`]
/// * [`ln_two()`]
pub fn ln_ten<T: RealNumber>() -> T {
    T::ln_10()
}

/// Returns `ln(2)`.
///
/// # See also:
///
/// * [`ln_ln_two()`]
/// * [`ln_ten()`]
pub fn ln_two<T: RealNumber>() -> T {
    T::ln_2()
}

/// Returns `1`.
pub use na::one;

/// Returns `1 / pi`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn one_over_pi<T: RealNumber>() -> T {
    T::frac_1_pi()
}

/// Returns `1 / sqrt(2)`.
pub fn one_over_root_two<T: RealNumber>() -> T {
    T::one() / root_two()
}

/// Returns `1 / (2pi)`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn one_over_two_pi<T: RealNumber>() -> T {
    T::frac_1_pi() * na::convert(0.5)
}

/// Returns `pi / 4`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn quarter_pi<T: RealNumber>() -> T {
    T::frac_pi_4()
}

/// Returns `sqrt(5)`.
///
/// # See also:
///
/// * [`root_three()`]
/// * [`root_two()`]
pub fn root_five<T: RealNumber>() -> T {
    na::convert::<_, T>(5.0).sqrt()
}

/// Returns `sqrt(pi / 2)`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn root_half_pi<T: RealNumber>() -> T {
    (T::pi() / na::convert(2.0)).sqrt()
}

/// Returns `sqrt(ln(4))`.
pub fn root_ln_four<T: RealNumber>() -> T {
    na::convert::<_, T>(4.0).ln().sqrt()
}

/// Returns the square root of pi.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn root_pi<T: RealNumber>() -> T {
    T::pi().sqrt()
}

/// Returns the square root of 3.
///
/// # See also:
///
/// * [`root_five()`]
/// * [`root_two()`]
pub fn root_three<T: RealNumber>() -> T {
    na::convert::<_, T>(3.0).sqrt()
}

/// Returns the square root of 2.
///
/// # See also:
///
/// * [`root_five()`]
/// * [`root_three()`]
pub fn root_two<T: RealNumber>() -> T {
    // TODO: there should be a crate::sqrt_2() on the RealNumber trait.
    na::convert::<_, T>(2.0).sqrt()
}

/// Returns the square root of 2pi.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn root_two_pi<T: RealNumber>() -> T {
    T::two_pi().sqrt()
}

/// Returns `1 / 3`.
///
/// # See also:
///
/// * [`two_thirds()`]
pub fn third<T: RealNumber>() -> T {
    na::convert(1.0 / 3.0)
}

/// Returns `3 / (2pi)`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn three_over_two_pi<T: RealNumber>() -> T {
    na::convert::<_, T>(3.0) / T::two_pi()
}

/// Returns `2 / pi`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_root_pi()`]
/// * [`two_pi()`]
pub fn two_over_pi<T: RealNumber>() -> T {
    T::frac_2_pi()
}

/// Returns `2 / sqrt(pi)`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_pi()`]
pub fn two_over_root_pi<T: RealNumber>() -> T {
    T::frac_2_sqrt_pi()
}

/// Returns `2pi`.
///
/// # See also:
///
/// * [`four_over_pi()`]
/// * [`half_pi()`]
/// * [`one_over_pi()`]
/// * [`one_over_two_pi()`]
/// * [`pi()`](crate::pi)
/// * [`quarter_pi()`]
/// * [`root_half_pi()`]
/// * [`root_pi()`]
/// * [`root_two_pi()`]
/// * [`three_over_two_pi()`]
/// * [`two_over_pi()`]
/// * [`two_over_root_pi()`]
pub fn two_pi<T: RealNumber>() -> T {
    T::two_pi()
}

/// Returns `2 / 3`.
///
/// # See also:
///
/// * [`third()`]
pub fn two_thirds<T: RealNumber>() -> T {
    na::convert(2.0 / 3.0)
}

/// Returns `0`.
pub use na::zero;
