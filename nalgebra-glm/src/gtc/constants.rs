use na::{self, RealField};

/// The Euler constant.
///
/// This is a shorthand alias for [`euler`](fn.euler.html).
pub fn e<N: RealField>() -> N {
    N::e()
}

/// The Euler constant.
pub fn euler<N: RealField>() -> N {
    N::e()
}

/// Returns `4 / pi`.
///
/// # See also:
///
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn four_over_pi<N: RealField>() -> N {
    na::convert::<_, N>(4.0) / N::pi()
}

/// Returns the golden ratio.
pub fn golden_ratio<N: RealField>() -> N {
    (N::one() + root_five()) / na::convert(2.0)
}

/// Returns `pi / 2`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn half_pi<N: RealField>() -> N {
    N::frac_pi_2()
}

/// Returns `ln(ln(2))`.
///
/// # See also:
///
/// * [`ln_ten`](fn.ln_ten.html)
/// * [`ln_two`](fn.ln_two.html)
pub fn ln_ln_two<N: RealField>() -> N {
    N::ln_2().ln()
}

/// Returns `ln(10)`.
///
/// # See also:
///
/// * [`ln_ln_two`](fn.ln_ln_two.html)
/// * [`ln_two`](fn.ln_two.html)
pub fn ln_ten<N: RealField>() -> N {
    N::ln_10()
}

/// Returns `ln(2)`.
///
/// # See also:
///
/// * [`ln_ln_two`](fn.ln_ln_two.html)
/// * [`ln_ten`](fn.ln_ten.html)
pub fn ln_two<N: RealField>() -> N {
    N::ln_2()
}

/// Returns `1`.
pub use na::one;

/// Returns `1 / pi`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn one_over_pi<N: RealField>() -> N {
    N::frac_1_pi()
}

/// Returns `1 / sqrt(2)`.
pub fn one_over_root_two<N: RealField>() -> N {
    N::one() / root_two()
}

/// Returns `1 / (2pi)`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn one_over_two_pi<N: RealField>() -> N {
    N::frac_1_pi() * na::convert(0.5)
}

/// Returns `pi / 4`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn quarter_pi<N: RealField>() -> N {
    N::frac_pi_4()
}

/// Returns `sqrt(5)`.
///
/// # See also:
///
/// * [`root_three`](fn.root_three.html)
/// * [`root_two`](fn.root_two.html)
pub fn root_five<N: RealField>() -> N {
    na::convert::<_, N>(5.0).sqrt()
}

/// Returns `sqrt(pi / 2)`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn root_half_pi<N: RealField>() -> N {
    (N::pi() / na::convert(2.0)).sqrt()
}

/// Returns `sqrt(ln(4))`.
pub fn root_ln_four<N: RealField>() -> N {
    na::convert::<_, N>(4.0).ln().sqrt()
}

/// Returns the square root of pi.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn root_pi<N: RealField>() -> N {
    N::pi().sqrt()
}

/// Returns the square root of 3.
///
/// # See also:
///
/// * [`root_five`](fn.root_five.html)
/// * [`root_two`](fn.root_two.html)
pub fn root_three<N: RealField>() -> N {
    na::convert::<_, N>(3.0).sqrt()
}

/// Returns the square root of 2.
///
/// # See also:
///
/// * [`root_five`](fn.root_five.html)
/// * [`root_three`](fn.root_three.html)
pub fn root_two<N: RealField>() -> N {
    // TODO: there should be a crate::sqrt_2() on the RealField trait.
    na::convert::<_, N>(2.0).sqrt()
}

/// Returns the square root of 2pi.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn root_two_pi<N: RealField>() -> N {
    N::two_pi().sqrt()
}

/// Returns `1 / 3`.
///
/// # See also:
///
/// * [`two_thirds`](fn.two_thirds.html)
pub fn third<N: RealField>() -> N {
    na::convert(1.0 / 3.0)
}

/// Returns `3 / (2pi)`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn three_over_two_pi<N: RealField>() -> N {
    na::convert::<_, N>(3.0) / N::two_pi()
}

/// Returns `2 / pi`.
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
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn two_over_pi<N: RealField>() -> N {
    N::frac_2_pi()
}

/// Returns `2 / sqrt(pi)`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_pi`](fn.two_pi.html)
pub fn two_over_root_pi<N: RealField>() -> N {
    N::frac_2_sqrt_pi()
}

/// Returns `2pi`.
///
/// # See also:
///
/// * [`four_over_pi`](fn.four_over_pi.html)
/// * [`half_pi`](fn.half_pi.html)
/// * [`one_over_pi`](fn.one_over_pi.html)
/// * [`one_over_two_pi`](fn.one_over_two_pi.html)
/// * [`pi`](fn.pi.html)
/// * [`quarter_pi`](fn.quarter_pi.html)
/// * [`root_half_pi`](fn.root_half_pi.html)
/// * [`root_pi`](fn.root_pi.html)
/// * [`root_two_pi`](fn.root_two_pi.html)
/// * [`three_over_two_pi`](fn.three_over_two_pi.html)
/// * [`two_over_pi`](fn.two_over_pi.html)
/// * [`two_over_root_pi`](fn.two_over_root_pi.html)
pub fn two_pi<N: RealField>() -> N {
    N::two_pi()
}

/// Returns `2 / 3`.
///
/// # See also:
///
/// * [`third`](fn.third.html)
pub fn two_thirds<N: RealField>() -> N {
    na::convert(2.0 / 3.0)
}

/// Returns `0`.
pub use na::zero;
