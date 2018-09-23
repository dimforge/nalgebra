use na::{self, Real};

/// The euler constant.
pub fn e<N: Real>() -> N {
    N::e()
}

/// The euler constant.
pub fn euler<N: Real>() -> N {
    N::e()
}

/// Returns `4 / pi`.
pub fn four_over_pi<N: Real>() -> N {
    na::convert::<_, N>(4.0) / N::pi()
}

/// Returns the golden ratio.
pub fn golden_ratio<N: Real>() -> N {
    (N::one() + root_five()) / na::convert(2.0)
}

/// Returns `pi / 2`.
pub fn half_pi<N: Real>() -> N {
    N::frac_pi_2()
}

/// Returns `ln(ln(2))`.
pub fn ln_ln_two<N: Real>() -> N {
    N::ln_2().ln()
}

/// Returns `ln(10)`.
pub fn ln_ten<N: Real>() -> N {
    N::ln_10()
}

/// Returns `ln(2)`.
pub fn ln_two<N: Real>() -> N {
    N::ln_2()
}

/// Returns `1`.
pub use na::one;

/// Returns `1 / pi`.
pub fn one_over_pi<N: Real>() -> N {
    N::frac_1_pi()
}

/// Returns `1 / sqrt(2)`.
pub fn one_over_root_two<N: Real>() -> N {
    N::one() / root_two()
}

/// Returns `1 / (2pi)`.
pub fn one_over_two_pi<N: Real>() -> N {
    N::frac_1_pi() * na::convert(0.5)
}

/// Returns `pi / 4`.
pub fn quarter_pi<N: Real>() -> N {
    N::frac_pi_4()
}

/// Returns `sqrt(5)`.
pub fn root_five<N: Real>() -> N {
    na::convert::<_, N>(5.0).sqrt()
}

/// Returns `sqrt(pi / 2)`.
pub fn root_half_pi<N: Real>() -> N {
    (N::pi() / na::convert(2.0)).sqrt()
}

/// Returns `sqrt(ln(4))`.
pub fn root_ln_four<N: Real>() -> N {
    na::convert::<_, N>(4.0).ln().sqrt()
}

/// Returns the square root of pi.
pub fn root_pi<N: Real>() -> N {
    N::pi().sqrt()
}

/// Returns the square root of 3.
pub fn root_three<N: Real>() -> N {
    na::convert::<_, N>(3.0).sqrt()
}

/// Returns the square root of 2.
pub fn root_two<N: Real>() -> N {
    // FIXME: there should be a ::sqrt_2() on the Real trait.
    na::convert::<_, N>(2.0).sqrt()
}

/// Returns the square root of 2pi.
pub fn root_two_pi<N: Real>() -> N {
    N::two_pi().sqrt()
}

/// Returns `1 / 3`.
pub fn third<N: Real>() -> N {
    na::convert(1.0 / 3.0)
}

/// Returns `3 / (2pi)`.
pub fn three_over_two_pi<N: Real>() -> N {
    na::convert::<_, N>(3.0) / N::two_pi()
}

/// Returns `2 / pi`.
pub fn two_over_pi<N: Real>() -> N {
    N::frac_2_pi()
}

/// Returns `2 / sqrt(pi)`.
pub fn two_over_root_pi<N: Real>() -> N {
    N::frac_2_sqrt_pi()
}

/// Returns `2pi`.
pub fn two_pi<N: Real>() -> N {
    N::two_pi()
}

/// Returns `2 / 3`.
pub fn two_thirds<N: Real>() -> N {
    na::convert(2.0 / 3.0)
}

/// Returns `0`.
pub use na::zero;