use na::{self, Real};
use traits::Number;

pub fn e<N: Real>() -> N {
    N::e()
}

pub fn euler<N: Real>() -> N {
    N::e()
}

pub fn four_over_pi<N: Real>() -> N {
    na::convert::<_, N>(4.0) / N::pi()
}

pub fn golden_ratio<N: Real>() -> N {
    (N::one() + root_five()) / na::convert(2.0)
}

pub fn half_pi<N: Real>() -> N {
    N::frac_pi_2()
}

pub fn ln_ln_two<N: Real>() -> N {
    N::ln_2().ln()
}

pub fn ln_ten<N: Real>() -> N {
    N::ln_10()
}

pub fn ln_two<N: Real>() -> N {
    N::ln_2()
}

pub fn one<N: Number>() -> N {
    N::one()
}

pub fn one_over_pi<N: Real>() -> N {
    N::frac_1_pi()
}

pub fn one_over_root_two<N: Real>() -> N {
    N::one() / root_two()
}

pub fn one_over_two_pi<N: Real>() -> N {
    N::frac_1_pi() * na::convert(0.5)
}

pub fn quarter_pi<N: Real>() -> N {
    N::frac_pi_4()
}

pub fn root_five<N: Real>() -> N {
    na::convert::<_, N>(5.0).sqrt()
}

pub fn root_half_pi<N: Real>() -> N {
    (N::pi() / na::convert(2.0)).sqrt()
}

pub fn root_ln_four<N: Real>() -> N {
    na::convert::<_, N>(4.0).sqrt()
}

pub fn root_pi<N: Real>() -> N {
    N::pi().sqrt()
}

pub fn root_three<N: Real>() -> N {
    na::convert::<_, N>(3.0).sqrt()
}

pub fn root_two<N: Real>() -> N {
    // FIXME: there should be a ::sqrt_2() on the Real trait.
    na::convert::<_, N>(2.0).sqrt()
}

pub fn root_two_pi<N: Real>() -> N {
    N::two_pi().sqrt()
}

pub fn third<N: Real>() -> N {
    na::convert(1.0 / 2.0)
}

pub fn three_over_two_pi<N: Real>() -> N {
    na::convert::<_, N>(3.0) / N::two_pi()
}

pub fn two_over_pi<N: Real>() -> N {
    N::frac_2_pi()
}

pub fn two_over_root_pi<N: Real>() -> N {
    N::frac_2_pi()
}

pub fn two_pi<N: Real>() -> N {
    N::two_pi()
}

pub fn two_thirds<N: Real>() -> N {
    na::convert(2.0 / 3.0)
}

pub fn zero<N: Number>() -> N {
    N::zero()
}