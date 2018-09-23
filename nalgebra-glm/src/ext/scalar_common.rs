use na;

use traits::Number;

/// Returns the maximum among three values.
pub fn max3_scalar<N: Number>(a: N, b: N, c: N) -> N {
    na::sup(&na::sup(&a, &b), &c)
}

/// Returns the maximum among four values.
pub fn max4_scalar<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::sup(&na::sup(&a, &b), &na::sup(&c, &d))
}

/// Returns the minimum among three values.
pub fn min3_scalar<N: Number>(a: N, b: N, c: N) -> N {
    na::inf(&na::inf(&a, &b), &c)
}

/// Returns the minimum among four values.
pub fn min4_scalar<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::inf(&na::inf(&a, &b), &na::inf(&c, &d))
}