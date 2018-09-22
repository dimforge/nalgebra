use na;

use traits::Number;

/// Returns the maximum value among three.
pub fn max3<N: Number>(a: N, b: N, c: N) -> N {
    na::sup(&na::sup(&a, &b), &c)
}

/// Returns the maximum value among four.
pub fn max4<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::sup(&na::sup(&a, &b), &na::sup(&c, &d))
}

/// Returns the maximum value among three.
pub fn min3<N: Number>(a: N, b: N, c: N) -> N {
    na::inf(&na::inf(&a, &b), &c)
}

/// Returns the maximum value among four.
pub fn min4<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::inf(&na::inf(&a, &b), &na::inf(&c, &d))
}