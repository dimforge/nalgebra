use na;

use traits::Number;

pub fn max3<N: Number>(a: N, b: N, c: N) -> N {
    na::sup(&na::sup(&a, &b), &c)
}

pub fn max4<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::sup(&na::sup(&a, &b), &na::sup(&c, &d))
}

pub fn min3<N: Number>(a: N, b: N, c: N) -> N {
    na::inf(&na::inf(&a, &b), &c)
}

pub fn min4<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::inf(&na::inf(&a, &b), &na::inf(&c, &d))
}