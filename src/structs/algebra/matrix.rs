#![macro_use]

macro_rules! use_matrix_group_modules(
    () => {
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
    }
);

macro_rules! matrix_group_approx_impl(
    ($t: ident, $($compN: ident),+) => {
        impl<N: AlgebraApproxEq> AlgebraApproxEq for $t<N> {
            type Eps = N::Eps;

            #[inline]
            fn default_epsilon() -> N::Eps {
                N::default_epsilon()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N::Eps) -> bool {
                $(AlgebraApproxEq::approx_eq_eps(&self.$compN, &other.$compN, &epsilon))&&+
            }
        }
    }
);
