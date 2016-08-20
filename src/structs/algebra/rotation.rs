#![macro_use]

macro_rules! use_special_orthogonal_group_modules(
    () => {
        use algebra::structure::{EuclideanGroupApprox,  SpecialEuclideanGroupApprox,
                                 OrthogonalGroupApprox, SpecialOrthogonalGroupApprox,
                                 GroupApprox, LoopApprox, MonoidApprox,
                                 QuasigroupApprox, SemigroupApprox,
                                 RealApprox};
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
        use algebra::ident::Identity;
        use algebra::ops::{Recip, Multiplicative};
    }
);

macro_rules! special_orthogonal_group_impl(
    ($t: ident, $point: ident, $vector: ident) => {
        /*
         * Operations.
         */
        impl<N: BaseNum> Identity<Multiplicative> for $t<N> {
            #[inline]
            fn id() -> Self {
                ::one()
            }
        }

        impl<N: Copy + AlgebraApproxEq<Eps = N>> AlgebraApproxEq for $t<N> {
            type Eps = N;

            #[inline]
            fn default_epsilon() -> N {
                <N as AlgebraApproxEq>::default_epsilon()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                AlgebraApproxEq::approx_eq_eps(&self.submatrix, &other.submatrix, &epsilon)
            }
        }

        impl<N: Copy> Recip for $t<N> {
            type Result = $t<N>;

            #[inline]
            fn recip(mut self) -> $t<N> {
                self.inverse_mut();

                self
            }
        }


        /*
         *
         * Algebraic structures.
         *
         */
        // FIXME: in the end, we will keep only RealApprox.
        impl<N: BaseNum + RealApprox> GroupApprox<Multiplicative> for $t<N> { }
        impl<N: BaseNum + RealApprox> LoopApprox<Multiplicative> for $t<N> { }
        impl<N: BaseNum + RealApprox> MonoidApprox<Multiplicative> for $t<N> { }
        impl<N: BaseNum + RealApprox> QuasigroupApprox<Multiplicative> for $t<N> { }
        impl<N: BaseNum + RealApprox> SemigroupApprox<Multiplicative> for $t<N> { }

        /*
         *
         * Matrix groups.
         *
         */
        impl<N: BaseNum + RealApprox> EuclideanGroupApprox<N, $point<N>> for $t<N> {
            #[inline]
            fn transform_point(&self, pt: &$point<N>) -> $point<N> {
                *self * *pt
            }

            #[inline]
            fn transform_vector(&self, v: &$vector<N>) -> $vector<N> {
                *self * *v
            }
        }

        impl<N: BaseNum + RealApprox> SpecialEuclideanGroupApprox<N, $point<N>> for $t<N> {
        }

        impl<N: BaseNum + RealApprox> OrthogonalGroupApprox<N, $point<N>> for $t<N> {
        }

        impl<N: BaseNum + RealApprox> SpecialOrthogonalGroupApprox<N, $point<N>> for $t<N> {
        }
    }
);
