#![macro_use]

macro_rules! use_vector_space_modules(
    () => {
        use algebra::structure::{FieldApprox, RingCommutativeApprox, GroupAbelianApprox,
                                 GroupApprox, LoopApprox, MonoidApprox, QuasigroupApprox,
                                 SemigroupApprox, VectorSpaceApprox, ModuleApprox,
                                 NormedSpaceApprox, InnerSpaceApprox,
                                 FiniteDimVectorSpaceApprox,
                                 Field, RingCommutative, GroupAbelian,
                                 Group, Loop, Monoid, Quasigroup,
                                 Semigroup, VectorSpace, Module, RealApprox};
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
        use algebra::ident::Identity;
        use algebra::ops::Additive;
    }
);

macro_rules! vector_space_impl(
    ($t: ident, $dimension: expr, $($compN: ident),+) => {
        /*
         * Identity & ApproxEq
         */
        impl<N: Copy + Identity<Additive>> Identity<Additive> for $t<N> {
            #[inline]
            fn id() -> Self {
                Repeat::repeat(Identity::id())
            }
        }

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

        /*
         *
         * Approximate algebraic structures.
         *
         */
        product_space_inherit_structure!($t, GroupAbelianApprox<Additive>);
        product_space_inherit_structure!($t, GroupApprox<Additive>);
        product_space_inherit_structure!($t, LoopApprox<Additive>);
        product_space_inherit_structure!($t, MonoidApprox<Additive>);
        product_space_inherit_structure!($t, QuasigroupApprox<Additive>);
        product_space_inherit_structure!($t, SemigroupApprox<Additive>);

        /*
         * Module.
         */
        impl<N> ModuleApprox<N> for $t<N> where N: Copy + Neg<Output = N> + Add<N, Output = N> +
                     AlgebraApproxEq + RingCommutativeApprox
            { }

        /*
         * Vector spaces.
         */
        impl<N> VectorSpaceApprox<N> for $t<N>
            where N: Copy + Neg<Output = N> + Add<N, Output = N> +
                     AlgebraApproxEq + FieldApprox { }

        impl<N> FiniteDimVectorSpaceApprox<N> for $t<N>
            where N: Copy + Zero + One + Neg<Output = N> + Add<N, Output = N> +
                     AlgebraApproxEq + FieldApprox {
            #[inline]
            fn dimension() -> usize {
                $dimension
            }

            #[inline]
            fn canonical_basis<F: FnOnce(&[$t<N>])>(f: F) {
                let basis = [
                    $($t::$compN()),*
                ];

                f(&basis[..])
            }

            #[inline]
            fn component(&self, i: usize) -> N {
                self[i]
            }

            #[inline]
            unsafe fn component_unchecked(&self, i: usize)  -> N {
                self.at_fast(i)
            }
        }

        impl<N: RealApprox> NormedSpaceApprox<N> for $t<N> {
            #[inline]
            fn norm_squared(&self) -> N {
                self.inner_product(self)
            }

            #[inline]
            fn norm(&self) -> N {
                self.norm_squared().sqrt()
            }

            #[inline]
            fn normalize(&self) -> Self {
                *self / self.norm()
            }

            #[inline]
            fn normalize_mut(&mut self) -> N {
                let n = self.norm();
                *self /= n;

                n
            }

            #[inline]
            fn try_normalize(&self, min_norm: &N) -> Option<Self> {
                let n = self.norm();

                if n <= *min_norm {
                    None
                }
                else {
                    Some(*self / n)
                }
            }

            #[inline]
            fn try_normalize_mut(&mut self, min_norm: &N) -> Option<N> {
                let n = self.norm();

                if n <= *min_norm {
                    None
                }
                else {
                    *self /= n;
                    Some(n)
                }
            }
        }

        impl<N: RealApprox> InnerSpaceApprox<N> for $t<N> {
            #[inline]
            fn inner_product(&self, other: &Self) -> N {
                fold_add!($(self.$compN * other.$compN ),+)
            }
        }

        /*
         *
         * Exact algebraic structures.
         *
         */

        product_space_inherit_structure!($t, GroupAbelian<Additive>);
        product_space_inherit_structure!($t, Group<Additive>);
        product_space_inherit_structure!($t, Loop<Additive>);
        product_space_inherit_structure!($t, Monoid<Additive>);
        product_space_inherit_structure!($t, Quasigroup<Additive>);
        product_space_inherit_structure!($t, Semigroup<Additive>);

        impl<N> VectorSpace<N> for $t<N>
            where N: Copy + Neg<Output = N> + Add<N, Output = N> + AlgebraApproxEq + Field
            { }

        impl<N> Module<N> for $t<N>
            where N: Copy + Neg<Output = N> + Add<N, Output = N> + AlgebraApproxEq + RingCommutative
            { }
    }
);

macro_rules! product_space_inherit_structure(
    ($t: ident, $marker: ident<$operator: ident>) => {
        impl<N> $marker<$operator> for $t<N>
            where N: Copy + Neg<Output = N> + Add<N, Output = N> + AlgebraApproxEq +
                     $marker<$operator>
                 { }
    }
);
