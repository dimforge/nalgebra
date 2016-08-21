#![macro_use]

macro_rules! use_euclidean_space_modules(
    () => {
        use algebra::structure::{AffineSpaceApprox, EuclideanSpaceApprox,
                                 FieldApprox, RealApprox};
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
    }
);


macro_rules! euclidean_space_impl(
    ($t: ident, $vector: ident) => {
        impl<N> AffineSpaceApprox<N> for $t<N>
            where N: Copy + Neg<Output = N> + Add<N, Output = N> +
                     Sub<N, Output = N> + AlgebraApproxEq + FieldApprox {
            type Translation = $vector<N>;
            
            #[inline]
            fn translate_by(&self, vector: &Self::Translation) -> Self {
                *self + *vector
            }

            #[inline]
            fn subtract(&self, other: &Self) -> Self::Translation {
                *self - *other
            }
        }

        impl<N: RealApprox> EuclideanSpaceApprox<N> for $t<N> {
            type Vector = $vector<N>;
        }
    }
);
