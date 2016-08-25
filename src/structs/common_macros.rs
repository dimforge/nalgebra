#![macro_use]

macro_rules! ref_binop(
    // Symmetric to `std::ops::forward_ref_binop!`
    (impl $imp:ident, $method:ident for $t:ident; $($compN:ident),+) => {
        impl<'a, 'b, N> $imp<&'a $t<N>> for &'b $t<N> where &'b N: $imp<&'a N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: &'a $t<N>) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(&self.$compN, &right.$compN)
                    ),+
                )
            }
        }

        impl<'a, N> $imp<&'a $t<N>> for $t<N> where N: $imp<&'a N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: &'a $t<N>) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(self.$compN, &right.$compN)
                    ),+
                )
            }
        }

        impl<'a, N> $imp<$t<N>> for &'a $t<N> where &'a N: $imp<N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: $t<N>) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(&self.$compN, right.$compN)
                    ),+
                )
            }
        }
    }
);

macro_rules! pointwise_mul(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Mul<N, Output = N>> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN * right.$compN),+)
            }
        }

        impl<N: MulAssign<N>> MulAssign<$t<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                $( self.$compN *= right.$compN; )+
            }
        }

        ref_binop!(impl Mul, mul for $t; $($compN),+);
    )
);

macro_rules! pointwise_div(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Div<N, Output = N>> Div<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn div(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN / right.$compN),+)
            }
        }

        impl<N: DivAssign<N>> DivAssign<$t<N>> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: $t<N>) {
                $( self.$compN /= right.$compN; )+
            }
        }

        ref_binop!(impl Div, div for $t; $($compN),+);
    )
);

macro_rules! pointwise_add(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Add<N, Output = N>> Add<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN + right.$compN),+)
            }
        }

        impl<N: AddAssign<N>> AddAssign<$t<N>> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: $t<N>) {
                $( self.$compN += right.$compN; )+
            }
        }

        ref_binop!(impl Add, add for $t; $($compN),+);
    )
);


macro_rules! pointwise_sub(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN - right.$compN),+)
            }
        }


        impl<N: SubAssign<N>> SubAssign<$t<N>> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: $t<N>) {
                $( self.$compN -= right.$compN; )+
            }
        }

        ref_binop!(impl Sub, sub for $t; $($compN),+);
    )
);


macro_rules! ref_binop_scalar_exact(
    (impl $imp:ident<$lhs:ident>, $method:ident for $t:ident; $($compN:ident),+) => {
        impl $imp<$t<$lhs>> for $lhs {
            type Output = $t<$lhs>;

            #[inline]
            fn $method(self, right: $t<$lhs>) -> $t<$lhs> {
                $t::new(
                    $(
                        $imp::$method(right.$compN, self)
                    ),+
                )
            }
        }

        impl<'a, 'b> $imp<&'a $t<$lhs>> for &'b $lhs {
            type Output = $t<$lhs>;

            #[inline]
            fn $method(self, right: &'a $t<$lhs>) -> $t<$lhs> {
                $t::new(
                    $(
                        $imp::$method(right.$compN, self)
                    ),+
                )
            }
        }

        impl<'a> $imp<$t<$lhs>> for &'a $lhs {
            type Output = $t<$lhs>;

            #[inline]
            fn $method(self, right: $t<$lhs>) -> $t<$lhs> {
                $t::new(
                    $(
                        $imp::$method(right.$compN, self)
                    ),+
                )
            }
        }

        impl<'a> $imp<&'a $t<$lhs>> for $lhs {
            type Output = $t<$lhs>;

            #[inline]
            fn $method(self, right: &'a $t<$lhs>) -> $t<$lhs> {
                $t::new(
                    $(
                        $imp::$method(right.$compN, self)
                    ),+
                )
            }
        }
    };
);


macro_rules! ref_binop_scalar(
    // Symmetric to `std::ops::forward_ref_binop!`
    (impl $imp:ident, $method:ident for $t:ident; $($compN:ident),+) => {
        ref_binop_scalar_exact!(impl $imp<f32>, $method for $t; $($compN),+);
        ref_binop_scalar_exact!(impl $imp<f64>, $method for $t; $($compN),+);

        impl<N: Copy + $imp<N, Output = N>> $imp<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: N) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(self.$compN, right)
                    ),+
                )
            }
        }

        impl<'a, 'b, N> $imp<&'a N> for &'b $t<N> where &'b N: $imp<&'a N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: &'a N) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(&self.$compN, &right)
                    ),+
                )
            }
        }

        impl<'a, N> $imp<N> for &'a $t<N> where for<'b> &'a N: $imp<&'b N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: N) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(&self.$compN, &right)
                    ),+
                )
            }
        }

        impl<'a, N> $imp<&'a N> for $t<N> where N: $imp<&'a N, Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self, right: &'a N) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(self.$compN, &right)
                    ),+
                )
            }
        }
    }
);


macro_rules! pointwise_scalar_mul(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + MulAssign<N>> MulAssign<N> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: N) {
                $( self.$compN *= right; )+
            }
        }

        ref_binop_scalar!(impl Mul, mul for $t; $($compN),+);
    )
);


macro_rules! pointwise_scalar_div(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + DivAssign<N>> DivAssign<N> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: N) {
                $( self.$compN /= right; )+
            }
        }

        ref_binop_scalar!(impl Div, div for $t; $($compN),+);
    )
);


macro_rules! pointwise_scalar_add(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + AddAssign<N>> AddAssign<N> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: N) {
                $( self.$compN += right; )+
            }
        }

        ref_binop_scalar!(impl Add, add for $t; $($compN),+);
    )
);

macro_rules! pointwise_scalar_sub(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + SubAssign<N>> SubAssign<N> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: N) {
                $( self.$compN -= right; )+
            }
        }

        ref_binop_scalar!(impl Sub, sub for $t; $($compN),+);
    )
);


macro_rules! ref_unop(
    // Symmetric to `std::ops::forward_ref_unop!`
    (impl $imp:ident, $method:ident for $t:ident; $($compN:ident),+) => {
        impl<N: $imp<Output = N> + Copy> $imp for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(self.$compN)
                    ),+
                )
            }
        }

        impl<'a, N> $imp for &'a $t<N> where &'a N: $imp<Output = N> {
            type Output = $t<N>;

            #[inline]
            fn $method(self) -> $t<N> {
                $t::new(
                    $(
                        $imp::$method(&self.$compN)
                    ),+
                )
            }
        }
    }
);


macro_rules! componentwise_neg(
    ($t: ident, $($compN: ident),+) => (
        ref_unop!(impl Neg, neg for $t; $($compN),+);
    )
);

macro_rules! componentwise_repeat(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy> Repeat<N> for $t<N> {
            fn repeat(val: N) -> $t<N> {
                $t {
                    $($compN: val ),+
                }
            }
        }
    )
);

macro_rules! componentwise_absolute(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Absolute<N>> Absolute<$t<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $t<N> {
                $t::new($(::abs(&m.$compN) ),+)
            }
        }
    )
);

macro_rules! componentwise_zero(
    ($t: ident, $($compN: ident),+ ) => (
        impl<N: Zero> Zero for $t<N> {
            #[inline]
            fn zero() -> $t<N> {
                $t {
                    $($compN: ::zero() ),+
                }
            }

            #[inline]
            fn is_zero(&self) -> bool {
                $(::is_zero(&self.$compN) )&&+
            }
        }
    )
);

macro_rules! componentwise_one(
    ($t: ident, $($compN: ident),+ ) => (
        impl<N: BaseNum> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t {
                    $($compN: ::one() ),+
                }
            }
        }
    )
);

// Implements Arbitrary by setting each components to Arbitrary::arbitrary.
macro_rules! componentwise_arbitrary(
    ($t: ident, $($compN: ident),+ ) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary> Arbitrary for $t<N> {
            #[inline]
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t { $($compN: Arbitrary::arbitrary(g),)* }
            }
        }
    )
);

// Implements Rand by setting each components to Rand::rand.
macro_rules! componentwise_rand(
    ($t: ident, $($compN: ident),+ ) => (
        impl<N: Rand> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                $t { $($compN: Rand::rand(rng), )* }
            }
        }
    )
);

macro_rules! component_basis_element(
    ($t: ident, $($compN: ident),+ ) => (
        /*
         *
         * Element of the canonical basis.
         *
         */
        impl<N: Zero + One> $t<N> {
            $(
                /// Create the element of the canonical basis having this component set to one and
                /// all the others set to zero.
                #[inline]
                pub fn $compN() -> $t<N> {
                    let mut res: $t<N> = ::zero();
        
                    res.$compN = ::one();
        
                    res
                }
             )+
        }
    )
);

// A function to create a new element from its component values.
macro_rules! component_new(
    ($t: ident, $($compN: ident),+) => (
        impl<N> $t<N> {
            /// Creation from component values.
            #[inline]
            pub fn new($($compN: N ),+) -> $t<N> {
                $t {
                    $($compN: $compN ),+
                }
            }
        }
    );
);


macro_rules! fold_add(
    // base case
    ($x:expr) => {
        $x
    };
    // `$x` followed by at least one `$y,`
    ($x:expr, $($y:expr),+) => {
        // call min! on the tail `$y`
        Add::add($x, fold_add!($($y),+))
    }
);
