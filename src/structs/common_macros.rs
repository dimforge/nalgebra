#![macro_use]

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
    )
);


macro_rules! pointwise_scalar_mul(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Mul<N, Output = N>> Mul<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: N) -> $t<N> {
                $t::new($(self.$compN * right),+)
            }
        }

        impl<N: Copy + MulAssign<N>> MulAssign<N> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: N) {
                $( self.$compN *= right; )+
            }
        }

        impl Mul<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn mul(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self * right.$compN),+)
            }
        }

        impl Mul<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn mul(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self * right.$compN),+)
            }
        }
    )
);


macro_rules! pointwise_scalar_div(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N>> Div<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn div(self, right: N) -> $t<N> {
                $t::new($(self.$compN / right),+)
            }
        }

        impl<N: Copy + DivAssign<N>> DivAssign<N> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: N) {
                $( self.$compN /= right; )+
            }
        }
    )
);


macro_rules! pointwise_scalar_add(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Add<N, Output = N>> Add<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: N) -> $t<N> {
                $t::new($(self.$compN + right),+)
            }
        }

        impl<N: Copy + AddAssign<N>> AddAssign<N> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: N) {
                $( self.$compN += right; )+
            }
        }

        impl Add<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn add(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self + right.$compN),+)
            }
        }

        impl Add<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn add(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self + right.$compN),+)
            }
        }
    )
);

macro_rules! pointwise_scalar_sub(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: N) -> $t<N> {
                $t::new($(self.$compN - right),+)
            }
        }

        impl<N: Copy + SubAssign<N>> SubAssign<N> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: N) {
                $( self.$compN -= right; )+
            }
        }

        impl Sub<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn sub(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self - right.$compN),+)
            }
        }

        impl Sub<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn sub(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self - right.$compN),+)
            }
        }
    )
);

macro_rules! componentwise_neg(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Neg<Output = N> + Copy> Neg for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn neg(self) -> $t<N> {
                $t::new($(-self.$compN ),+)
            }
        }
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
