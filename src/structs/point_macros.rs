#![macro_use]

macro_rules! origin_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Zero> Origin for $t<N> {
            #[inline]
            fn origin() -> $t<N> {
                $t {
                    $($compN: ::zero() ),+
                }
            }

            #[inline]
            fn is_origin(&self) -> bool {
                $(self.$compN.is_zero() )&&+
            }
        }
    )
);

macro_rules! point_sub_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $tv<N> {
                *self.as_vector() - *right.as_vector()
            }
        }
    )
);

macro_rules! point_add_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N: Copy + Add<N, Output = N>> Add<$tv<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: $tv<N>) -> $t<N> {
                $t::new($(self.$compN + right.$compN),+)
            }
        }

        impl<N: Copy + AddAssign<N>> AddAssign<$tv<N>> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: $tv<N>) {
                $( self.$compN += right.$compN; )+
            }
        }
    )
);

macro_rules! point_sub_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<$tv<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: $tv<N>) -> $t<N> {
                $t::new($(self.$compN - right.$compN),+)
            }
        }

        impl<N: Copy + SubAssign<N>> SubAssign<$tv<N>> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: $tv<N>) {
                $( self.$compN -= right.$compN; )+
            }
        }
    )
);

macro_rules! point_as_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N> $t<N> {
            /// Converts this point to its associated vector.
            #[inline]
            pub fn to_vector(self) -> $tv<N> {
                $tv::new(
                    $(self.$compN),+
                )
            }

            /// Converts a reference to this point to a reference to its associated vector.
            #[inline]
            pub fn as_vector<'a>(&'a self) -> &'a $tv<N> {
                unsafe {
                    mem::transmute(self)
                }
            }

            #[inline]
            fn set_coords(&mut self, v: $tv<N>) {
                $(self.$compN = v.$compN;)+
            }
        }

        impl<N> PointAsVector for $t<N> {
            type Vector = $tv<N>;
            
            #[inline]
            fn to_vector(self) -> $tv<N> {
                self.to_vector()
            }

            #[inline]
            fn as_vector<'a>(&'a self) -> &'a $tv<N> {
                self.as_vector()
            }

            #[inline]
            fn set_coords(&mut self, v: $tv<N>) {
                self.set_coords(v)
            }
        }
    )
);

macro_rules! point_to_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + One + Zero> ToHomogeneous<$t2<N>> for $t<N> {
            fn to_homogeneous(&self) -> $t2<N> {
                let mut res: $t2<N> = Origin::origin();

                $( res.$compN = self.$compN; )+
                res.$extra    = ::one();

                res
            }
        }
    )
);

macro_rules! point_from_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N> + One + Zero> FromHomogeneous<$t2<N>> for $t<N> {
            fn from(v: &$t2<N>) -> $t<N> {
                let mut res: $t<N> = Origin::origin();

                $( res.$compN = v.$compN / v.$extra; )+

                res
            }
        }
    )
);

macro_rules! num_float_point_impl(
    ($t: ident, $tv: ident) => (
        impl<N> NumPoint<N> for $t<N>
            where N: BaseNum {
        }

        impl<N> FloatPoint<N> for $t<N>
            where N: BaseFloat + ApproxEq<N> {
        }
    )
);

macro_rules! arbitrary_point_impl(
    ($t: ident, $($compN: ident),*) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary> Arbitrary for $t<N> {
            #[inline]
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t {
                    $($compN: Arbitrary::arbitrary(g),)*
                }
            }
        }
    )
);

macro_rules! point_display_impl(
    ($t: ident) => (
        impl<N: fmt::Display> fmt::Display for $t<N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                // FIXME: differenciate them from vectors ?
                try!(write!(f, "("));

                let mut it = self.iter();

                try!(write!(f, "{}", *it.next().unwrap()));

                for comp in it {
                    try!(write!(f, ", {}", *comp));
                }

                write!(f, ")")
            }
        }
    )
);
