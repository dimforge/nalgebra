#![macro_use]

macro_rules! point_impl(
    ($t: ident, $tv: ident | $($compN: ident),+) => (

        euclidean_space_impl!($t, $tv);

        /*
         *
         * Origin.
         *
         */
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


        /*
         *
         * Point - Point
         *
         */
        impl<N: Copy + Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $tv<N> {
                *self.as_vector() - *right.as_vector()
            }
        }


        /*
         *
         * Point + Vector
         *
         */
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


        /*
         *
         * Point - Vector
         *
         */
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



        /*
         *
         * Point as vector.
         *
         */
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
            pub fn as_vector(&self) -> &$tv<N> {
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
            fn as_vector(&self) -> &$tv<N> {
                self.as_vector()
            }

            #[inline]
            fn set_coords(&mut self, v: $tv<N>) {
                self.set_coords(v)
            }
        }



        /*
         *
         * NumPoint / FloatPoint
         *
         */
        impl<N> NumPoint<N> for $t<N>
            where N: BaseNum {
        }

        impl<N> FloatPoint<N> for $t<N>
            where N: BaseFloat + ApproxEq<N> {
        }


        /*
         *
         * Display
         *
         */
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
    );
    ($t: ident, $tv: ident, $th: ident, $comp_extra: ident | $($compN: ident),+) => (
        point_impl!($t, $tv | $($compN),+);

        /*
         *
         * ToHomogeneous / FromHomogeneous
         *
         */
        impl<N: Copy + One + Zero> ToHomogeneous<$th<N>> for $t<N> {
            fn to_homogeneous(&self) -> $th<N> {
                let mut res: $th<N> = Origin::origin();

                $( res.$compN = self.$compN; )+
                res.$comp_extra = ::one();

                res
            }
        }

        impl<N: Copy + Div<N, Output = N> + One + Zero> FromHomogeneous<$th<N>> for $t<N> {
            fn from(v: &$th<N>) -> $t<N> {
                let mut res: $t<N> = Origin::origin();

                $( res.$compN = v.$compN / v.$comp_extra; )+

                res
            }
        }
    )
);
