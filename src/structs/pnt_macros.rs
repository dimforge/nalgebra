#![macro_use]

macro_rules! orig_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Zero> Orig for $t<N> {
            #[inline]
            fn orig() -> $t<N> {
                $t {
                    $($compN: ::zero() ),+
                }
            }

            #[inline]
            fn is_orig(&self) -> bool {
                $(self.$compN.is_zero() )&&+
            }
        }
    )
);

macro_rules! pnt_sub_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $tv<N> {
                *self.as_vec() - *right.as_vec()
            }
        }
    )
);

macro_rules! pnt_add_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N: Copy + Add<N, Output = N>> Add<$tv<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: $tv<N>) -> $t<N> {
                $t::new($(self.$compN + right.$compN),+)
            }
        }
    )
);

macro_rules! pnt_sub_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<$tv<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: $tv<N>) -> $t<N> {
                $t::new($(self.$compN - right.$compN),+)
            }
        }
    )
);

macro_rules! pnt_as_vec_impl(
    ($t: ident, $tv: ident, $($compN: ident),+) => (
        impl<N> $t<N> {
            /// Converts this point to its associated vector.
            #[inline]
            pub fn to_vec(self) -> $tv<N> {
                $tv::new(
                    $(self.$compN),+
                )
            }

            /// Converts a reference to this point to a reference to its associated vector.
            #[inline]
            pub fn as_vec<'a>(&'a self) -> &'a $tv<N> {
                unsafe {
                    mem::transmute(self)
                }
            }

            #[inline]
            fn set_coords(&mut self, v: $tv<N>) {
                $(self.$compN = v.$compN;)+
            }
        }

        impl<N> PntAsVec<$tv<N>> for $t<N> {
            #[inline]
            fn to_vec(self) -> $tv<N> {
                self.to_vec()
            }

            #[inline]
            fn as_vec<'a>(&'a self) -> &'a $tv<N> {
                self.as_vec()
            }

            #[inline]
            fn set_coords(&mut self, v: $tv<N>) {
                self.set_coords(v)
            }
        }
    )
);

macro_rules! pnt_to_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + One + Zero> ToHomogeneous<$t2<N>> for $t<N> {
            fn to_homogeneous(&self) -> $t2<N> {
                let mut res: $t2<N> = Orig::orig();

                $( res.$compN = self.$compN; )+
                res.$extra    = ::one();

                res
            }
        }
    )
);

macro_rules! pnt_from_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N> + One + Zero> FromHomogeneous<$t2<N>> for $t<N> {
            fn from(v: &$t2<N>) -> $t<N> {
                let mut res: $t<N> = Orig::orig();

                $( res.$compN = v.$compN / v.$extra; )+

                res
            }
        }
    )
);

macro_rules! num_float_pnt_impl(
    ($t: ident, $tv: ident) => (
        impl<N> NumPnt<N, $tv<N>> for $t<N>
            where N: BaseNum {
        }

        impl<N> FloatPnt<N, $tv<N>> for $t<N>
            where N: BaseFloat + ApproxEq<N> {
        }
    )
);
