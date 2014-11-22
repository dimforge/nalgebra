#![macro_escape]

macro_rules! submat_impl(
    ($t: ident, $submat: ident) => (
        impl<N> $t<N> {
            #[inline]
            pub fn submat<'r>(&'r self) -> &'r $submat<N> {
                &self.submat
            }
        }
    )
)

macro_rules! rotate_impl(
    ($trhs: ident, $t: ident, $tv: ident, $tp: ident) => (
        /*
         * FIXME: we use the double dispatch trick here so that we can rotate vectors _and_
         * points. Remove this as soon as rust supports multidispatch.
         */
        pub trait $trhs<N> {
            fn rotate(left: &$t<N>, right: &Self) -> Self;
            fn inv_rotate(left: &$t<N>, right: &Self) -> Self;
        }

        impl<N, V: $trhs<N>> Rotate<V> for $t<N> {
            #[inline(always)]
            fn rotate(&self, other: &V) -> V {
                $trhs::rotate(self, other)
            }

            #[inline(always)]
            fn inv_rotate(&self, other: &V) -> V {
                $trhs::inv_rotate(self, other)
            }
        }

        impl<N: BaseNum + Clone> $trhs<N> for $tv<N> {
            #[inline]
            fn rotate(t: &$t<N>, v: &$tv<N>) -> $tv<N> {
                *t * *v
            }

            #[inline]
            fn inv_rotate(t: &$t<N>, v: &$tv<N>) -> $tv<N> {
                *v * *t
            }
        }

        impl<N: BaseNum + Clone> $trhs<N> for $tp<N> {
            #[inline]
            fn rotate(t: &$t<N>, p: &$tp<N>) -> $tp<N> {
                *t * *p
            }

            #[inline]
            fn inv_rotate(t: &$t<N>, p: &$tp<N>) -> $tp<N> {
                *p * *t
            }
        }
    )
)

macro_rules! transform_impl(
    ($trhs: ident, $t: ident, $tv: ident, $tp: ident) => (
        /*
         * FIXME: we use the double dispatch trick here so that we can transform vectors _and_
         * points. Remove this as soon as rust supports multidispatch.
         */
        pub trait $trhs<N> {
            fn transform(left: &$t<N>, right: &Self) -> Self;
            fn inv_transform(left: &$t<N>, right: &Self) -> Self;
        }

        impl<N, V: $trhs<N>> Transform<V> for $t<N> {
            #[inline(always)]
            fn transform(&self, other: &V) -> V {
                $trhs::transform(self, other)
            }

            #[inline(always)]
            fn inv_transform(&self, other: &V) -> V {
                $trhs::inv_transform(self, other)
            }
        }

        impl<N: BaseNum + Clone> $trhs<N> for $tv<N> {
            #[inline]
            fn transform(t: &$t<N>, v: &$tv<N>) -> $tv<N> {
                t.rotate(v)
            }

            #[inline]
            fn inv_transform(t: &$t<N>, v: &$tv<N>) -> $tv<N> {
                t.inv_rotate(v)
            }
        }

        impl<N: BaseNum + Clone> $trhs<N> for $tp<N> {
            #[inline]
            fn transform(t: &$t<N>, p: &$tp<N>) -> $tp<N> {
                t.rotate(p)
            }

            #[inline]
            fn inv_transform(t: &$t<N>, p: &$tp<N>) -> $tp<N> {
                t.inv_rotate(p)
            }
        }
    )
)

macro_rules! dim_impl(
    ($t: ident, $dim: expr) => (
        impl<N> Dim for $t<N> {
            #[inline]
            fn dim(_: Option<$t<N>>) -> uint {
                $dim
            }
        }
    )
)

macro_rules! rotation_matrix_impl(
    ($t: ident, $tlv: ident, $tav: ident) => (
        impl<N: Zero + BaseNum + Cast<f64> + BaseFloat> RotationMatrix<N, $tlv<N>, $tav<N>, $t<N>> for $t<N> {
            #[inline]
            fn to_rot_mat(&self) -> $t<N> {
                self.clone()
            }
        }
    )
)

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: BaseNum + Clone> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t { submat: ::one() }
            }
        }
    )
)

macro_rules! rot_mul_rot_impl(
    ($t: ident, $mulrhs: ident) => (
        impl<N: BaseNum + Clone> $mulrhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t { submat: left.submat * right.submat }
            }
        }
    )
)

macro_rules! rot_mul_vec_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        impl<N: BaseNum + Clone> $mulrhs<N, $tv<N>> for $tv<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$tv<N>) -> $tv<N> {
                left.submat * *right
            }
        }
    )
)

macro_rules! rot_mul_pnt_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        rot_mul_vec_impl!($t, $tv, $mulrhs)
    )
)

macro_rules! vec_mul_rot_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        impl<N: BaseNum + Clone> $mulrhs<N, $tv<N>> for $t<N> {
            #[inline]
            fn binop(left: &$tv<N>, right: &$t<N>) -> $tv<N> {
                *left * right.submat
            }
        }
    )
)

macro_rules! pnt_mul_rot_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        vec_mul_rot_impl!($t, $tv, $mulrhs)
    )
)

macro_rules! inv_impl(
    ($t: ident) => (
        impl<N: Clone> Inv for $t<N> {
            #[inline]
            fn inv(&mut self) -> bool {
                self.transpose();

                // always succeed
                true
            }

            #[inline]
            fn inv_cpy(m: &$t<N>) -> Option<$t<N>> {
                // always succeed
                Some(Transpose::transpose_cpy(m))
            }
        }
    )
)

macro_rules! transpose_impl(
    ($t: ident) => (
        impl<N: Clone> Transpose for $t<N> {
            #[inline]
            fn transpose_cpy(m: &$t<N>) -> $t<N> {
                $t { submat: Transpose::transpose_cpy(&m.submat) }
            }

            #[inline]
            fn transpose(&mut self) {
                self.submat.transpose()
            }
        }
    )
)

macro_rules! row_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Clone + Zero> Row<$tv<N>> for $t<N> {
            #[inline]
            fn nrows(&self) -> uint {
                self.submat.nrows()
            }
            #[inline]
            fn row(&self, i: uint) -> $tv<N> {
                self.submat.row(i)
            }

            #[inline]
            fn set_row(&mut self, i: uint, row: $tv<N>) {
                self.submat.set_row(i, row);
            }
        }
    )
)

macro_rules! col_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Clone + Zero> Col<$tv<N>> for $t<N> {
            #[inline]
            fn ncols(&self) -> uint {
                self.submat.ncols()
            }
            #[inline]
            fn col(&self, i: uint) -> $tv<N> {
                self.submat.col(i)
            }

            #[inline]
            fn set_col(&mut self, i: uint, col: $tv<N>) {
                self.submat.set_col(i, col);
            }
        }
    )
)

macro_rules! index_impl(
    ($t: ident) => (
        impl<N> Index<(uint, uint), N> for $t<N> {
            fn index(&self, i: &(uint, uint)) -> &N {
                &self.submat[*i]
            }
        }

        impl<N> IndexMut<(uint, uint), N> for $t<N> {
            fn index_mut(&mut self, i: &(uint, uint)) -> &mut N {
                &mut self.submat[*i]
            }
        }
    )
)

macro_rules! to_homogeneous_impl(
    ($t: ident, $tm: ident) => (
        impl<N: BaseNum + Clone> ToHomogeneous<$tm<N>> for $t<N> {
            #[inline]
            fn to_homogeneous(&self) -> $tm<N> {
                self.submat.to_homogeneous()
            }
        }
    )
)

macro_rules! approx_eq_impl(
    ($t: ident) => (
        impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
            #[inline]
            fn approx_epsilon(_: Option<$t<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_eq(a: &$t<N>, b: &$t<N>) -> bool {
                ApproxEq::approx_eq(&a.submat, &b.submat)
            }

            #[inline]
            fn approx_eq_eps(a: &$t<N>, b: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&a.submat, &b.submat, epsilon)
            }
        }
    )
)

macro_rules! absolute_impl(
    ($t: ident, $tm: ident) => (
        impl<N: Absolute<N>> Absolute<$tm<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $tm<N> {
                Absolute::abs(&m.submat)
            }
        }
    )
)
