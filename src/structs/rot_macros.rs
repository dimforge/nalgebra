#![macro_escape]

macro_rules! submat_impl(
    ($t: ident, $submat: ident) => (
        impl<N> $t<N> {
            #[inline]
            pub fn submat<'r>(&'r self) -> &'r $submat<N> {
                &'r self.submat
            }
        }
    )
)

macro_rules! rotate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Num + Clone> Rotate<$tv<N>> for $t<N> {
            #[inline]
            fn rotate(&self, v: &$tv<N>) -> $tv<N> {
                self * *v
            }

            #[inline]
            fn inv_rotate(&self, v: &$tv<N>) -> $tv<N> {
                v * *self
            }
        }
    )
)

macro_rules! transform_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Num + Clone> Transform<$tv<N>> for $t<N> {
            #[inline]
            fn transform(&self, v: &$tv<N>) -> $tv<N> {
                self.rotate(v)
            }

            #[inline]
            fn inv_transform(&self, v: &$tv<N>) -> $tv<N> {
                self.inv_rotate(v)
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
        impl<N: Cast<f32> + FloatMath + Clone>
        RotationMatrix<$tlv<N>, $tav<N>, $t<N>> for $t<N> {
            #[inline]
            fn to_rot_mat(&self) -> $t<N> {
                self.clone()
            }
        }
    )
)

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: Num + Clone> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t { submat: One::one() }
            }
        }
    )
)

macro_rules! rot_mul_rot_impl(
    ($t: ident, $mulrhs: ident) => (
        impl<N: Num + Clone> $mulrhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t { submat: left.submat * right.submat }
            }
        }
    )
)

macro_rules! rot_mul_vec_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        impl<N: Num + Clone> $mulrhs<N, $tv<N>> for $tv<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$tv<N>) -> $tv<N> {
                left.submat * *right
            }
        }
    )
)

macro_rules! vec_mul_rot_impl(
    ($t: ident, $tv: ident, $mulrhs: ident) => (
        impl<N: Num + Clone> $mulrhs<N, $tv<N>> for $t<N> {
            #[inline]
            fn binop(left: &$tv<N>, right: &$t<N>) -> $tv<N> {
                *left * right.submat
            }
        }
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

macro_rules! to_homogeneous_impl(
    ($t: ident, $tm: ident) => (
        impl<N: Num + Clone> ToHomogeneous<$tm<N>> for $t<N> {
            #[inline]
            fn to_homogeneous(m: &$t<N>) -> $tm<N> {
                ToHomogeneous::to_homogeneous(&m.submat)
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
        impl<N: Signed> Absolute<$tm<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $tm<N> {
                Absolute::abs(&m.submat)
            }
        }
    )
)
