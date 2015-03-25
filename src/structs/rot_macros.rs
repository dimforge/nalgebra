#![macro_use]

macro_rules! submat_impl(
    ($t: ident, $submat: ident) => (
        impl<N> $t<N> {
            #[inline]
            pub fn submat<'r>(&'r self) -> &'r $submat<N> {
                &self.submat
            }
        }
    )
);

macro_rules! rotate_impl(
    ($t: ident, $tv: ident, $tp: ident) => (
        impl<N: BaseNum> Rotate<$tv<N>> for $t<N> {
            #[inline]
            fn rotate(&self, v: &$tv<N>) -> $tv<N> {
                *self * *v
            }

            #[inline]
            fn inv_rotate(&self, v: &$tv<N>) -> $tv<N> {
                *v * *self
            }
        }

        impl<N: BaseNum> Rotate<$tp<N>> for $t<N> {
            #[inline]
            fn rotate(&self, p: &$tp<N>) -> $tp<N> {
                *self * *p
            }

            #[inline]
            fn inv_rotate(&self, p: &$tp<N>) -> $tp<N> {
                *p * *self
            }
        }
    )
);

macro_rules! transform_impl(
    ($t: ident, $tv: ident, $tp: ident) => (
        impl<N: BaseNum> Transform<$tv<N>> for $t<N> {
            #[inline]
            fn transform(&self, v: &$tv<N>) -> $tv<N> {
                self.rotate(v)
            }

            #[inline]
            fn inv_transform(&self, v: &$tv<N>) -> $tv<N> {
                self.inv_rotate(v)
            }
        }

        impl<N: BaseNum> Transform<$tp<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$tp<N>) -> $tp<N> {
                self.rotate(p)
            }

            #[inline]
            fn inv_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.inv_rotate(p)
            }
        }
    )
);

macro_rules! dim_impl(
    ($t: ident, $dim: expr) => (
        impl<N> Dim for $t<N> {
            #[inline]
            fn dim(_: Option<$t<N>>) -> usize {
                $dim
            }
        }
    )
);

macro_rules! rotation_matrix_impl(
    ($t: ident, $tlv: ident, $tav: ident) => (
        impl<N: Zero + BaseNum + Cast<f64> + BaseFloat> RotationMatrix<N, $tlv<N>, $tav<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn to_rot_mat(&self) -> $t<N> {
                self.clone()
            }
        }
    )
);

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: BaseNum> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t { submat: ::one() }
            }
        }
    )
);

macro_rules! rot_mul_rot_impl(
    ($t: ident) => (
        impl<N: BaseNum> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t { submat: self.submat * right.submat }
            }
        }
    )
);

macro_rules! rot_mul_vec_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.submat * right
            }
        }
    )
);

macro_rules! rot_mul_pnt_impl(
    ($t: ident, $tv: ident) => (
        rot_mul_vec_impl!($t, $tv);
    )
);

macro_rules! vec_mul_rot_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$t<N>> for $tv<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $tv<N> {
                self * right.submat
            }
        }
    )
);

macro_rules! pnt_mul_rot_impl(
    ($t: ident, $tv: ident) => (
        vec_mul_rot_impl!($t, $tv);
    )
);

macro_rules! inv_impl(
    ($t: ident) => (
        impl<N: Copy> Inv for $t<N> {
            #[inline]
            fn inv_mut(&mut self) -> bool {
                self.transpose_mut();

                // always succeed
                true
            }

            #[inline]
            fn inv(&self) -> Option<$t<N>> {
                // always succeed
                Some(self.transpose())
            }
        }
    )
);

macro_rules! transpose_impl(
    ($t: ident) => (
        impl<N: Copy> Transpose for $t<N> {
            #[inline]
            fn transpose(&self) -> $t<N> {
                $t { submat: Transpose::transpose(&self.submat) }
            }

            #[inline]
            fn transpose_mut(&mut self) {
                self.submat.transpose_mut()
            }
        }
    )
);

macro_rules! row_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Zero> Row<$tv<N>> for $t<N> {
            #[inline]
            fn nrows(&self) -> usize {
                self.submat.nrows()
            }
            #[inline]
            fn row(&self, i: usize) -> $tv<N> {
                self.submat.row(i)
            }

            #[inline]
            fn set_row(&mut self, i: usize, row: $tv<N>) {
                self.submat.set_row(i, row);
            }
        }
    )
);

macro_rules! col_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Zero> Col<$tv<N>> for $t<N> {
            #[inline]
            fn ncols(&self) -> usize {
                self.submat.ncols()
            }
            #[inline]
            fn col(&self, i: usize) -> $tv<N> {
                self.submat.col(i)
            }

            #[inline]
            fn set_col(&mut self, i: usize, col: $tv<N>) {
                self.submat.set_col(i, col);
            }
        }
    )
);

macro_rules! index_impl(
    ($t: ident) => (
        impl<N> Index<(usize, usize)> for $t<N> {
            type Output = N;

            fn index(&self, i: (usize, usize)) -> &N {
                &self.submat[i]
            }
        }
    )
);

macro_rules! to_homogeneous_impl(
    ($t: ident, $tm: ident) => (
        impl<N: BaseNum> ToHomogeneous<$tm<N>> for $t<N> {
            #[inline]
            fn to_homogeneous(&self) -> $tm<N> {
                self.submat.to_homogeneous()
            }
        }
    )
);

macro_rules! approx_eq_impl(
    ($t: ident) => (
        impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
            #[inline]
            fn approx_epsilon(_: Option<$t<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_ulps(_: Option<$t<N>>) -> u32 {
                ApproxEq::approx_ulps(None::<N>)
            }

            #[inline]
            fn approx_eq(&self, other: &$t<N>) -> bool {
                ApproxEq::approx_eq(&self.submat, &other.submat)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&self.submat, &other.submat, epsilon)
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
                ApproxEq::approx_eq_ulps(&self.submat, &other.submat, ulps)
            }
        }
    )
);

macro_rules! absolute_impl(
    ($t: ident, $tm: ident) => (
        impl<N: Absolute<N>> Absolute<$tm<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $tm<N> {
                Absolute::abs(&m.submat)
            }
        }
    )
);
