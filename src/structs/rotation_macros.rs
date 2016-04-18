#![macro_use]

macro_rules! submat_impl(
    ($t: ident, $submatrix: ident) => (
        impl<N> $t<N> {
            /// This rotation's underlying matrix.
            #[inline]
            pub fn submatrix<'r>(&'r self) -> &'r $submatrix<N> {
                &self.submatrix
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
            fn inverse_rotate(&self, v: &$tv<N>) -> $tv<N> {
                *v * *self
            }
        }

        impl<N: BaseNum> Rotate<$tp<N>> for $t<N> {
            #[inline]
            fn rotate(&self, p: &$tp<N>) -> $tp<N> {
                *self * *p
            }

            #[inline]
            fn inverse_rotate(&self, p: &$tp<N>) -> $tp<N> {
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
            fn inverse_transform(&self, v: &$tv<N>) -> $tv<N> {
                self.inverse_rotate(v)
            }
        }

        impl<N: BaseNum> Transform<$tp<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$tp<N>) -> $tp<N> {
                self.rotate(p)
            }

            #[inline]
            fn inverse_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.inverse_rotate(p)
            }
        }
    )
);

macro_rules! dim_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> Dimension for $t<N> {
            #[inline]
            fn dimension(_: Option<$t<N>>) -> usize {
                $dimension
            }
        }
    )
);

macro_rules! rotation_matrix_impl(
    ($t: ident, $tlv: ident, $tav: ident) => (
        impl<N: Zero + BaseNum + Cast<f64> + BaseFloat> RotationMatrix<N, $tlv<N>, $tav<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn to_rotation_matrix(&self) -> $t<N> {
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
                $t { submatrix: ::one() }
            }
        }
    )
);

macro_rules! eye_impl(
    ($t: ident) => (
        impl<N: BaseNum> Eye for $t<N> {
            #[inline]
            fn new_identity(dimension: usize) -> $t<N> {
                if dimension != ::dimension::<$t<N>>() {
                    panic!("Dimension mismatch: should be {}, got {}.", ::dimension::<$t<N>>(), dimension);
                }
                else {
                    ::one()
                }
            }
        }
    )
);

macro_rules! diag_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Zero> Diagonal<$tv<N>> for $t<N> {
            #[inline]
            fn from_diagonal(diagonal: &$tv<N>) -> $t<N> {
                $t { submatrix: Diagonal::from_diagonal(diagonal) }
            }

            #[inline]
            fn diagonal(&self) -> $tv<N> {
                self.submatrix.diagonal()
            }
        }
    )
);

macro_rules! rotation_mul_rotation_impl(
    ($t: ident) => (
        impl<N: BaseNum> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t { submatrix: self.submatrix * right.submatrix }
            }
        }

        impl<N: Copy + BaseNum> MulAssign<$t<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                self.submatrix *= right.submatrix
            }
        }
    )
);

macro_rules! rotation_mul_vec_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.submatrix * right
            }
        }
    )
);

macro_rules! rotation_mul_point_impl(
    ($t: ident, $tv: ident) => (
        rotation_mul_vec_impl!($t, $tv);
    )
);

macro_rules! vec_mul_rotation_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$t<N>> for $tv<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $tv<N> {
                self * right.submatrix
            }
        }

        impl<N: Copy + BaseNum> MulAssign<$t<N>> for $tv<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                *self *= right.submatrix
            }
        }
    )
);

macro_rules! point_mul_rotation_impl(
    ($t: ident, $tv: ident) => (
        vec_mul_rotation_impl!($t, $tv);
    )
);

macro_rules! inverse_impl(
    ($t: ident) => (
        impl<N: Copy> Inverse for $t<N> {
            #[inline]
            fn inverse_mut(&mut self) -> bool {
                self.transpose_mut();

                // always succeed
                true
            }

            #[inline]
            fn inverse(&self) -> Option<$t<N>> {
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
                $t { submatrix: Transpose::transpose(&self.submatrix) }
            }

            #[inline]
            fn transpose_mut(&mut self) {
                self.submatrix.transpose_mut()
            }
        }
    )
);

macro_rules! row_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Zero> Row<$tv<N>> for $t<N> {
            #[inline]
            fn nrows(&self) -> usize {
                self.submatrix.nrows()
            }
            #[inline]
            fn row(&self, i: usize) -> $tv<N> {
                self.submatrix.row(i)
            }

            #[inline]
            fn set_row(&mut self, i: usize, row: $tv<N>) {
                self.submatrix.set_row(i, row);
            }
        }
    )
);

macro_rules! column_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Zero> Column<$tv<N>> for $t<N> {
            #[inline]
            fn ncols(&self) -> usize {
                self.submatrix.ncols()
            }
            #[inline]
            fn column(&self, i: usize) -> $tv<N> {
                self.submatrix.column(i)
            }

            #[inline]
            fn set_column(&mut self, i: usize, column: $tv<N>) {
                self.submatrix.set_column(i, column);
            }
        }
    )
);

macro_rules! index_impl(
    ($t: ident) => (
        impl<N> Index<(usize, usize)> for $t<N> {
            type Output = N;

            fn index(&self, i: (usize, usize)) -> &N {
                &self.submatrix[i]
            }
        }
    )
);

macro_rules! to_homogeneous_impl(
    ($t: ident, $tm: ident) => (
        impl<N: BaseNum> ToHomogeneous<$tm<N>> for $t<N> {
            #[inline]
            fn to_homogeneous(&self) -> $tm<N> {
                self.submatrix.to_homogeneous()
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
                ApproxEq::approx_eq(&self.submatrix, &other.submatrix)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&self.submatrix, &other.submatrix, epsilon)
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
                ApproxEq::approx_eq_ulps(&self.submatrix, &other.submatrix, ulps)
            }
        }
    )
);

macro_rules! absolute_impl(
    ($t: ident, $tm: ident) => (
        impl<N: Absolute<N>> Absolute<$tm<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $tm<N> {
                Absolute::abs(&m.submatrix)
            }
        }
    )
);

macro_rules! rotation_display_impl(
    ($t: ident) => (
        impl<N: fmt::Display + BaseFloat> fmt::Display for $t<N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let precision = f.precision().unwrap_or(3);

                try!(writeln!(f, "Rotation matrix {{"));
                try!(write!(f, "{:.*}", precision, self.submatrix));
                writeln!(f, "}}")
            }
        }
    )
);
