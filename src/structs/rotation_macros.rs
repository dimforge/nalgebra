#![macro_use]

macro_rules! rotation_impl(
    ($t: ident, $submatrix: ident, $vector: ident, $rotvector: ident, $point: ident, $homogeneous: ident) => (

        special_orthogonal_group_impl!($t, $point, $vector);

        impl<N> $t<N> {
            /// This rotation's underlying matrix.
            #[inline]
            pub fn submatrix(&self) -> &$submatrix<N> {
                &self.submatrix
            }
        }


        /*
         *
         * Rotate Vector and Point
         *
         */
        impl<N: BaseNum> Rotate<$vector<N>> for $t<N> {
            #[inline]
            fn rotate(&self, v: &$vector<N>) -> $vector<N> {
                *self * *v
            }

            #[inline]
            fn inverse_rotate(&self, v: &$vector<N>) -> $vector<N> {
                *v * *self
            }
        }

        impl<N: BaseNum> Rotate<$point<N>> for $t<N> {
            #[inline]
            fn rotate(&self, p: &$point<N>) -> $point<N> {
                *self * *p
            }

            #[inline]
            fn inverse_rotate(&self, p: &$point<N>) -> $point<N> {
                *p * *self
            }
        }


        /*
         *
         * Transform Vector and Point
         *
         */
        impl<N: BaseNum> Transform<$vector<N>> for $t<N> {
            #[inline]
            fn transform(&self, v: &$vector<N>) -> $vector<N> {
                self.rotate(v)
            }

            #[inline]
            fn inverse_transform(&self, v: &$vector<N>) -> $vector<N> {
                self.inverse_rotate(v)
            }
        }

        impl<N: BaseNum> Transform<$point<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$point<N>) -> $point<N> {
                self.rotate(p)
            }

            #[inline]
            fn inverse_transform(&self, p: &$point<N>) -> $point<N> {
                self.inverse_rotate(p)
            }
        }



        /*
         *
         * Rotation Matrix
         *
         */
        impl<N: Zero + BaseNum + Cast<f64> + BaseFloat> RotationMatrix<N, $vector<N>, $rotvector<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn to_rotation_matrix(&self) -> $t<N> {
                self.clone()
            }
        }


        /*
         *
         * One
         *
         */
        impl<N: BaseNum> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t { submatrix: ::one() }
            }
        }


        /*
         *
         * Eye
         *
         */
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


        /*
         *
         * Diagonal
         *
         */
        impl<N: Copy + Zero> Diagonal<$vector<N>> for $t<N> {
            #[inline]
            fn from_diagonal(diagonal: &$vector<N>) -> $t<N> {
                $t { submatrix: Diagonal::from_diagonal(diagonal) }
            }

            #[inline]
            fn diagonal(&self) -> $vector<N> {
                self.submatrix.diagonal()
            }
        }


        /*
         *
         * Rotation * Rotation
         *
         */
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


        /*
         *
         * Rotation * Vector
         *
         */
        impl<N: BaseNum> Mul<$vector<N>> for $t<N> {
            type Output = $vector<N>;

            #[inline]
            fn mul(self, right: $vector<N>) -> $vector<N> {
                self.submatrix * right
            }
        }

        impl<N: BaseNum> Mul<$t<N>> for $vector<N> {
            type Output = $vector<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $vector<N> {
                self * right.submatrix
            }
        }

        impl<N: Copy + BaseNum> MulAssign<$t<N>> for $vector<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                *self *= right.submatrix
            }
        }


        /*
         *
         * Rotation * Point
         *
         */
        impl<N: BaseNum> Mul<$point<N>> for $t<N> {
            type Output = $point<N>;

            #[inline]
            fn mul(self, right: $point<N>) -> $point<N> {
                self.submatrix * right
            }
        }

        impl<N: BaseNum> Mul<$t<N>> for $point<N> {
            type Output = $point<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $point<N> {
                self * right.submatrix
            }
        }

        impl<N: Copy + BaseNum> MulAssign<$t<N>> for $point<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                *self *= right.submatrix
            }
        }


        /*
         *
         * Inverse
         *
         */
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


        /*
         *
         * Transpose
         *
         */
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


        /*
         *
         * Row
         *
         */
        impl<N: Copy + Zero> Row<$vector<N>> for $t<N> {
            #[inline]
            fn nrows(&self) -> usize {
                self.submatrix.nrows()
            }
            #[inline]
            fn row(&self, i: usize) -> $vector<N> {
                self.submatrix.row(i)
            }

            #[inline]
            fn set_row(&mut self, i: usize, row: $vector<N>) {
                self.submatrix.set_row(i, row);
            }
        }


        /*
         *
         * Column
         *
         */
        impl<N: Copy + Zero> Column<$vector<N>> for $t<N> {
            #[inline]
            fn ncols(&self) -> usize {
                self.submatrix.ncols()
            }
            #[inline]
            fn column(&self, i: usize) -> $vector<N> {
                self.submatrix.column(i)
            }

            #[inline]
            fn set_column(&mut self, i: usize, column: $vector<N>) {
                self.submatrix.set_column(i, column);
            }
        }


        /*
         *
         * Index
         *
         */
        impl<N> Index<(usize, usize)> for $t<N> {
            type Output = N;

            fn index(&self, i: (usize, usize)) -> &N {
                &self.submatrix[i]
            }
        }


        /*
         *
         * ToHomogeneous
         *
         */
        impl<N: BaseNum> ToHomogeneous<$homogeneous<N>> for $t<N> {
            #[inline]
            fn to_homogeneous(&self) -> $homogeneous<N> {
                self.submatrix.to_homogeneous()
            }
        }


        /*
         *
         * ApproxEq
         *
         */
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


        /*
         *
         * Absolute
         *
         */
        impl<N: Absolute<N>> Absolute<$submatrix<N>> for $t<N> {
            #[inline]
            fn abs(m: &$t<N>) -> $submatrix<N> {
                Absolute::abs(&m.submatrix)
            }
        }



        /*
         *
         * Display
         *
         */
        impl<N: fmt::Display + BaseFloat> fmt::Display for $t<N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let precision = f.precision().unwrap_or(3);

                try!(writeln!(f, "Rotation matrix {{"));
                try!(write!(f, "{:.*}", precision, self.submatrix));
                writeln!(f, "}}")
            }
        }


        /*
         *
         * Arbitrary
         *
         */
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + BaseFloat> Arbitrary for $t<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t::new(Arbitrary::arbitrary(g))
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
