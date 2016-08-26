#![macro_use]

macro_rules! isometry_impl(
    ($t: ident, $rotmatrix: ident, $vector: ident, $rotvector: ident, $point: ident,
     $homogeneous: ident) => (
        impl<N: BaseFloat> $t<N> {
            /// Creates a new isometry from an axis-angle rotation, and a vector.
            #[inline]
            pub fn new(translation: $vector<N>, rotation: $rotvector<N>) -> $t<N> {
                $t {
                    rotation:    $rotmatrix::new(rotation),
                    translation: translation
                }
            }

            /// Creates a new isometry from a rotation matrix and a vector.
            #[inline]
            pub fn from_rotation_matrix(translation: $vector<N>, rotation: $rotmatrix<N>) -> $t<N> {
                $t {
                    rotation:    rotation,
                    translation: translation
                }
            }
        }


        /*
         *
         * RotationMatrix
         *
         */
        impl<N: Cast<f64> + BaseFloat>
        RotationMatrix<N, $vector<N>, $rotvector<N>> for $t<N> {
            type Output = $rotmatrix<N>;

            #[inline]
            fn to_rotation_matrix(&self) -> $rotmatrix<N> {
                self.rotation
            }
        }


        /*
         *
         * One
         *
         */
        impl<N: BaseFloat> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::from_rotation_matrix(::zero(), ::one())
            }
        }


        /*
         *
         * Isometry × Isometry
         *
         */
        impl<N: BaseFloat> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self.translation + self.rotation * right.translation,
                    self.rotation * right.rotation)
            }
        }

        impl<N: BaseFloat> MulAssign<$t<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                self.translation += self.rotation * right.translation;
                self.rotation    *= right.rotation;
            }
        }


        /*
         *
         * Isometry × Rotation
         *
         */
        impl<N: BaseFloat> Mul<$rotmatrix<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $rotmatrix<N>) -> $t<N> {
                $t::from_rotation_matrix(self.translation, self.rotation * right)
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $rotmatrix<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self * right.translation,
                    self * right.rotation)
            }
        }

        impl<N: BaseFloat> MulAssign<$rotmatrix<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $rotmatrix<N>) {
                self.rotation *= right
            }
        }


        /*
         *
         * Isometry × Point
         *
         */
        impl<N: BaseNum> Mul<$point<N>> for $t<N> {
            type Output = $point<N>;

            #[inline]
            fn mul(self, right: $point<N>) -> $point<N> {
                self.rotation * right + self.translation
            }
        }


        /*
         *
         * Isometry × Vector
         *
         */
        impl<N: BaseNum> Mul<$vector<N>> for $t<N> {
            type Output = $vector<N>;

            #[inline]
            fn mul(self, right: $vector<N>) -> $vector<N> {
                self.rotation * right
            }
        }


        /*
         *
         * Translation
         *
         */
        impl<N: BaseFloat> Translation<$vector<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $vector<N> {
                self.translation
            }

            #[inline]
            fn inverse_translation(&self) -> $vector<N> {
                -self.translation
            }

            #[inline]
            fn append_translation_mut(&mut self, t: &$vector<N>) {
                self.translation = *t + self.translation
            }

            #[inline]
            fn append_translation(&self, t: &$vector<N>) -> $t<N> {
                $t::from_rotation_matrix(*t + self.translation, self.rotation)
            }

            #[inline]
            fn prepend_translation_mut(&mut self, t: &$vector<N>) {
                self.translation = self.translation + self.rotation * *t
            }

            #[inline]
            fn prepend_translation(&self, t: &$vector<N>) -> $t<N> {
                $t::from_rotation_matrix(self.translation + self.rotation * *t, self.rotation)
            }

            #[inline]
            fn set_translation(&mut self, t: $vector<N>) {
                self.translation = t
            }
        }


        /*
         *
         * Translate
         *
         */
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Translate<$point<N>> for $t<N> {
            #[inline]
            fn translate(&self, v: &$point<N>) -> $point<N> {
                *v + self.translation
            }

            #[inline]
            fn inverse_translate(&self, v: &$point<N>) -> $point<N> {
                *v - self.translation
            }
        }


        /*
         *
         * Rotation
         *
         */
        impl<N: Cast<f64> + BaseFloat> Rotation<$rotvector<N>> for $t<N> {
            #[inline]
            fn rotation(&self) -> $rotvector<N> {
                self.rotation.rotation()
            }

            #[inline]
            fn inverse_rotation(&self) -> $rotvector<N> {
                self.rotation.inverse_rotation()
            }

            #[inline]
            fn append_rotation_mut(&mut self, rotation: &$rotvector<N>) {
                let delta = $rotmatrix::new(*rotation);

                self.rotation    = delta * self.rotation;
                self.translation = delta * self.translation;
            }

            #[inline]
            fn append_rotation(&self, rotation: &$rotvector<N>) -> $t<N> {
                let delta = $rotmatrix::new(*rotation);

                $t::from_rotation_matrix(delta * self.translation, delta * self.rotation)
            }

            #[inline]
            fn prepend_rotation_mut(&mut self, rotation: &$rotvector<N>) {
                let delta = $rotmatrix::new(*rotation);

                self.rotation = self.rotation * delta;
            }

            #[inline]
            fn prepend_rotation(&self, rotation: &$rotvector<N>) -> $t<N> {
                let delta = $rotmatrix::new(*rotation);

                $t::from_rotation_matrix(self.translation, self.rotation * delta)
            }

            #[inline]
            fn set_rotation(&mut self, rotation: $rotvector<N>) {
                // FIXME: should the translation be changed too?
                self.rotation.set_rotation(rotation)
            }
        }


        /*
         *
         * Rotate
         *
         */
        impl<N: BaseNum> Rotate<$vector<N>> for $t<N> {
            #[inline]
            fn rotate(&self, v: &$vector<N>) -> $vector<N> {
                self.rotation.rotate(v)
            }

            #[inline]
            fn inverse_rotate(&self, v: &$vector<N>) -> $vector<N> {
                self.rotation.inverse_rotate(v)
            }
        }


        /*
         *
         * Transformation
         *
         */
        impl<N: BaseFloat> Transformation<$t<N>> for $t<N> {
            fn transformation(&self) -> $t<N> {
                *self
            }

            fn inverse_transformation(&self) -> $t<N> {
                // inversion will never fails
                Inverse::inverse(self).unwrap()
            }

            fn append_transformation_mut(&mut self, t: &$t<N>) {
                *self = *t * *self
            }

            fn append_transformation(&self, t: &$t<N>) -> $t<N> {
                *t * *self
            }

            fn prepend_transformation_mut(&mut self, t: &$t<N>) {
                *self = *self * *t
            }

            fn prepend_transformation(&self, t: &$t<N>) -> $t<N> {
                *self * *t
            }

            fn set_transformation(&mut self, t: $t<N>) {
                *self = t
            }
        }


        /*
         *
         * Transform
         *
         */
        impl<N: BaseNum> Transform<$point<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$point<N>) -> $point<N> {
                self.rotation.transform(p) + self.translation
            }

            #[inline]
            fn inverse_transform(&self, p: &$point<N>) -> $point<N> {
                self.rotation.inverse_transform(&(*p - self.translation))
            }
        }


        /*
         *
         * Inverse
         *
         */
        impl<N: BaseNum + Neg<Output = N>> Inverse for $t<N> {
            #[inline]
            fn inverse_mut(&mut self) -> bool {
                self.rotation.inverse_mut();
                self.translation = self.rotation * -self.translation;
                // always succeed
                true
            }

            #[inline]
            fn inverse(&self) -> Option<$t<N>> {
                let mut res = *self;
                res.inverse_mut();
                // always succeed
                Some(res)
            }
        }


        /*
         *
         * ToHomogeneous
         *
         */
        impl<N: BaseNum> ToHomogeneous<$homogeneous<N>> for $t<N> {
            fn to_homogeneous(&self) -> $homogeneous<N> {
                let mut res = self.rotation.to_homogeneous();

                // copy the translation
                let dimension = Dimension::dimension(None::<$homogeneous<N>>);

                res.set_column(dimension - 1, self.translation.as_point().to_homogeneous().to_vector());

                res
            }
        }


        /*
         *
         * ApproxEq
         *
         */
        impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
            #[inline]
            fn approx_epsilon() -> N {
                <N as ApproxEq<N>>::approx_epsilon()
            }

            #[inline]
            fn approx_ulps() -> u32 {
                <N as ApproxEq<N>>::approx_ulps()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&self.rotation, &other.rotation, epsilon) &&
                    ApproxEq::approx_eq_eps(&self.translation, &other.translation, epsilon)
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
                ApproxEq::approx_eq_ulps(&self.rotation, &other.rotation, ulps) &&
                    ApproxEq::approx_eq_ulps(&self.translation, &other.translation, ulps)
            }
        }


        /*
         *
         * Rand
         *
         */
        impl<N: Rand + BaseFloat> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                $t::new(rng.gen(), rng.gen())
            }
        }


        /*
         *
         * AbsoluteRotate
         *
         */
        impl<N: BaseFloat> AbsoluteRotate<$vector<N>> for $t<N> {
            #[inline]
            fn absolute_rotate(&self, v: &$vector<N>) -> $vector<N> {
                self.rotation.absolute_rotate(v)
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
                $t::from_rotation_matrix(
                    Arbitrary::arbitrary(g),
                    Arbitrary::arbitrary(g)
                )
            }
        }


        /*
         *
         * Display
         *
         */
        impl<N: fmt::Display + BaseFloat> fmt::Display for $t<N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                try!(writeln!(f, "Isometry {{"));

                if let Some(precision) = f.precision() {
                    try!(writeln!(f, "... translation: {:.*}", precision, self.translation));
                    try!(writeln!(f, "... rotation matrix:"));
                    try!(write!(f, "{:.*}", precision, *self.rotation.submatrix()));
                }
                else {
                    try!(writeln!(f, "... translation: {}", self.translation));
                    try!(writeln!(f, "... rotation matrix:"));
                    try!(write!(f, "{}", *self.rotation.submatrix()));
                }

                writeln!(f, "}}")
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
