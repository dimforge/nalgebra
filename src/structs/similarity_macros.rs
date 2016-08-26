#![macro_use]

macro_rules! similarity_impl(
    ($t: ident,
     $isometry: ident, $rotation_matrix: ident,
     $vector: ident, $rotvector: ident,
     $point: ident,
     $homogeneous_matrix: ident) => (
        impl<N: BaseFloat> $t<N> {
            /*
             *
             * Constructors.
             *
             */
            /// Creates a new similarity transformation from a vector, an axis-angle rotation, and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new(translation: $vector<N>, rotation: $rotvector<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: $isometry::new(translation, rotation)
                }
            }

            /// Creates a new similarity transformation from a rotation matrix, a vector, and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn from_rotation_matrix(translation: $vector<N>, rotation: $rotation_matrix<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: $isometry::from_rotation_matrix(translation, rotation)
                }
            }

            /// Creates a new similarity transformation from an isometry and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn from_isometry(isometry: $isometry<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: isometry
                }
            }

            /*
             *
             * Methods related to scaling.
             *
             */
            /// The scale factor of this similarity transformation.
            #[inline]
            pub fn scale(&self) -> N {
                self.scale
            }

            /// The inverse scale factor of this similarity transformation.
            #[inline]
            pub fn inverse_scale(&self) -> N {
                ::one::<N>() / self.scale
            }

            /// Appends in-place a scale to this similarity transformation.
            #[inline]
            pub fn append_scale_mut(&mut self, s: &N) {
                assert!(!s.is_zero(), "Cannot append a zero scale to a similarity transformation.");
                self.scale = *s * self.scale;
                self.isometry.translation = self.isometry.translation * *s;
            }

            /// Appends a scale to this similarity transformation.
            #[inline]
            pub fn append_scale(&self, s: &N) -> $t<N> {
                assert!(!s.is_zero(), "Cannot append a zero scale to a similarity transformation.");
                $t::from_rotation_matrix(self.isometry.translation * *s, self.isometry.rotation, self.scale * *s)
            }

            /// Prepends in-place a scale to this similarity transformation.
            #[inline]
            pub fn prepend_scale_mut(&mut self, s: &N) {
                assert!(!s.is_zero(), "Cannot prepend a zero scale to a similarity transformation.");
                self.scale = self.scale * *s;
            }

            /// Prepends a scale to this similarity transformation.
            #[inline]
            pub fn prepend_scale(&self, s: &N) -> $t<N> {
                assert!(!s.is_zero(), "A similarity transformation scale must not be zero.");
                $t::from_isometry(self.isometry, self.scale * *s)
            }

            /// Sets the scale of this similarity transformation.
            #[inline]
            pub fn set_scale(&mut self, s: N) {
                assert!(!s.is_zero(), "A similarity transformation scale must not be zero.");
                self.scale = s
            }
        }

        /*
         *
         * One Impl.
         *
         */
        impl<N: BaseFloat> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::from_isometry(::one(), ::one())
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
         * Similarity × Similarity
         *
         */
        impl<N: BaseFloat> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self.isometry.translation + self.isometry.rotation * (right.isometry.translation * self.scale),
                    self.isometry.rotation * right.isometry.rotation,
                    self.scale * right.scale)
            }
        }

        impl<N: BaseFloat> MulAssign<$t<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                self.isometry.translation += self.isometry.rotation * (right.isometry.translation * self.scale);
                self.isometry.rotation    *= right.isometry.rotation;
                self.scale                *= right.scale;
            }
        }


        /*
         *
         * Similarity × Isometry
         *
         */
        impl<N: BaseFloat> Mul<$isometry<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $isometry<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self.isometry.translation + self.isometry.rotation * (right.translation * self.scale),
                    self.isometry.rotation * right.rotation,
                    self.scale)
            }
        }

        impl<N: BaseFloat> MulAssign<$isometry<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $isometry<N>) {
                self.isometry.translation += self.isometry.rotation * (right.translation * self.scale);
                self.isometry.rotation    *= right.rotation;
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $isometry<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self.translation + self.rotation * right.isometry.translation,
                    self.rotation * right.isometry.rotation,
                    right.scale)
            }
        }

        /*
         *
         * Similarity × Rotation
         *
         */
        impl<N: BaseFloat> Mul<$rotation_matrix<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $rotation_matrix<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self.isometry.translation,
                    self.isometry.rotation * right,
                    self.scale)
            }
        }

        impl<N: BaseFloat> MulAssign<$rotation_matrix<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $rotation_matrix<N>) {
                self.isometry.rotation *= right;
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $rotation_matrix<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::from_rotation_matrix(
                    self * right.isometry.translation,
                    self * right.isometry.rotation,
                    right.scale)
            }
        }

        /*
         *
         * Similarity × { Point, Vector }
         *
         */
        impl<N: BaseNum> Mul<$vector<N>> for $t<N> {
            type Output = $vector<N>;

            #[inline]
            fn mul(self, right: $vector<N>) -> $vector<N> {
                self.isometry * (right * self.scale)
            }
        }

        impl<N: BaseNum> Mul<$point<N>> for $t<N> {
            type Output = $point<N>;

            #[inline]
            fn mul(self, right: $point<N>) -> $point<N> {
                self.isometry * (right * self.scale)
            }
        }

        // NOTE: there is no viable pre-multiplication definition because of the translation
        // component.

        /*
         *
         * Similarity × Point
         *
         */
        impl<N: BaseNum> Transform<$point<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$point<N>) -> $point<N> {
                self.isometry.transform(&(*p * self.scale))
            }

            #[inline]
            fn inverse_transform(&self, p: &$point<N>) -> $point<N> {
                self.isometry.inverse_transform(p) / self.scale
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
                self.scale = ::one::<N>() / self.scale;
                self.isometry.inverse_mut();
                // We multiply (instead of dividing) by self.scale because it has already been
                // inverted.
                self.isometry.translation = self.isometry.translation * self.scale;

                // Always succeed.
                true
            }

            #[inline]
            fn inverse(&self) -> Option<$t<N>> {
                let mut res = *self;
                res.inverse_mut();

                // Always succeed.
                Some(res)
            }
        }


        /*
         *
         * ToHomogeneous
         *
         */
        impl<N: BaseNum> ToHomogeneous<$homogeneous_matrix<N>> for $t<N> {
            fn to_homogeneous(&self) -> $homogeneous_matrix<N> {
                let mut res = (*self.isometry.rotation.submatrix() * self.scale).to_homogeneous();

                // copy the translation
                let dimension = Dimension::dimension(None::<$homogeneous_matrix<N>>);

                res.set_column(dimension - 1, self.isometry.translation.as_point().to_homogeneous().to_vector());

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
                ApproxEq::approx_eq_eps(&self.scale, &other.scale, epsilon) &&
                    ApproxEq::approx_eq_eps(&self.isometry, &other.isometry, epsilon)
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
                ApproxEq::approx_eq_ulps(&self.scale, &other.scale, ulps) &&
                    ApproxEq::approx_eq_ulps(&self.isometry, &other.isometry, ulps)
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
                let mut scale: N = rng.gen();
                while scale.is_zero() {
                    scale = rng.gen();
                }

                $t::from_isometry(rng.gen(), scale)
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
                $t::from_isometry(
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
                try!(writeln!(f, "Similarity transformation {{"));

                if let Some(precision) = f.precision() {
                    try!(writeln!(f, "... scale factor: {:.*}", precision, self.scale));
                    try!(writeln!(f, "... translation: {:.*}", precision, self.isometry.translation));
                    try!(writeln!(f, "... rotation matrix:"));
                    try!(write!(f, "{:.*}", precision, *self.isometry.rotation.submatrix()));
                }
                else {
                    try!(writeln!(f, "... scale factor: {}", self.scale));
                    try!(writeln!(f, "... translation: {}", self.isometry.translation));
                    try!(writeln!(f, "... rotation matrix:"));
                    try!(write!(f, "{}", *self.isometry.rotation.submatrix()));
                }

                writeln!(f, "}}")
            }
        }
    )
);
