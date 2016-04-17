#![macro_use]

macro_rules! sim_impl(
    ($t: ident, $isometry: ident, $rotation_matrix: ident, $subvector: ident, $subrotvector: ident) => (
        impl<N: BaseFloat> $t<N> {
            /// Creates a new similarity transformation from a vector, an axis-angle rotation, and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new(translation: $subvector<N>, rotation: $subrotvector<N>, scale: N) -> $t<N> {
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
            pub fn new_with_rotation_matrix(translation: $subvector<N>, rotation: $rotation_matrix<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: $isometry::new_with_rotation_matrix(translation, rotation)
                }
            }

            /// Creates a new similarity transformation from an isometry and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new_with_isometry(isometry: $isometry<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: isometry
                }
            }
        }
    )
);

macro_rules! sim_scale_impl(
    ($t: ident) => (
        impl<N: BaseFloat> $t<N> {
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
                $t::new_with_rotation_matrix(self.isometry.translation * *s, self.isometry.rotation, self.scale * *s)
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
                $t::new_with_isometry(self.isometry, self.scale * *s)
            }

            /// Sets the scale of this similarity transformation.
            #[inline]
            pub fn set_scale(&mut self, s: N) {
                assert!(!s.is_zero(), "A similarity transformation scale must not be zero.");
                self.scale = s
            }
        }
    )
);

macro_rules! sim_one_impl(
    ($t: ident) => (
        impl<N: BaseFloat> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new_with_isometry(::one(), ::one())
            }
        }
    )
);

macro_rules! sim_mul_sim_impl(
    ($t: ident) => (
        impl<N: BaseFloat> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
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
    )
);

macro_rules! sim_mul_isometry_impl(
    ($t: ident, $ti: ident) => (
        impl<N: BaseFloat> Mul<$ti<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $ti<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
                    self.isometry.translation + self.isometry.rotation * (right.translation * self.scale),
                    self.isometry.rotation * right.rotation,
                    self.scale)
            }
        }

        impl<N: BaseFloat> MulAssign<$ti<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $ti<N>) {
                self.isometry.translation += self.isometry.rotation * (right.translation * self.scale);
                self.isometry.rotation    *= right.rotation;
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $ti<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
                    self.translation + self.rotation * right.isometry.translation,
                    self.rotation * right.isometry.rotation,
                    right.scale)
            }
        }
    )
);

macro_rules! sim_mul_rotation_impl(
    ($t: ident, $tr: ident) => (
        impl<N: BaseFloat> Mul<$tr<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $tr<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
                    self.isometry.translation,
                    self.isometry.rotation * right,
                    self.scale)
            }
        }

        impl<N: BaseFloat> MulAssign<$tr<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $tr<N>) {
                self.isometry.rotation *= right;
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $tr<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
                    self * right.isometry.translation,
                    self * right.isometry.rotation,
                    right.scale)
            }
        }
    )
);

macro_rules! sim_mul_point_vec_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.isometry * (right * self.scale)
            }
        }


        // NOTE: there is no viable pre-multiplication definition because of the translation
        // component.
    )
);

macro_rules! sim_transform_impl(
    ($t: ident, $tp: ident) => (
        impl<N: BaseNum> Transform<$tp<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$tp<N>) -> $tp<N> {
                self.isometry.transform(&(*p * self.scale))
            }

            #[inline]
            fn inverse_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.isometry.inverse_transform(p) / self.scale
            }
        }
    )
);

macro_rules! sim_inverse_impl(
    ($t: ident) => (
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
    )
);

macro_rules! sim_to_homogeneous_impl(
    ($t: ident, $th: ident) => (
        impl<N: BaseNum> ToHomogeneous<$th<N>> for $t<N> {
            fn to_homogeneous(&self) -> $th<N> {
                let mut res = (*self.isometry.rotation.submatrix() * self.scale).to_homogeneous();

                // copy the translation
                let dimension = Dimension::dimension(None::<$th<N>>);

                res.set_col(dimension - 1, self.isometry.translation.as_point().to_homogeneous().to_vector());

                res
            }
        }
    )
);

macro_rules! sim_approx_eq_impl(
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
    )
);

macro_rules! sim_rand_impl(
    ($t: ident) => (
        impl<N: Rand + BaseFloat> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                let mut scale: N = rng.gen();
                while scale.is_zero() {
                    scale = rng.gen();
                }

                $t::new_with_isometry(rng.gen(), scale)
            }
        }
    )
);

macro_rules! sim_arbitrary_impl(
    ($t: ident) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + BaseFloat> Arbitrary for $t<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t::new_with_isometry(
                    Arbitrary::arbitrary(g),
                    Arbitrary::arbitrary(g)
                )
            }
        }
    )
);

macro_rules! sim_display_impl(
    ($t: ident) => (
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
