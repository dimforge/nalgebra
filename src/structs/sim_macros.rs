#![macro_use]

macro_rules! sim_impl(
    ($t: ident, $iso: ident, $rotmat: ident, $subvec: ident, $subrotvec: ident) => (
        impl<N: BaseFloat> $t<N> {
            /// Creates a new similarity transformation from a vector, an axis-angle rotation, and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new(translation: $subvec<N>, rotation: $subrotvec<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: $iso::new(translation, rotation)
                }
            }

            /// Creates a new similarity transformation from a rotation matrix, a vector, and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new_with_rotmat(translation: $subvec<N>, rotation: $rotmat<N>, scale: N) -> $t<N> {
                assert!(!scale.is_zero(), "A similarity transformation scale factor cannot be zero.");

                $t {
                    scale:    scale,
                    isometry: $iso::new_with_rotmat(translation, rotation)
                }
            }

            /// Creates a new similarity transformation from an isometry and a scale factor.
            ///
            /// The scale factor may be negative but not zero.
            #[inline]
            pub fn new_with_iso(isometry: $iso<N>, scale: N) -> $t<N> {
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
            pub fn inv_scale(&self) -> N {
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
                $t::new_with_rotmat(self.isometry.translation * *s, self.isometry.rotation, self.scale * *s)
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
                $t::new_with_iso(self.isometry, self.scale * *s)
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
                $t::new_with_iso(::one(), ::one())
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
                $t::new_with_rotmat(
                    self.isometry.translation + self.isometry.rotation * (right.isometry.translation * self.scale),
                    self.isometry.rotation * right.isometry.rotation,
                    self.scale * right.scale)
            }
        }
    )
);

macro_rules! sim_mul_iso_impl(
    ($t: ident, $ti: ident) => (
        impl<N: BaseFloat> Mul<$ti<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $ti<N>) -> $t<N> {
                $t::new_with_rotmat(
                    self.isometry.translation + self.isometry.rotation * (right.translation * self.scale),
                    self.isometry.rotation * right.rotation,
                    self.scale)
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $ti<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotmat(
                    self.translation + self.rotation * right.isometry.translation,
                    self.rotation * right.isometry.rotation,
                    right.scale)
            }
        }
    )
);

macro_rules! sim_mul_rot_impl(
    ($t: ident, $tr: ident) => (
        impl<N: BaseFloat> Mul<$tr<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $tr<N>) -> $t<N> {
                $t::new_with_rotmat(
                    self.isometry.translation,
                    self.isometry.rotation * right,
                    self.scale)
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $tr<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotmat(
                    self * right.isometry.translation,
                    self * right.isometry.rotation,
                    right.scale)
            }
        }
    )
);

macro_rules! sim_mul_pnt_vec_impl(
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
            fn inv_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.isometry.inv_transform(p) / self.scale
            }
        }
    )
);

macro_rules! sim_inv_impl(
    ($t: ident) => (
        impl<N: BaseNum + Neg<Output = N>> Inv for $t<N> {
            #[inline]
            fn inv_mut(&mut self) -> bool {
                self.scale = ::one::<N>() / self.scale;
                self.isometry.inv_mut();
                // We multiply (instead of dividing) by self.scale because it has already been
                // inverted.
                self.isometry.translation = self.isometry.translation * self.scale;

                // Always succeed.
                true
            }

            #[inline]
            fn inv(&self) -> Option<$t<N>> {
                let mut res = *self;
                res.inv_mut();

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
                let mut res = (*self.isometry.rotation.submat() * self.scale).to_homogeneous();

                // copy the translation
                let dim = Dim::dim(None::<$th<N>>);

                res.set_col(dim - 1, self.isometry.translation.as_pnt().to_homogeneous().to_vec());

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

                $t::new_with_iso(rng.gen(), scale)
            }
        }
    )
);

macro_rules! sim_arbitrary_impl(
    ($t: ident) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + BaseFloat> Arbitrary for $t<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t::new_with_iso(
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
                    try!(write!(f, "{:.*}", precision, *self.isometry.rotation.submat()));
                }
                else {
                    try!(writeln!(f, "... scale factor: {}", self.scale));
                    try!(writeln!(f, "... translation: {}", self.isometry.translation));
                    try!(writeln!(f, "... rotation matrix:"));
                    try!(write!(f, "{}", *self.isometry.rotation.submat()));
                }

                writeln!(f, "}}")
            }
        }
    )
);
