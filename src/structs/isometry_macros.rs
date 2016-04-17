#![macro_use]

macro_rules! isometry_impl(
    ($t: ident, $submatrix: ident, $subvector: ident, $subrotvector: ident) => (
        impl<N: BaseFloat> $t<N> {
            /// Creates a new isometry from an axis-angle rotation, and a vector.
            #[inline]
            pub fn new(translation: $subvector<N>, rotation: $subrotvector<N>) -> $t<N> {
                $t {
                    rotation:    $submatrix::new(rotation),
                    translation: translation
                }
            }

            /// Creates a new isometry from a rotation matrix and a vector.
            #[inline]
            pub fn new_with_rotation_matrix(translation: $subvector<N>, rotation: $submatrix<N>) -> $t<N> {
                $t {
                    rotation:    rotation,
                    translation: translation
                }
            }
        }
    )
);

macro_rules! rotation_matrix_impl(
    ($t: ident, $trotation: ident, $tlv: ident, $tav: ident) => (
        impl<N: Cast<f64> + BaseFloat>
        RotationMatrix<N, $tlv<N>, $tav<N>> for $t<N> {
            type Output = $trotation<N>;

            #[inline]
            fn to_rotation_matrix(&self) -> $trotation<N> {
                self.rotation
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

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: BaseFloat> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new_with_rotation_matrix(::zero(), ::one())
            }
        }
    )
);

macro_rules! isometry_mul_isometry_impl(
    ($t: ident) => (
        impl<N: BaseFloat> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
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
    )
);

macro_rules! isometry_mul_rotation_impl(
    ($t: ident, $rotation: ident) => (
        impl<N: BaseFloat> Mul<$rotation<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $rotation<N>) -> $t<N> {
                $t::new_with_rotation_matrix(self.translation, self.rotation * right)
            }
        }

        impl<N: BaseFloat> Mul<$t<N>> for $rotation<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotation_matrix(
                    self * right.translation,
                    self * right.rotation)
            }
        }

        impl<N: BaseFloat> MulAssign<$rotation<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $rotation<N>) {
                self.rotation *= right
            }
        }
    )
);

macro_rules! isometry_mul_point_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.rotation * right + self.translation
            }
        }
    )
);

macro_rules! isometry_mul_vec_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>> for $t<N> {
            type Output = $tv<N>;

            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.rotation * right
            }
        }
    )
);

macro_rules! translation_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseFloat> Translation<$tv<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $tv<N> {
                self.translation
            }

            #[inline]
            fn inverse_translation(&self) -> $tv<N> {
                -self.translation
            }

            #[inline]
            fn append_translation_mut(&mut self, t: &$tv<N>) {
                self.translation = *t + self.translation
            }

            #[inline]
            fn append_translation(&self, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotation_matrix(*t + self.translation, self.rotation)
            }

            #[inline]
            fn prepend_translation_mut(&mut self, t: &$tv<N>) {
                self.translation = self.translation + self.rotation * *t
            }

            #[inline]
            fn prepend_translation(&self, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotation_matrix(self.translation + self.rotation * *t, self.rotation)
            }

            #[inline]
            fn set_translation(&mut self, t: $tv<N>) {
                self.translation = t
            }
        }
    )
);

macro_rules! translate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Translate<$tv<N>> for $t<N> {
            #[inline]
            fn translate(&self, v: &$tv<N>) -> $tv<N> {
                *v + self.translation
            }

            #[inline]
            fn inverse_translate(&self, v: &$tv<N>) -> $tv<N> {
                *v - self.translation
            }
        }
    )
);

macro_rules! rotation_impl(
    ($t: ident, $trotation: ident, $tav: ident) => (
        impl<N: Cast<f64> + BaseFloat> Rotation<$tav<N>> for $t<N> {
            #[inline]
            fn rotation(&self) -> $tav<N> {
                self.rotation.rotation()
            }

            #[inline]
            fn inverse_rotation(&self) -> $tav<N> {
                self.rotation.inverse_rotation()
            }

            #[inline]
            fn append_rotation_mut(&mut self, rotation: &$tav<N>) {
                let delta = $trotation::new(*rotation);

                self.rotation    = delta * self.rotation;
                self.translation = delta * self.translation;
            }

            #[inline]
            fn append_rotation(&self, rotation: &$tav<N>) -> $t<N> {
                let delta = $trotation::new(*rotation);

                $t::new_with_rotation_matrix(delta * self.translation, delta * self.rotation)
            }

            #[inline]
            fn prepend_rotation_mut(&mut self, rotation: &$tav<N>) {
                let delta = $trotation::new(*rotation);

                self.rotation = self.rotation * delta;
            }

            #[inline]
            fn prepend_rotation(&self, rotation: &$tav<N>) -> $t<N> {
                let delta = $trotation::new(*rotation);

                $t::new_with_rotation_matrix(self.translation, self.rotation * delta)
            }

            #[inline]
            fn set_rotation(&mut self, rotation: $tav<N>) {
                // FIXME: should the translation be changed too?
                self.rotation.set_rotation(rotation)
            }
        }
    )
);

macro_rules! rotate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Rotate<$tv<N>> for $t<N> {
            #[inline]
            fn rotate(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.rotate(v)
            }

            #[inline]
            fn inverse_rotate(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.inverse_rotate(v)
            }
        }
    )
);

macro_rules! transformation_impl(
    ($t: ident) => (
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
    )
);

macro_rules! transform_impl(
    ($t: ident, $tp: ident) => (
        impl<N: BaseNum> Transform<$tp<N>> for $t<N> {
            #[inline]
            fn transform(&self, p: &$tp<N>) -> $tp<N> {
                self.rotation.transform(p) + self.translation
            }

            #[inline]
            fn inverse_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.rotation.inverse_transform(&(*p - self.translation))
            }
        }
    )
);

macro_rules! inverse_impl(
    ($t: ident) => (
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
    )
);

macro_rules! to_homogeneous_impl(
    ($t: ident, $th: ident) => (
        impl<N: BaseNum> ToHomogeneous<$th<N>> for $t<N> {
            fn to_homogeneous(&self) -> $th<N> {
                let mut res = self.rotation.to_homogeneous();

                // copy the translation
                let dimension = Dimension::dimension(None::<$th<N>>);

                res.set_col(dimension - 1, self.translation.as_point().to_homogeneous().to_vector());

                res
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
    )
);

macro_rules! rand_impl(
    ($t: ident) => (
        impl<N: Rand + BaseFloat> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                $t::new(rng.gen(), rng.gen())
            }
        }
    )
);

macro_rules! absolute_rotate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseFloat> AbsoluteRotate<$tv<N>> for $t<N> {
            #[inline]
            fn absolute_rotate(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.absolute_rotate(v)
            }
        }
    )
);

macro_rules! arbitrary_isometry_impl(
    ($t: ident) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + BaseFloat> Arbitrary for $t<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t::new_with_rotation_matrix(
                    Arbitrary::arbitrary(g),
                    Arbitrary::arbitrary(g)
                )
            }
        }
    )
);

macro_rules! isometry_display_impl(
    ($t: ident) => (
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
