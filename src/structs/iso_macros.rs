#![macro_escape]

macro_rules! iso_impl(
    ($t: ident, $submat: ident, $subvec: ident, $subrotvec: ident) => (
        impl<N: BaseFloat> $t<N> {
            /// Creates a new isometry from a rotation matrix and a vector.
            #[inline]
            pub fn new(translation: $subvec<N>, rotation: $subrotvec<N>) -> $t<N> {
                $t {
                    rotation:    $submat::new(rotation),
                    translation: translation
                }
            }

            /// Creates a new isometry from a rotation matrix and a vector.
            #[inline]
            pub fn new_with_rotmat(translation: $subvec<N>, rotation: $submat<N>) -> $t<N> {
                $t {
                    rotation:    rotation,
                    translation: translation
                }
            }
        }
    )
);

macro_rules! rotation_matrix_impl(
    ($t: ident, $trot: ident, $tlv: ident, $tav: ident) => (
        impl<N: Cast<f64> + BaseFloat>
        RotationMatrix<N, $tlv<N>, $tav<N>, $trot<N>> for $t<N> {
            #[inline]
            fn to_rot_mat(&self) -> $trot<N> {
                self.rotation
            }
        }
    )
);


macro_rules! dim_impl(
    ($t: ident, $dim: expr) => (
        impl<N> Dim for $t<N> {
            #[inline]
            fn dim(_: Option<$t<N>>) -> uint {
                $dim
            }
        }
    )
);

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: BaseFloat> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new_with_rotmat(::zero(), ::one())
            }
        }
    )
);

macro_rules! iso_mul_iso_impl(
    ($t: ident) => (
        impl<N: BaseFloat> Mul<$t<N>, $t<N>> for $t<N> {
            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new_with_rotmat(
                    self.translation + self.rotation * right.translation,
                    self.rotation * right.rotation)
            }
        }
    )
);

macro_rules! iso_mul_pnt_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$tv<N>, $tv<N>> for $t<N> {
            #[inline]
            fn mul(self, right: $tv<N>) -> $tv<N> {
                self.rotation * right + self.translation
            }
        }
    )
);

macro_rules! pnt_mul_iso_impl(
    ($t: ident, $tv: ident) => (
        impl<N: BaseNum> Mul<$t<N>, $tv<N>> for $tv<N> {
            #[inline]
            fn mul(self, right: $t<N>) -> $tv<N> {
                (self + right.translation) * right.rotation
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
            fn inv_translation(&self) -> $tv<N> {
                -self.translation
            }

            #[inline]
            fn append_translation(&mut self, t: &$tv<N>) {
                self.translation = *t + self.translation
            }

            #[inline]
            fn append_translation_cpy(&self, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotmat(*t + self.translation, self.rotation)
            }

            #[inline]
            fn prepend_translation(&mut self, t: &$tv<N>) {
                self.translation = self.translation + self.rotation * *t
            }

            #[inline]
            fn prepend_translation_cpy(&self, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotmat(self.translation + self.rotation * *t, self.rotation)
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
        impl<N: Copy + Add<N, N> + Sub<N, N>> Translate<$tv<N>> for $t<N> {
            #[inline]
            fn translate(&self, v: &$tv<N>) -> $tv<N> {
                *v + self.translation
            }

            #[inline]
            fn inv_translate(&self, v: &$tv<N>) -> $tv<N> {
                *v - self.translation
            }
        }
    )
);

macro_rules! rotation_impl(
    ($t: ident, $trot: ident, $tav: ident) => (
        impl<N: Cast<f64> + BaseFloat> Rotation<$tav<N>> for $t<N> {
            #[inline]
            fn rotation(&self) -> $tav<N> {
                self.rotation.rotation()
            }

            #[inline]
            fn inv_rotation(&self) -> $tav<N> {
                self.rotation.inv_rotation()
            }

            #[inline]
            fn append_rotation(&mut self, rot: &$tav<N>) {
                let delta = $trot::new(*rot);

                self.rotation    = delta * self.rotation;
                self.translation = delta * self.translation;
            }

            #[inline]
            fn append_rotation_cpy(&self, rot: &$tav<N>) -> $t<N> {
                let delta = $trot::new(*rot);

                $t::new_with_rotmat(delta * self.translation, delta * self.rotation)
            }

            #[inline]
            fn prepend_rotation(&mut self, rot: &$tav<N>) {
                let delta = $trot::new(*rot);

                self.rotation = self.rotation * delta;
            }

            #[inline]
            fn prepend_rotation_cpy(&self, rot: &$tav<N>) -> $t<N> {
                let delta = $trot::new(*rot);

                $t::new_with_rotmat(self.translation, self.rotation * delta)
            }

            #[inline]
            fn set_rotation(&mut self, rot: $tav<N>) {
                // FIXME: should the translation be changed too?
                self.rotation.set_rotation(rot)
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
            fn inv_rotate(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.inv_rotate(v)
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

            fn inv_transformation(&self) -> $t<N> {
                // inversion will never fails
                Inv::inv_cpy(self).unwrap()
            }

            fn append_transformation(&mut self, t: &$t<N>) {
                *self = *t * *self
            }

            fn append_transformation_cpy(&self, t: &$t<N>) -> $t<N> {
                *t * *self
            }

            fn prepend_transformation(&mut self, t: &$t<N>) {
                *self = *self * *t
            }

            fn prepend_transformation_cpy(&self, t: &$t<N>) -> $t<N> {
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
            fn inv_transform(&self, p: &$tp<N>) -> $tp<N> {
                self.rotation.inv_transform(&(*p - self.translation))
            }
        }
    )
);

macro_rules! inv_impl(
    ($t: ident) => (
        impl<N: BaseNum> Inv for $t<N> {
            #[inline]
            fn inv(&mut self) -> bool {
                self.rotation.inv();
                self.translation = self.rotation * -self.translation;
                // always succeed
                true
            }

            #[inline]
            fn inv_cpy(&self) -> Option<$t<N>> {
                let mut res = *self;
                res.inv();
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
                let dim = Dim::dim(None::<$th<N>>);

                res.set_col(dim - 1, self.translation.as_pnt().to_homogeneous().to_vec());

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
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&self.rotation, &other.rotation, epsilon) &&
                    ApproxEq::approx_eq_eps(&self.translation, &other.translation, epsilon)
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
