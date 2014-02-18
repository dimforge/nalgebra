#[macro_escape];

macro_rules! iso_impl(
    ($t: ident, $submat: ident, $subvec: ident, $subrotvec: ident) => (
        impl<N: Clone + Float + Float + Num> $t<N> {
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
)

macro_rules! rotation_matrix_impl(
    ($t: ident, $trot: ident, $tlv: ident, $tav: ident) => (
        impl<N: Cast<f32> + Float + Float + Num + Clone>
        RotationMatrix<$tlv<N>, $tav<N>, $trot<N>> for $t<N> {
            #[inline]
            fn to_rot_mat(&self) -> $trot<N> {
                self.rotation.clone()
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

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: Float + Float + Num + Clone> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new_with_rotmat(Zero::zero(), One::one())
            }
        }
    )
)

macro_rules! iso_mul_iso_impl(
    ($t: ident, $tmul: ident) => (
        impl<N: Num + Float + Float + Clone> $tmul<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new_with_rotmat(
                    left.translation + left.rotation * right.translation,
                    left.rotation * right.rotation)
            }
        }
    )
)

macro_rules! iso_mul_vec_impl(
    ($t: ident, $tv: ident, $tmul: ident) => (
        impl<N: Num + Clone> $tmul<N, $tv<N>> for $tv<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$tv<N>) -> $tv<N> {
                left.translation + left.rotation * *right
            }
        }
    )
)

macro_rules! vec_mul_iso_impl(
    ($t: ident, $tv: ident, $tmul: ident) => (
        impl<N: Clone + Num> $tmul<N, $tv<N>> for $t<N> {
            #[inline]
            fn binop(left: &$tv<N>, right: &$t<N>) -> $tv<N> {
                (left + right.translation) * right.rotation
            }
        }
    )
)

macro_rules! translation_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Float + Num + Float + Clone> Translation<$tv<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $tv<N> {
                self.translation.clone()
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
            fn append_translation_cpy(iso: &$t<N>, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotmat(*t + iso.translation, iso.rotation.clone())
            }

            #[inline]
            fn prepend_translation(&mut self, t: &$tv<N>) {
                self.translation = self.translation + self.rotation * *t
            }

            #[inline]
            fn prepend_translation_cpy(iso: &$t<N>, t: &$tv<N>) -> $t<N> {
                $t::new_with_rotmat(iso.translation + iso.rotation * *t, iso.rotation.clone())
            }

            #[inline]
            fn set_translation(&mut self, t: $tv<N>) {
                self.translation = t
            }
        }
    )
)

macro_rules! translate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Clone + Add<N, N> + Sub<N, N>> Translate<$tv<N>> for $t<N> {
            #[inline]
            fn translate(&self, v: &$tv<N>) -> $tv<N> {
                v + self.translation
            }

            #[inline]
            fn inv_translate(&self, v: &$tv<N>) -> $tv<N> {
                v - self.translation
            }
        }
    )
)

macro_rules! rotation_impl(
    ($t: ident, $trot: ident, $tav: ident) => (
        impl<N: Cast<f32> + Num + Float + Float + Clone> Rotation<$tav<N>> for $t<N> {
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
                let delta = $trot::new(rot.clone());

                self.rotation    = delta * self.rotation;
                self.translation = delta * self.translation;
            }

            #[inline]
            fn append_rotation_cpy(t: &$t<N>, rot: &$tav<N>) -> $t<N> {
                let delta = $trot::new(rot.clone());

                $t::new_with_rotmat(delta * t.translation, delta * t.rotation)
            }

            #[inline]
            fn prepend_rotation(&mut self, rot: &$tav<N>) {
                let delta = $trot::new(rot.clone());

                self.rotation    = self.rotation * delta;
            }

            #[inline]
            fn prepend_rotation_cpy(t: &$t<N>, rot: &$tav<N>) -> $t<N> {
                let delta = $trot::new(rot.clone());

                $t::new_with_rotmat(t.translation.clone(), t.rotation * delta)
            }

            #[inline]
            fn set_rotation(&mut self, rot: $tav<N>) {
                // FIXME: should the translation be changed too?
                self.rotation.set_rotation(rot)
            }
        }
    )
)

macro_rules! rotate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Num + Clone> Rotate<$tv<N>> for $t<N> {
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
)

macro_rules! transformation_impl(
    ($t: ident) => (
        impl<N: Num + Float + Float + Clone> Transformation<$t<N>> for $t<N> {
            fn transformation(&self) -> $t<N> {
                self.clone()
            }

            fn inv_transformation(&self) -> $t<N> {
                // inversion will never fails
                Inv::inv_cpy(self).unwrap()
            }

            fn append_transformation(&mut self, t: &$t<N>) {
                *self = *t * *self
            }

            fn append_transformation_cpy(iso: &$t<N>, t: &$t<N>) -> $t<N> {
                t * *iso
            }

            fn prepend_transformation(&mut self, t: &$t<N>) {
                *self = *self * *t
            }

            fn prepend_transformation_cpy(iso: &$t<N>, t: &$t<N>) -> $t<N> {
                *iso * *t
            }

            fn set_transformation(&mut self, t: $t<N>) {
                *self = t
            }
        }
    )
)

macro_rules! transform_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Num + Clone> Transform<$tv<N>> for $t<N> {
            #[inline]
            fn transform(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.transform(v) + self.translation
            }

            #[inline]
            fn inv_transform(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.inv_transform(&(v - self.translation))
            }
        }
    )
)

macro_rules! inv_impl(
    ($t: ident) => (
        impl<N: Clone + Num> Inv for $t<N> {
            #[inline]
            fn inv(&mut self) -> bool {
                self.rotation.inv();
                self.translation = self.rotation * -self.translation;

                // always succeed
                true
            }

            #[inline]
            fn inv_cpy(m: &$t<N>) -> Option<$t<N>> {
                let mut res = m.clone();

                res.inv();

                // always succeed
                Some(res)
            }
        }
    )
)

macro_rules! to_homogeneous_impl(
    ($t: ident, $th: ident) => (
        impl<N: Num + Clone> ToHomogeneous<$th<N>> for $t<N> {
            fn to_homogeneous(m: &$t<N>) -> $th<N> {
                let mut res = ToHomogeneous::to_homogeneous(&m.rotation);

                // copy the translation
                let dim = Dim::dim(None::<$th<N>>);

                res.set_col(dim - 1, ToHomogeneous::to_homogeneous(&m.translation));

                res
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
                ApproxEq::approx_eq(&a.rotation, &b.rotation) &&
                    ApproxEq::approx_eq(&a.translation, &b.translation)
            }

            #[inline]
            fn approx_eq_eps(a: &$t<N>, b: &$t<N>, epsilon: &N) -> bool {
                ApproxEq::approx_eq_eps(&a.rotation, &b.rotation, epsilon) &&
                    ApproxEq::approx_eq_eps(&a.translation, &b.translation, epsilon)
            }
        }
    )
)

macro_rules! rand_impl(
    ($t: ident) => (
        impl<N: Rand + Clone + Float + Float + Num> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                $t::new(rng.gen(), rng.gen())
            }
        }
    )
)

macro_rules! absolute_rotate_impl(
    ($t: ident, $tv: ident) => (
        impl<N: Signed> AbsoluteRotate<$tv<N>> for $t<N> {
            #[inline]
            fn absolute_rotate(&self, v: &$tv<N>) -> $tv<N> {
                self.rotation.absolute_rotate(v)
            }
        }
    )
)
