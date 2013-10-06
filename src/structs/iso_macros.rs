#[macro_escape];

macro_rules! iso_impl(
    ($t: ident, $submat: ident, $subvec: ident) => (
        impl<N> $t<N> {
            /// Creates a new isometry from a rotation matrix and a vector.
            #[inline]
            pub fn new(translation: $subvec<N>, rotation: $submat<N>) -> $t<N> {
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
        impl<N: NumCast + Algebraic + Trigonometric + Num + Clone>
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
        impl<N: One + Zero + Clone> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new(Zero::zero(), One::one())
            }
        }
    )
)

macro_rules! iso_mul_iso_impl(
    ($t: ident, $tmul: ident) => (
        impl<N: Num + Clone> $tmul<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.translation + left.rotation * right.translation, left.rotation * right.rotation)
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
        impl<N: Neg<N> + Add<N, N> + Clone> Translation<$tv<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $tv<N> {
                self.translation.clone()
            }

            #[inline]
            fn inv_translation(&self) -> $tv<N> {
                -self.translation
            }

            #[inline]
            fn translate_by(&mut self, t: &$tv<N>) {
                self.translation = self.translation + *t
            }

            #[inline]
            fn translated(&self, t: &$tv<N>) -> $t<N> {
                $t::new(self.translation + *t, self.rotation.clone())
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
        impl<N: Num + Trigonometric + Algebraic + NumCast + Clone> Rotation<$tav<N>> for $t<N> {
            #[inline]
            fn rotation(&self) -> $tav<N> {
                self.rotation.rotation()
            }

            #[inline]
            fn inv_rotation(&self) -> $tav<N> {
                self.rotation.inv_rotation()
            }


            #[inline]
            fn rotate_by(&mut self, rot: &$tav<N>) {
                // FIXME: this does not seem opitmal
                let mut delta: $trot<N> = One::one();
                delta.rotate_by(rot);
                self.rotation.rotate_by(rot);
                self.translation = delta * self.translation;
            }

            #[inline]
            fn rotated(&self, rot: &$tav<N>) -> $t<N> {
                // FIXME: this does not seem opitmal
                let _1: $trot<N> = One::one();
                let delta = _1.rotated(rot);

                $t::new(delta * self.translation, self.rotation.rotated(rot))
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
        impl<N: Num + Clone> Transformation<$t<N>> for $t<N> {
            fn transformation(&self) -> $t<N> {
                self.clone()
            }

            fn inv_transformation(&self) -> $t<N> {
                // inversion will never fails
                self.inverted().unwrap()
            }

            fn transform_by(&mut self, other: &$t<N>) {
                *self = other * *self
            }

            fn transformed(&self, t: &$t<N>) -> $t<N> {
                t * *self
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
            fn invert(&mut self) -> bool {
                self.rotation.invert();
                self.translation = self.rotation  * -self.translation;

                // always succeed
                true
            }

            #[inline]
            fn inverted(&self) -> Option<$t<N>> {
                let mut res = self.clone();

                res.invert();

                // always succeed
                Some(res)
            }
        }
    )
)

macro_rules! to_homogeneous_impl(
    ($t: ident, $th: ident) => (
        impl<N: One + Zero + Clone> ToHomogeneous<$th<N>> for $t<N> {
            fn to_homogeneous(&self) -> $th<N> {
                let mut res = self.rotation.to_homogeneous();

                // copy the translation
                let dim = Dim::dim(None::<$th<N>>);

                res.set_col(dim - 1, self.translation.to_homogeneous());

                res
            }
        }
    )
)

macro_rules! approx_eq_impl(
    ($t: ident) => (
        impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
            #[inline]
            fn approx_epsilon() -> N {
                fail!("approx_epsilon is broken since rust revision 8693943676487c01fa09f5f3daf0df6a1f71e24d.")
                // ApproxEq::<N>::approx_epsilon()
            }

            #[inline]
            fn approx_eq(&self, other: &$t<N>) -> bool {
                self.rotation.approx_eq(&other.rotation) &&
                    self.translation.approx_eq(&other.translation)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                self.rotation.approx_eq_eps(&other.rotation, epsilon) &&
                    self.translation.approx_eq_eps(&other.translation, epsilon)
            }
        }
    )
)

macro_rules! rand_impl(
    ($t: ident) => (
        impl<N: Rand + Clone + Trigonometric + Algebraic + Num> Rand for $t<N> {
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
