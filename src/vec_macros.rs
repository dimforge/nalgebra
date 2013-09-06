#[macro_escape];

macro_rules! new_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N> $t<N> {
            /// Creates a new vector.
            #[inline]
            pub fn new($comp0: N $( , $compN: N )*) -> $t<N> {
                $t {
                    $comp0: $comp0
                    $(, $compN: $compN )*
                }
            }
        }
    )
)

macro_rules! ord_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Ord> Ord for $t<N> {
            #[inline]
            fn lt(&self, other: &$t<N>) -> bool {
                self.$comp0 < other.$comp0 $(&& self.$compN < other.$compN)*
            }

            #[inline]
            fn le(&self, other: &$t<N>) -> bool {
                self.$comp0 <= other.$comp0 $(&& self.$compN <= other.$compN)*
            }

            #[inline]
            fn gt(&self, other: &$t<N>) -> bool {
                self.$comp0 > other.$comp0 $(&& self.$compN > other.$compN)*
            }

            #[inline]
            fn ge(&self, other: &$t<N>) -> bool {
                self.$comp0 >= other.$comp0 $(&& self.$compN >= other.$compN)*
            }
        }
    )
)

macro_rules! orderable_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Orderable> Orderable for $t<N> {
            #[inline]
            fn max(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.max(&other.$comp0) $(, self.$compN.max(&other.$compN))*)
            }

            #[inline]
            fn min(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.min(&other.$comp0) $(, self.$compN.min(&other.$compN))*)
            }

            #[inline]
            fn clamp(&self, min: &$t<N>, max: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.clamp(&min.$comp0, &max.$comp0)
                        $(, self.$compN.clamp(&min.$comp0, &max.$comp0))*)
            }
        }
    )
)

macro_rules! vec_axis_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Zero + One> $t<N> {
            /// Create a unit vector with its `$comp0` component equal to 1.0.
            #[inline]
            pub fn $comp0() -> $t<N> {
                let mut res: $t<N> = Zero::zero();

                res.$comp0 = One::one();

                res
            }

            $(
                /// Create a unit vector with its `$compN` component equal to 1.0.
                #[inline]
                pub fn $compN() -> $t<N> {
                    let mut res: $t<N> = Zero::zero();

                    res.$compN = One::one();

                    res
                }
            )*
        }
    )
)

macro_rules! vec_cast_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<Nin: NumCast + Clone, Nout: Clone + NumCast> VecCast<$t<Nout>> for $t<Nin> {
            #[inline]
            fn from(v: $t<Nin>) -> $t<Nout> {
                $t::new(NumCast::from(v.$comp0.clone()) $(, NumCast::from(v.$compN.clone()))*)
            }
        }
    )
)

macro_rules! indexable_impl(
    ($t: ident, $dim: expr) => (
        impl<N: Clone> Indexable<uint, N> for $t<N> {
            #[inline]
            fn at(&self, i: uint) -> N {
                unsafe {
                    cast::transmute::<&$t<N>, &[N, ..$dim]>(self)[i].clone()
                }
            }

            #[inline]
            fn set(&mut self, i: uint, val: N) {
                unsafe {
                    cast::transmute::<&mut $t<N>, &mut [N, ..$dim]>(self)[i] = val
                }
            }

            #[inline]
            fn swap(&mut self, i1: uint, i2: uint) {
                unsafe {
                    cast::transmute::<&mut $t<N>, &mut [N, ..$dim]>(self).swap(i1, i2)
                }
            }
        }
    )
)

macro_rules! new_repeat_impl(
    ($t: ident, $param: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone> $t<N> {
            /// Creates a new vector with all its components equal to a given value.
            #[inline]
            pub fn new_repeat($param: N) -> $t<N> {
                $t{
                    $comp0: $param.clone()
                    $(, $compN: $param.clone() )*
                }
            }
        }
    )
)

macro_rules! iterable_impl(
    ($t: ident, $dim: expr) => (
        impl<N> Iterable<N> for $t<N> {
            #[inline]
            fn iter<'l>(&'l self) -> VecIterator<'l, N> {
                unsafe {
                    cast::transmute::<&'l $t<N>, &'l [N, ..$dim]>(self).iter()
                }
            }
        }
    )
)

macro_rules! iterable_mut_impl(
    ($t: ident, $dim: expr) => (
        impl<N> IterableMut<N> for $t<N> {
            #[inline]
            fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N> {
                unsafe {
                    cast::transmute::<&'l mut $t<N>, &'l mut [N, ..$dim]>(self).mut_iter()
                }
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

macro_rules! basis_impl(
    ($t: ident, $dim: expr) => (
        impl<N: Clone + Num + Algebraic + ApproxEq<N>> Basis for $t<N> {
            #[inline]
            fn canonical_basis(f: &fn($t<N>) -> bool) {
                for i in range(0u, $dim) {
                    let mut basis_element : $t<N> = Zero::zero();

                    basis_element.set(i, One::one());

                    if !f(basis_element) { return }
                }
            }

            #[inline]
            fn orthonormal_subspace_basis(&self, f: &fn($t<N>) -> bool) {
                // compute the basis of the orthogonal subspace using Gram-Schmidt
                // orthogonalization algorithm
                let mut basis: ~[$t<N>] = ~[];

                for i in range(0u, $dim) {
                    let mut basis_element : $t<N> = Zero::zero();

                    basis_element.set(i, One::one());

                    if basis.len() == $dim - 1 {
                        break;
                    }

                    let mut elt = basis_element.clone();

                    elt = elt - *self * basis_element.dot(self);

                    for v in basis.iter() {
                        elt = elt - v * elt.dot(v)
                    };

                    if !elt.sqnorm().approx_eq(&Zero::zero()) {
                        let new_element = elt.normalized();

                        if !f(new_element.clone()) { return };

                        basis.push(new_element);
                    }
                }
            }
        }
    )
)

macro_rules! add_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Add<N, N>> Add<$t<N>, $t<N>> for $t<N> {
            #[inline]
            fn add(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0 + other.$comp0 $(, self.$compN + other.$compN)*)
            }
        }
    )
)

macro_rules! sub_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Sub<N, N>> Sub<$t<N>, $t<N>> for $t<N> {
            #[inline]
            fn sub(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0 - other.$comp0 $(, self.$compN - other.$compN)*)
            }
        }
    )
)

macro_rules! neg_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Neg<N>> Neg<$t<N>> for $t<N> {
            #[inline]
            fn neg(&self) -> $t<N> {
                $t::new(-self.$comp0 $(, -self.$compN )*)
            }
        }
    )
)

macro_rules! dot_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Num + Clone> Vec<N> for $t<N> {
            #[inline]
            fn dot(&self, other: &$t<N>) -> N {
                self.$comp0 * other.$comp0 $(+ self.$compN * other.$compN )*
            }

            #[inline]
            fn sub_dot(&self, a: &$t<N>, b: &$t<N>) -> N {
                (self.$comp0 - a.$comp0) * b.$comp0 $(+ (self.$compN - a.$compN) * b.$compN )*
            }
        }
    )
)

macro_rules! scalar_mul_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Mul<N, N>> Mul<N, $t<N>> for $t<N> {
            #[inline]
            fn mul(&self, s: &N) -> $t<N> {
                $t::new(self.$comp0 * *s $(, self.$compN * *s)*)
            }
        }
    )
)

macro_rules! scalar_div_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Div<N, N>> Div<N, $t<N>> for $t<N> {
            #[inline]
            fn div(&self, s: &N) -> $t<N> {
                $t::new(self.$comp0 / *s $(, self.$compN / *s)*)
            }
        }
    )
)

macro_rules! scalar_add_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Add<N, N>> ScalarAdd<N> for $t<N> {
            #[inline]
            fn scalar_add(&self, s: &N) -> $t<N> {
                $t::new(self.$comp0 + *s $(, self.$compN + *s)*)
            }

            #[inline]
            fn scalar_add_inplace(&mut self, s: &N) {
                self.$comp0   = self.$comp0 + *s;
                $(self.$compN = self.$compN + *s;)*
            }
        }
    )
)

macro_rules! scalar_sub_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Sub<N, N>> ScalarSub<N> for $t<N> {
            #[inline]
            fn scalar_sub(&self, s: &N) -> $t<N> {
                $t::new(self.$comp0 - *s $(, self.$compN - *s)*)
            }

            #[inline]
            fn scalar_sub_inplace(&mut self, s: &N) {
                self.$comp0   = self.$comp0 - *s;
                $(self.$compN = self.$compN - *s;)*
            }
        }
    )
)

macro_rules! translation_impl(
    ($t: ident) => (
        impl<N: Clone + Add<N, N> + Neg<N>> Translation<$t<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $t<N> {
                self.clone()
            }

            #[inline]
            fn inv_translation(&self) -> $t<N> {
                -self
            }

            #[inline]
            fn translate_by(&mut self, t: &$t<N>) {
                *self = *self + *t;
            }

            #[inline]
            fn translated(&self, t: &$t<N>) -> $t<N> {
                self + *t
            }

            #[inline]
            fn set_translation(&mut self, t: $t<N>) {
                *self = t
            }
        }
    )
)

macro_rules! norm_impl(
    ($t: ident) => (
        impl<N: Clone + Num + Algebraic> AlgebraicVec<N> for $t<N> {
            #[inline]
            fn sqnorm(&self) -> N {
                self.dot(self)
            }

            #[inline]
            fn norm(&self) -> N {
                self.sqnorm().sqrt()
            }

            #[inline]
            fn normalized(&self) -> $t<N> {
                let mut res : $t<N> = self.clone();

                res.normalize();

                res
            }

            #[inline]
            fn normalize(&mut self) -> N {
                let l = self.norm();

                *self = *self / l;

                l
            }
        }
    )
)

macro_rules! round_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Round> Round for $t<N> {
            fn floor(&self) -> $t<N> {
                $t::new(self.$comp0.floor() $(, self.$compN.floor())*)
            }

            fn ceil(&self) -> $t<N> {
                $t::new(self.$comp0.ceil() $(, self.$compN.ceil())*)
            }

            fn round(&self) -> $t<N> {
                $t::new(self.$comp0.round() $(, self.$compN.round())*)
            }

            fn trunc(&self) -> $t<N> {
                $t::new(self.$comp0.trunc() $(, self.$compN.trunc())*)
            }

            fn fract(&self) -> $t<N> {
                $t::new(self.$comp0.fract() $(, self.$compN.fract())*)
            }
        }
    )
)

macro_rules! approx_eq_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
            #[inline]
            fn approx_epsilon() -> N {
                fail!("approx_epsilon is broken since rust revision 8693943676487c01fa09f5f3daf0df6a1f71e24d.")
                // ApproxEq::<N>::approx_epsilon()
            }

            #[inline]
            fn approx_eq(&self, other: &$t<N>) -> bool {
                self.$comp0.approx_eq(&other.$comp0) $(&& self.$compN.approx_eq(&other.$compN))*
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, eps: &N) -> bool {
                self.$comp0.approx_eq_eps(&other.$comp0, eps) $(&& self.$compN.approx_eq_eps(&other.$compN, eps))*
            }
        }
    )
)

macro_rules! one_impl(
    ($t: ident) => (
        impl<N: Clone + One> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t::new_repeat(One::one())
            }
        }
    )
)

macro_rules! from_iterator_impl(
    ($t: ident, $param0: ident $(, $paramN: ident)*) => (
        impl<N: Clone> FromIterator<N> for $t<N> {
            #[inline]
            fn from_iterator<I: Iterator<N>>($param0: &mut I) -> $t<N> {
                $t::new($param0.next().unwrap() $(, $paramN.next().unwrap())*)
            }
        }
    )
)

macro_rules! bounded_impl(
    ($t: ident) => (
        impl<N: Bounded + Clone> Bounded for $t<N> {
            #[inline]
            fn max_value() -> $t<N> {
                $t::new_repeat(Bounded::max_value())
            }

            #[inline]
            fn min_value() -> $t<N> {
                $t::new_repeat(Bounded::min_value())
            }
        }
    )
)

macro_rules! to_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + One + Zero> ToHomogeneous<$t2<N>> for $t<N> {
            fn to_homogeneous(&self) -> $t2<N> {
                let mut res: $t2<N> = One::one();

                res.$comp0    = self.$comp0.clone();
                $( res.$compN = self.$compN.clone(); )*

                res
            }
        }
    )
)

macro_rules! from_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Div<N, N> + One + Zero> FromHomogeneous<$t2<N>> for $t<N> {
            fn from(v: &$t2<N>) -> $t<N> {
                let mut res: $t<N> = Zero::zero();

                res.$comp0    = v.$comp0.clone();
                $( res.$compN = v.$compN.clone(); )*

                res = res / v.$extra;

                res
            }
        }
    )
)

macro_rules! translate_impl(
    ($t: ident) => (
        impl<N: Clone + Add<N, N> + Sub<N, N>> Translate<$t<N>> for $t<N> {
            fn translate(&self, other: &$t<N>) -> $t<N> {
                *other + *self
            }

            fn inv_translate(&self, other: &$t<N>) -> $t<N> {
                *other - *self
            }
        }
    )
)

macro_rules! rotate_impl(
    ($t: ident) => (
        impl<N, O: Clone> Rotate<O> for $t<N> {
            fn rotate(&self, other: &O) -> O {
                other.clone()
            }

            fn inv_rotate(&self, other: &O) -> O {
                other.clone()
            }
        }
    )
)

macro_rules! transform_impl(
    ($t: ident) => (
        impl<N: Clone + Add<N, N> + Sub<N, N>> Transform<$t<N>> for $t<N> {
            fn transform(&self, other: &$t<N>) -> $t<N> {
                self.translate(other)
            }

            fn inv_transform(&self, other: &$t<N>) -> $t<N> {
                self.inv_translate(other)
            }
        }
    )
)
