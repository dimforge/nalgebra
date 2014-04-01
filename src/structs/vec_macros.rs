#![macro_escape]

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

macro_rules! at_fast_impl(
    ($t: ident, $dim: expr) => (
        impl<N: Clone> $t<N> {
            /// Unsafe read access to a vector element by index.
            #[inline]
            pub unsafe fn at_fast(&self, i: uint) -> N {
                (*cast::transmute::<&$t<N>, &[N, ..$dim]>(self)
                 .unsafe_ref(i)).clone()
            }

            /// Unsafe write access to a vector element by index.
            #[inline]
            pub unsafe fn set_fast(&mut self, i: uint, val: N) {
                (*cast::transmute::<&mut $t<N>, &mut [N, ..$dim]>(self).unsafe_mut_ref(i)) = val
            }
        }
    )
)

// FIXME: N should be bounded by TotalOrd instead of Float…
// However, f32/f64 does not implement TotalOrd…
macro_rules! ord_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Float + Eq + Clone> PartialOrd for $t<N> {
            #[inline]
            fn inf(a: &$t<N>, b: &$t<N>) -> $t<N> {
                $t::new(a.$comp0.min(b.$comp0.clone())
                        $(, a.$compN.min(b.$compN))*)
            }

            #[inline]
            fn sup(a: &$t<N>, b: &$t<N>) -> $t<N> {
                $t::new(a.$comp0.max(b.$comp0.clone())
                        $(, a.$compN.max(b.$compN.clone()))*)
            }

            #[inline]
            #[allow(unused_mut)] // otherwise there will be a warning for is_eq or Vec1.
            fn partial_cmp(a: &$t<N>, b: &$t<N>) -> PartialOrdering {
                let is_lt     = a.$comp0 <  b.$comp0;
                let mut is_eq = a.$comp0 == b.$comp0;

                if is_lt { // <
                    $(
                        if a.$compN > b.$compN {
                            return NotComparable
                        }
                     )*

                    Less
                }
                else { // >=
                    $(
                        if a.$compN < b.$compN {
                            return NotComparable
                        }
                        else if a.$compN > b.$compN {
                            is_eq = false;
                        }

                     )*

                    if is_eq {
                        Equal
                    }
                    else {
                        Greater
                    }
                }
            }

            #[inline]
            fn partial_lt(a: &$t<N>, b: &$t<N>) -> bool {
                a.$comp0 < b.$comp0 $(&& a.$compN < b.$compN)*
            }

            #[inline]
            fn partial_le(a: &$t<N>, b: &$t<N>) -> bool {
                a.$comp0 <= b.$comp0 $(&& a.$compN <= b.$compN)*
            }

            #[inline]
            fn partial_gt(a: &$t<N>, b: &$t<N>) -> bool {
                a.$comp0 > b.$comp0 $(&& a.$compN > b.$compN)*
            }

            #[inline]
            fn partial_ge(a: &$t<N>, b: &$t<N>) -> bool {
                a.$comp0 >= b.$comp0 $(&& a.$compN >= b.$compN)*
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
    ($t: ident, $tcast: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<Nin: Clone, Nout: Clone + Cast<Nin>> $tcast<Nout> for $t<Nin> {
            #[inline]
            fn to(v: $t<Nin>) -> $t<Nout> {
                $t::new(Cast::from(v.$comp0.clone()) $(, Cast::from(v.$compN.clone()))*)
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

            #[inline]
            unsafe fn unsafe_at(&self, i: uint) -> N {
                (*cast::transmute::<&$t<N>, &[N, ..$dim]>(self).unsafe_ref(i)).clone()
            }

            #[inline]
            unsafe fn unsafe_set(&mut self, i: uint, val: N) {
                (*cast::transmute::<&mut $t<N>, &mut [N, ..$dim]>(self).unsafe_mut_ref(i)) = val
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
            fn iter<'l>(&'l self) -> Items<'l, N> {
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
            fn mut_iter<'l>(&'l mut self) -> MutItems<'l, N> {
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

macro_rules! container_impl(
    ($t: ident) => (
        impl<N> Container for $t<N> {
            #[inline]
            fn len(&self) -> uint {
                Dim::dim(None::<$t<N>>)
            }
        }
    )
)

macro_rules! basis_impl(
    ($t: ident, $trhs: ident, $dim: expr) => (
        impl<N: Clone + Num + Float + ApproxEq<N> + $trhs<N, $t<N>>> Basis for $t<N> {
            #[inline]
            fn canonical_basis(f: |$t<N>| -> bool) {
                for i in range(0u, $dim) {
                    let mut basis_element : $t<N> = Zero::zero();

                    unsafe {
                        basis_element.set_fast(i, One::one());
                    }

                    if !f(basis_element) { return }
                }
            }

            #[inline]
            fn orthonormal_subspace_basis(n: &$t<N>, f: |$t<N>| -> bool) {
                // compute the basis of the orthogonal subspace using Gram-Schmidt
                // orthogonalization algorithm
                let mut basis: Vec<$t<N>> = Vec::new();

                for i in range(0u, $dim) {
                    let mut basis_element : $t<N> = Zero::zero();

                    unsafe {
                        basis_element.set_fast(i, One::one());
                    }

                    if basis.len() == $dim - 1 {
                        break;
                    }

                    let mut elt = basis_element.clone();

                    elt = elt - *n * Dot::dot(&basis_element, n);

                    for v in basis.iter() {
                        elt = elt - v * Dot::dot(&elt, v)
                    };

                    if !ApproxEq::approx_eq(&Norm::sqnorm(&elt), &Zero::zero()) {
                        let new_element = Norm::normalize_cpy(&elt);

                        if !f(new_element.clone()) { return };

                        basis.push(new_element);
                    }
                }
            }
        }
    )
)

macro_rules! add_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Add<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 + right.$comp0 $(, left.$compN + right.$compN)*)
            }
        }
    )
)

macro_rules! sub_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Sub<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 - right.$comp0 $(, left.$compN - right.$compN)*)
            }
        }
    )
)

macro_rules! mul_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Mul<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 * right.$comp0 $(, left.$compN * right.$compN)*)
            }
        }
    )
)

macro_rules! div_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Div<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 / right.$comp0 $(, left.$compN / right.$compN)*)
            }
        }
    )
)

macro_rules! neg_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Neg<N>> Neg<$t<N>> for $t<N> {
            #[inline]
            fn neg(&self) -> $t<N> {
                $t::new(-self.$comp0 $(, -self.$compN )*)
            }
        }
    )
)

macro_rules! dot_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Num> Dot<N> for $t<N> {
            #[inline]
            fn dot(a: &$t<N>, b: &$t<N>) -> N {
                a.$comp0 * b.$comp0 $(+ a.$compN * b.$compN )*
            }

            #[inline]
            fn sub_dot(a: &$t<N>, b: &$t<N>, c: &$t<N>) -> N {
                (a.$comp0 - b.$comp0) * c.$comp0 $(+ (a.$compN - b.$compN) * c.$compN )*
            }
        }
    )
)

macro_rules! scalar_mul_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 * *right $(, left.$compN * *right)*)
            }
        }
    )
)

macro_rules! scalar_div_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 / *right $(, left.$compN / *right)*)
            }
        }
    )
)

macro_rules! scalar_add_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 + *right $(, left.$compN + *right)*)
            }
        }
    )
)

macro_rules! scalar_sub_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 - *right $(, left.$compN - *right)*)
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
            fn append_translation(&mut self, t: &$t<N>) {
                *self = *t + *self;
            }

            #[inline]
            fn append_translation_cpy(transform: &$t<N>, t: &$t<N>) -> $t<N> {
                *t + *transform
            }

            #[inline]
            fn prepend_translation(&mut self, t: &$t<N>) {
                *self = *self + *t;
            }

            #[inline]
            fn prepend_translation_cpy(transform: &$t<N>, t: &$t<N>) -> $t<N> {
                transform + *t
            }

            #[inline]
            fn set_translation(&mut self, t: $t<N>) {
                *self = t
            }
        }
    )
)

macro_rules! norm_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + Num + Float> Norm<N> for $t<N> {
            #[inline]
            fn sqnorm(v: &$t<N>) -> N {
                Dot::dot(v, v)
            }

            #[inline]
            fn norm(v: &$t<N>) -> N {
                Norm::sqnorm(v).sqrt()
            }

            #[inline]
            fn normalize_cpy(v: &$t<N>) -> $t<N> {
                let mut res : $t<N> = v.clone();

                let _ = res.normalize();

                res
            }

            #[inline]
            fn normalize(&mut self) -> N {
                let l = Norm::norm(self);

                self.$comp0 = self.$comp0 / l;
                $(self.$compN = self.$compN / l;)*

                l
            }
        }
    )
)

macro_rules! round_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Round> Round for $t<N> {
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
            fn approx_epsilon(_: Option<$t<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_eq(a: &$t<N>, b: &$t<N>) -> bool {
                ApproxEq::approx_eq(&a.$comp0, &b.$comp0)
                $(&& ApproxEq::approx_eq(&a.$compN, &b.$compN))*
            }

            #[inline]
            fn approx_eq_eps(a: &$t<N>, b: &$t<N>, eps: &N) -> bool {
                ApproxEq::approx_eq_eps(&a.$comp0, &b.$comp0, eps)
                $(&& ApproxEq::approx_eq_eps(&a.$compN, &b.$compN, eps))*
            }
        }
    )
)

macro_rules! one_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: One> One for $t<N> {
            #[inline]
            fn one() -> $t<N> {
                $t {
                    $comp0: One::one()
                    $(, $compN: One::one() )*
                }
            }
        }
    )
)

macro_rules! from_iterator_impl(
    ($t: ident, $param0: ident $(, $paramN: ident)*) => (
        impl<N> FromIterator<N> for $t<N> {
            #[inline]
            fn from_iter<I: Iterator<N>>(mut $param0: I) -> $t<N> {
                $t::new($param0.next().unwrap() $(, $paramN.next().unwrap())*)
            }
        }
    )
)

macro_rules! bounded_impl(
    ($t: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Bounded> Bounded for $t<N> {
            #[inline]
            fn max_value() -> $t<N> {
                $t {
                    $comp0: Bounded::max_value()
                    $(, $compN: Bounded::max_value() )*
                }
            }

            #[inline]
            fn min_value() -> $t<N> {
                $t {
                    $comp0: Bounded::min_value()
                    $(, $compN: Bounded::min_value() )*
                }
            }
        }
    )
)

macro_rules! to_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Clone + One + Zero> ToHomogeneous<$t2<N>> for $t<N> {
            fn to_homogeneous(v: &$t<N>) -> $t2<N> {
                let mut res: $t2<N> = One::one();

                res.$comp0    = v.$comp0.clone();
                $( res.$compN = v.$compN.clone(); )*

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

                res.$comp0    = v.$comp0.clone() / v.$extra;
                $( res.$compN = v.$compN.clone() / v.$extra; )*

                res
            }
        }
    )
)

macro_rules! translate_impl(
    ($t: ident) => (
        impl<N: Add<N, N> + Sub<N, N>> Translate<$t<N>> for $t<N> {
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
