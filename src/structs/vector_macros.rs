#![macro_use]

macro_rules! new_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N> $t<N> {
            /// Creates a new vector.
            #[inline]
            pub fn new($($compN: N ),+) -> $t<N> {
                $t {
                    $($compN: $compN ),+
                }
            }
        }
    );
);

macro_rules! conversion_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> AsRef<[N; $dimension]> for $t<N> {
            #[inline]
            fn as_ref(&self) -> &[N; $dimension] {
                unsafe {
                    mem::transmute(self)
                }
            }
        }

        impl<N> AsMut<[N; $dimension]> for $t<N> {
            #[inline]
            fn as_mut(&mut self) -> &mut [N; $dimension] {
                unsafe {
                    mem::transmute(self)
                }
            }
        }

        impl<'a, N> From<&'a [N; $dimension]> for &'a $t<N> {
            #[inline]
            fn from(arr: &'a [N; $dimension]) -> &'a $t<N> {
                unsafe {
                    mem::transmute(arr)
                }
            }
        }

        impl<'a, N> From<&'a mut [N; $dimension]> for &'a mut $t<N> {
            #[inline]
            fn from(arr: &'a mut [N; $dimension]) -> &'a mut $t<N> {
                unsafe {
                    mem::transmute(arr)
                }
            }
        }
    )
);

macro_rules! at_fast_impl(
    ($t: ident, $dimension: expr) => (
        impl<N: Copy> $t<N> {
            /// Unsafe read access to a vector element by index.
            #[inline]
            pub unsafe fn at_fast(&self, i: usize) -> N {
                (*self.as_ref().get_unchecked(i))
            }

            /// Unsafe write access to a vector element by index.
            #[inline]
            pub unsafe fn set_fast(&mut self, i: usize, val: N) {
                (*self.as_mut().get_unchecked_mut(i)) = val
            }
        }
    )
);

// FIXME: N should be bounded by Ord instead of BaseFloat…
// However, f32/f64 does not implement Ord…
macro_rules! pord_impl(
    ($t: ident, $comp0: ident, $($compN: ident),*) => (
        impl<N: BaseFloat> PartialOrder for $t<N> {
            #[inline]
            fn inf(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.min(other.$comp0)
                        $(, self.$compN.min(other.$compN))*)
            }

            #[inline]
            fn sup(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.max(other.$comp0)
                        $(, self.$compN.max(other.$compN))*)
            }

            #[inline]
            #[allow(unused_mut)] // otherwise there will be a warning for is_eq or Vector1.
            fn partial_cmp(&self, other: &$t<N>) -> PartialOrdering {
                let is_lt     = self.$comp0 <  other.$comp0;
                let mut is_eq = self.$comp0 == other.$comp0;

                if is_lt { // <
                    $(
                        if self.$compN > other.$compN {
                            return PartialOrdering::NotComparable
                        }
                     )*

                    PartialOrdering::PartialLess
                }
                else { // >=
                    $(
                        if self.$compN < other.$compN {
                            return PartialOrdering::NotComparable
                        }
                        else if self.$compN > other.$compN {
                            is_eq = false;
                        }

                     )*

                    if is_eq {
                        PartialOrdering::PartialEqual
                    }
                    else {
                        PartialOrdering::PartialGreater
                    }
                }
            }

            #[inline]
            fn partial_lt(&self, other: &$t<N>) -> bool {
                self.$comp0 < other.$comp0 $(&& self.$compN < other.$compN)*
            }

            #[inline]
            fn partial_le(&self, other: &$t<N>) -> bool {
                self.$comp0 <= other.$comp0 $(&& self.$compN <= other.$compN)*
            }

            #[inline]
            fn partial_gt(&self, other: &$t<N>) -> bool {
                self.$comp0 > other.$comp0 $(&& self.$compN > other.$compN)*
            }

            #[inline]
            fn partial_ge(&self, other: &$t<N>) -> bool {
                self.$comp0 >= other.$comp0 $(&& self.$compN >= other.$compN)*
            }
        }
    )
);

macro_rules! vec_axis_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Zero + One> $t<N> {
            $(
                /// Create a unit vector with its `$compN` component equal to 1.0.
                #[inline]
                pub fn $compN() -> $t<N> {
                    let mut res: $t<N> = ::zero();

                    res.$compN = ::one();

                    res
                }
            )+
        }
    )
);

macro_rules! vec_cast_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<Nin: Copy, Nout: Copy + Cast<Nin>> Cast<$t<Nin>> for $t<Nout> {
            #[inline]
            fn from(v: $t<Nin>) -> $t<Nout> {
                $t::new($(Cast::from(v.$compN)),+)
            }
        }
    )
);

macro_rules! indexable_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> Shape<usize> for $t<N> {
            #[inline]
            fn shape(&self) -> usize {
                $dimension
            }
        }

        impl<N: Copy> Indexable<usize, N> for $t<N> {
            #[inline]
            fn swap(&mut self, i1: usize, i2: usize) {
                unsafe {
                    mem::transmute::<&mut $t<N>, &mut [N; $dimension]>(self).swap(i1, i2)
                }
            }

            #[inline]
            unsafe fn unsafe_at(&self, i: usize) -> N {
                (*mem::transmute::<&$t<N>, &[N; $dimension]>(self).get_unchecked(i))
            }

            #[inline]
            unsafe fn unsafe_set(&mut self, i: usize, val: N) {
                (*mem::transmute::<&mut $t<N>, &mut [N; $dimension]>(self).get_unchecked_mut(i)) = val
            }
        }
    )
);

macro_rules! index_impl(
    ($t: ident) => (
        impl<N, T> Index<T> for $t<N> where [N]: Index<T> {
            type Output = <[N] as Index<T>>::Output;

            fn index(&self, i: T) -> &<[N] as Index<T>>::Output {
                &self.as_ref()[i]
            }
        }

        impl<N, T> IndexMut<T> for $t<N> where [N]: IndexMut<T> {
            fn index_mut(&mut self, i: T) -> &mut <[N] as Index<T>>::Output {
                &mut self.as_mut()[i]
            }
        }
    )
);

macro_rules! repeat_impl(
    ($t: ident, $param: ident, $($compN: ident),+) => (
        impl<N: Copy> Repeat<N> for $t<N> {
            /// Creates a new vector with all its components equal to a given value.
            #[inline]
            fn repeat($param: N) -> $t<N> {
                $t{
                    $($compN: $param ),+
                }
            }
        }
    )
);

macro_rules! iterable_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> Iterable<N> for $t<N> {
            #[inline]
            fn iter<'l>(&'l self) -> Iter<'l, N> {
                unsafe {
                    mem::transmute::<&'l $t<N>, &'l [N; $dimension]>(self).iter()
                }
            }
        }
    )
);

macro_rules! iterable_mut_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> IterableMut<N> for $t<N> {
            #[inline]
            fn iter_mut<'l>(&'l mut self) -> IterMut<'l, N> {
                unsafe {
                    mem::transmute::<&'l mut $t<N>, &'l mut [N; $dimension]>(self).iter_mut()
                }
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

macro_rules! container_impl(
    ($t: ident) => (
        impl<N> $t<N> {
            /// The dimension of this entity.
            #[inline]
            pub fn len(&self) -> usize {
                Dimension::dimension(None::<$t<N>>)
            }
        }
    )
);

macro_rules! basis_impl(
    ($t: ident, $dimension: expr) => (
        impl<N: BaseFloat + ApproxEq<N>> Basis for $t<N> {
            #[inline]
            fn canonical_basis<F: FnMut($t<N>) -> bool>(mut f: F) {
                for i in 0 .. $dimension {
                    if !f(Basis::canonical_basis_element(i).unwrap()) { return }
                }
            }

            #[inline]
            fn orthonormal_subspace_basis<F: FnMut($t<N>) -> bool>(n: &$t<N>, mut f: F) {
                // Compute the basis of the orthogonal subspace using Gram-Schmidt
                // orthogonalization algorithm.
                let mut basis: Vec<$t<N>> = Vec::new();

                for i in 0 .. $dimension {
                    let mut basis_element : $t<N> = ::zero();

                    unsafe {
                        basis_element.set_fast(i, ::one());
                    }

                    if basis.len() == $dimension - 1 {
                        break;
                    }

                    let mut elt = basis_element;

                    elt = elt - *n * Dot::dot(&basis_element, n);

                    for v in basis.iter() {
                        elt = elt - *v * Dot::dot(&elt, v)
                    };

                    if !ApproxEq::approx_eq(&Norm::norm_squared(&elt), &::zero()) {
                        let new_element = Norm::normalize(&elt);

                        if !f(new_element) { return };

                        basis.push(new_element);
                    }
                }
            }

            #[inline]
            fn canonical_basis_element(i: usize) -> Option<$t<N>> {
                if i < $dimension {
                    let mut basis_element : $t<N> = ::zero();

                    unsafe {
                        basis_element.set_fast(i, ::one());
                    }

                    Some(basis_element)
                }
                else {
                    None
                }
            }
        }
    )
);

macro_rules! axpy_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Axpy<N>> Axpy<N> for $t<N> {
            #[inline]
            fn axpy(&mut self, a: &N, x: &$t<N>) {
                $( self.$compN.axpy(a, &x.$compN); )+
            }
        }
    )
);

macro_rules! add_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Add<N, Output = N>> Add<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN + right.$compN),+)
            }
        }

        impl<N: AddAssign<N>> AddAssign<$t<N>> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: $t<N>) {
                $( self.$compN += right.$compN; )+
            }
        }
    )
);

macro_rules! scalar_add_impl(
    ($t: ident, $($compN: ident),+) => (
        // $t against scalar
        impl<N: Copy + Add<N, Output = N>> Add<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: N) -> $t<N> {
                $t::new($(self.$compN + right),+)
            }
        }

        impl<N: Copy + AddAssign<N>> AddAssign<N> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: N) {
                $( self.$compN += right; )+
            }
        }

        impl Add<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn add(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self + right.$compN),+)
            }
        }

        impl Add<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn add(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self + right.$compN),+)
            }
        }
    )
);

macro_rules! sub_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN - right.$compN),+)
            }
        }

        impl<N: SubAssign<N>> SubAssign<$t<N>> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: $t<N>) {
                $( self.$compN -= right.$compN; )+
            }
        }
    )
);

macro_rules! scalar_sub_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Sub<N, Output = N>> Sub<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: N) -> $t<N> {
                $t::new($(self.$compN - right),+)
            }
        }

        impl<N: Copy + SubAssign<N>> SubAssign<N> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: N) {
                $( self.$compN -= right; )+
            }
        }

        impl Sub<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn sub(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self - right.$compN),+)
            }
        }

        impl Sub<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn sub(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self - right.$compN),+)
            }
        }
    )
);

macro_rules! mul_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Mul<N, Output = N>> Mul<$t<N>> for $t<N> {
            type Output = $t<N>;
            #[inline]
            fn mul(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN * right.$compN),+)
            }
        }

        impl<N: MulAssign<N>> MulAssign<$t<N>> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: $t<N>) {
                $( self.$compN *= right.$compN; )+
            }
        }
    )
);

macro_rules! scalar_mul_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Mul<N, Output = N>> Mul<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: N) -> $t<N> {
                $t::new($(self.$compN * right),+)
            }
        }

        impl<N: Copy + MulAssign<N>> MulAssign<N> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: N) {
                $( self.$compN *= right; )+
            }
        }

        impl Mul<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn mul(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self * right.$compN),+)
            }
        }

        impl Mul<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn mul(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self * right.$compN),+)
            }
        }
    )
);

macro_rules! div_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N>> Div<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn div(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN / right.$compN),+)
            }
        }

        impl<N: DivAssign<N>> DivAssign<$t<N>> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: $t<N>) {
                $( self.$compN /= right.$compN; )+
            }
        }
    )
);

macro_rules! scalar_div_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N>> Div<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn div(self, right: N) -> $t<N> {
                $t::new($(self.$compN / right),+)
            }
        }

        impl<N: Copy + DivAssign<N>> DivAssign<N> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: N) {
                $( self.$compN /= right; )+
            }
        }
    )
);

macro_rules! neg_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Neg<Output = N> + Copy> Neg for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn neg(self) -> $t<N> {
                $t::new($(-self.$compN ),+)
            }
        }
    )
);

macro_rules! add (
    // base case
    ($x:expr) => {
        $x
    };
    // `$x` followed by at least one `$y,`
    ($x:expr, $($y:expr),+) => {
        // call min! on the tail `$y`
        Add::add($x, add!($($y),+))
    }
);

macro_rules! dot_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: BaseNum> Dot<N> for $t<N> {
            #[inline]
            fn dot(&self, other: &$t<N>) -> N {
                add!($(self.$compN * other.$compN ),+)
            }
        }
    )
);

macro_rules! translation_impl(
    ($t: ident) => (
        impl<N: Copy + Add<N, Output = N> + Neg<Output = N>> Translation<$t<N>> for $t<N> {
            #[inline]
            fn translation(&self) -> $t<N> {
                *self
            }

            #[inline]
            fn inverse_translation(&self) -> $t<N> {
                -*self
            }

            #[inline]
            fn append_translation_mut(&mut self, t: &$t<N>) {
                *self = *t + *self;
            }

            #[inline]
            fn append_translation(&self, t: &$t<N>) -> $t<N> {
                *t + *self
            }

            #[inline]
            fn prepend_translation_mut(&mut self, t: &$t<N>) {
                *self = *self + *t;
            }

            #[inline]
            fn prepend_translation(&self, t: &$t<N>) -> $t<N> {
                *self + *t
            }

            #[inline]
            fn set_translation(&mut self, t: $t<N>) {
                *self = t
            }
        }
    )
);

macro_rules! norm_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: BaseFloat> Norm<N> for $t<N> {
            #[inline]
            fn norm_squared(&self) -> N {
                Dot::dot(self, self)
            }

            #[inline]
            fn normalize(&self) -> $t<N> {
                let mut res : $t<N> = *self;
                let _ = res.normalize_mut();
                res
            }

            #[inline]
            fn normalize_mut(&mut self) -> N {
                let l = Norm::norm(self);

                $(self.$compN = self.$compN / l;)*

                l
            }
        }
    )
);

macro_rules! approx_eq_impl(
    ($t: ident, $($compN: ident),+) => (
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
            fn approx_eq(&self, other: &$t<N>) -> bool {
                $(ApproxEq::approx_eq(&self.$compN, &other.$compN))&&+
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, eps: &N) -> bool {
                $(ApproxEq::approx_eq_eps(&self.$compN, &other.$compN, eps))&&+
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
                $(ApproxEq::approx_eq_ulps(&self.$compN, &other.$compN, ulps))&&+
            }
        }
    )
);

macro_rules! zero_one_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N> One for $t<N>
            where N: Copy + One + Sub<N, Output = N> + Add<N, Output = N> {
            #[inline]
            fn one() -> $t<N> {
                $t {
                    $($compN: ::one() ),+
                }
            }
        }

        impl<N: Zero> Zero for $t<N> {
            #[inline]
            fn zero() -> $t<N> {
                $t {
                    $($compN: ::zero() ),+
                }
            }

            #[inline]
            fn is_zero(&self) -> bool {
                $(self.$compN.is_zero() )&&+
            }
        }
    )
);

macro_rules! from_iterator_impl(
    ($t: ident, $param0: ident) => (
        impl<N> FromIterator<N> for $t<N> {
            #[inline]
            fn from_iter<I: IntoIterator<Item = N>>($param0: I) -> $t<N> {
                let mut $param0 = $param0.into_iter();
                $t::new($param0.next().unwrap())
            }
        }
    );
    ($t: ident, $param0: ident, $($paramN: ident),+) => (
        impl<N> FromIterator<N> for $t<N> {
            #[inline]
            fn from_iter<I: IntoIterator<Item = N>>($param0: I) -> $t<N> {
                let mut $param0 = $param0.into_iter();
                $t::new($param0.next().unwrap(),
                        $($paramN.next().unwrap()),+)
            }
        }
    )
);

macro_rules! bounded_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Bounded> Bounded for $t<N> {
            #[inline]
            fn max_value() -> $t<N> {
                $t {
                    $($compN: Bounded::max_value() ),+
                }
            }

            #[inline]
            fn min_value() -> $t<N> {
                $t {
                    $($compN: Bounded::min_value() ),+
                }
            }
        }
    )
);

macro_rules! vec_to_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + One + Zero> ToHomogeneous<$t2<N>> for $t<N> {
            fn to_homogeneous(&self) -> $t2<N> {
                let mut res: $t2<N> = ::zero();

                $( res.$compN = self.$compN; )+

                res
            }
        }
    )
);

macro_rules! vec_from_homogeneous_impl(
    ($t: ident, $t2: ident, $extra: ident, $($compN: ident),+) => (
        impl<N: Copy + Div<N, Output = N> + One + Zero> FromHomogeneous<$t2<N>> for $t<N> {
            fn from(v: &$t2<N>) -> $t<N> {
                let mut res: $t<N> = ::zero();

                $( res.$compN = v.$compN; )+

                res
            }
        }
    )
);

macro_rules! translate_impl(
    ($tv: ident, $t: ident) => (
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Translate<$t<N>> for $tv<N> {
            fn translate(&self, other: &$t<N>) -> $t<N> {
                *other + *self
            }

            fn inverse_translate(&self, other: &$t<N>) -> $t<N> {
                *other - *self
            }
        }
    )
);

macro_rules! rotate_impl(
    ($t: ident) => (
        impl<N, O: Copy> Rotate<O> for $t<N> {
            fn rotate(&self, other: &O) -> O {
                *other
            }

            fn inverse_rotate(&self, other: &O) -> O {
                *other
            }
        }
    )
);

macro_rules! transform_impl(
    ($tv: ident, $t: ident) => (
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Transform<$t<N>> for $tv<N> {
            fn transform(&self, other: &$t<N>) -> $t<N> {
                self.translate(other)
            }

            fn inverse_transform(&self, other: &$t<N>) -> $t<N> {
                self.inverse_translate(other)
            }
        }
    )
);

macro_rules! vec_as_point_impl(
    ($tv: ident, $t: ident, $($compN: ident),+) => (
        impl<N> $tv<N> {
            /// Converts this vector to a point.
            #[inline]
            pub fn to_point(self) -> $t<N> {
                $t::new(
                    $(self.$compN),+
                )
            }

            /// Reinterprets this vector as a point.
            #[inline]
            pub fn as_point(&self) -> &$t<N> {
                unsafe {
                    mem::transmute(self)
                }
            }
        }
    )
);

macro_rules! num_float_vec_impl(
    ($t: ident) => (
        impl<N> NumVector<N> for $t<N>
            where N: BaseNum {
        }

        impl<N> FloatVector<N> for $t<N>
            where N: BaseFloat + ApproxEq<N> {
        }
    )
);

macro_rules! absolute_vec_impl(
  ($t: ident, $($compN: ident),+) => (
    impl<N: Absolute<N>> Absolute<$t<N>> for $t<N> {
        #[inline]
        fn abs(m: &$t<N>) -> $t<N> {
            $t::new($(::abs(&m.$compN) ),+)
        }
    }
  )
);

macro_rules! arbitrary_impl(
    ($t: ident, $($compN: ident),*) => (
        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary> Arbitrary for $t<N> {
            #[inline]
            fn arbitrary<G: Gen>(g: &mut G) -> $t<N> {
                $t { $($compN: Arbitrary::arbitrary(g),)* }
            }
        }
    )
);

macro_rules! rand_impl(
    ($t: ident, $($compN: ident),*) => (
        impl<N: Rand> Rand for $t<N> {
            #[inline]
            fn rand<R: Rng>(rng: &mut R) -> $t<N> {
                $t { $($compN: Rand::rand(rng), )* }
            }
        }
    )
);

macro_rules! mean_impl(
    ($t: ident) => (
        impl<N: BaseFloat + Cast<f64>> Mean<N> for $t<N> {
            #[inline]
            fn mean(&self) -> N {
                let normalizer = ::cast(1.0f64 / self.len() as f64);
                self.iter().fold(::zero(), |acc, x| acc + *x * normalizer)
            }
        }
    )
);

macro_rules! vec_display_impl(
    ($t: ident) => (
        impl<N: fmt::Display> fmt::Display for $t<N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                try!(write!(f, "("));

                let mut it = self.iter();

                let precision = f.precision().unwrap_or(8);

                try!(write!(f, "{:.*}", precision, *it.next().unwrap()));

                for comp in it {
                    try!(write!(f, ", {:.*}", precision, *comp));
                }

                write!(f, ")")
            }
        }
    )
);
