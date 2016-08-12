#![macro_use]

macro_rules! vectorlike_impl(
    ($t: ident, $dimension: expr, $($compN: ident),+) => (
        componentwise_neg!($t, $($compN),+);
        componentwise_repeat!($t, $($compN),+);
        componentwise_arbitrary!($t, $($compN),+);
        componentwise_rand!($t, $($compN),+);
        pointwise_scalar_add!($t, $($compN),+);
        pointwise_scalar_sub!($t, $($compN),+);
        pointwise_scalar_mul!($t, $($compN),+);
        pointwise_scalar_div!($t, $($compN),+);
        component_new!($t, $($compN),+);
        partial_order_impl!($t, $($compN),+);


        /*
         *
         * Cast between inner scalar type.
         *
         */
        impl<Nin: Copy, Nout: Copy + Cast<Nin>> Cast<$t<Nin>> for $t<Nout> {
            #[inline]
            fn from(v: $t<Nin>) -> $t<Nout> {
                $t::new($(Cast::from(v.$compN)),+)
            }
        }


        /*
         *
         * ApproxEq
         *
         */
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


        /*
         *
         * Unsafe indexing.
         *
         */
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


        /*
         *
         * Axpy
         *
         */
        impl<N: Axpy<N>> Axpy<N> for $t<N> {
            #[inline]
            fn axpy(&mut self, a: &N, x: &$t<N>) {
                $( self.$compN.axpy(a, &x.$compN); )+
            }
        }


        /*
         *
         * Bounded
         *
         */
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


        /*
         *
         * Container
         *
         */
        impl<N> $t<N> {
            /// The dimension of this entity.
            #[inline]
            pub fn len(&self) -> usize {
                Dimension::dimension(None::<$t<N>>)
            }
        }


        /*
         *
         * Conversions from/to slices
         *
         */
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

        impl<'a, N: Clone> From<&'a [N; $dimension]> for $t<N> {
            #[inline]
            fn from(arr: &'a [N; $dimension]) -> $t<N> {
                let vref: &$t<N> = From::from(arr);
                vref.clone()
            }
        }


        /*
         *
         * Dimension
         *
         */
        impl<N> Dimension for $t<N> {
            #[inline]
            fn dimension(_: Option<$t<N>>) -> usize {
                $dimension
            }
        }


        /*
         *
         * Indexable
         *
         */
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


        /*
         *
         * Index
         *
         */
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


        /*
         *
         * Iterable
         *
         */
        impl<N> Iterable<N> for $t<N> {
            #[inline]
            fn iter(&self) -> Iter<N> {
                unsafe {
                    mem::transmute::<&$t<N>, &[N; $dimension]>(self).iter()
                }
            }
        }

        impl<N> IterableMut<N> for $t<N> {
            #[inline]
            fn iter_mut(&mut self) -> IterMut<N> {
                unsafe {
                    mem::transmute::<&mut $t<N>, &mut [N; $dimension]>(self).iter_mut()
                }
            }
        }
    )
);

macro_rules! vector_impl(
    ($t: ident, $tp: ident, $($compN: ident),+) => (
        pointwise_add!($t, $($compN),+);
        pointwise_sub!($t, $($compN),+);
        pointwise_mul!($t, $($compN),+);
        pointwise_div!($t, $($compN),+);
        componentwise_zero!($t, $($compN),+);
        componentwise_one!($t, $($compN),+);
        componentwise_absolute!($t, $($compN),+);
        component_basis_element!($t, $($compN),+);


        /*
         *
         * Dot product
         *
         */
        impl<N: BaseNum> Dot<N> for $t<N> {
            #[inline]
            fn dot(&self, other: &$t<N>) -> N {
                add!($(self.$compN * other.$compN ),+)
            }
        }


        /*
         *
         * Norm
         *
         */
        impl<N: BaseFloat> Norm for $t<N> {
            type NormType = N;

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
                let n = ::norm(self);
                *self /= n;

                n
            }

            #[inline]
            fn try_normalize(&self, min_norm: N) -> Option<$t<N>> {
                let n = ::norm(self);

                if n <= min_norm {
                    None
                }
                else {
                    Some(*self / n)
                }
            }
        }


        /*
         *
         * Translation
         *
         */
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



        /*
         *
         * Translate
         *
         */
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Translate<$tp<N>> for $t<N> {
            fn translate(&self, other: &$tp<N>) -> $tp<N> {
                *other + *self
            }

            fn inverse_translate(&self, other: &$tp<N>) -> $tp<N> {
                *other - *self
            }
        }


        /*
         *
         * Rotate
         *
         */
        impl<N, O: Copy> Rotate<O> for $t<N> {
            fn rotate(&self, other: &O) -> O {
                *other
            }

            fn inverse_rotate(&self, other: &O) -> O {
                *other
            }
        }

        impl<N, O: Copy> Rotate<O> for $tp<N> {
            fn rotate(&self, other: &O) -> O {
                *other
            }

            fn inverse_rotate(&self, other: &O) -> O {
                *other
            }
        }



        /*
         *
         * Transform
         *
         */
        impl<N: Copy + Add<N, Output = N> + Sub<N, Output = N>> Transform<$tp<N>> for $t<N> {
            fn transform(&self, other: &$tp<N>) -> $tp<N> {
                self.translate(other)
            }

            fn inverse_transform(&self, other: &$tp<N>) -> $tp<N> {
                self.inverse_translate(other)
            }
        }



        /*
         *
         * Conversion to point.
         *
         */
        impl<N> $t<N> {
            /// Converts this vector to a point.
            #[inline]
            pub fn to_point(self) -> $tp<N> {
                $tp::new(
                    $(self.$compN),+
                )
            }

            /// Reinterprets this vector as a point.
            #[inline]
            pub fn as_point(&self) -> &$tp<N> {
                unsafe {
                    mem::transmute(self)
                }
            }
        }


        /*
         *
         * NumVector / FloatVector
         *
         */
        impl<N> NumVector<N> for $t<N>
            where N: BaseNum {
        }

        impl<N> FloatVector<N> for $t<N>
            where N: BaseFloat {
        }



        /*
         *
         * Mean
         *
         */
        impl<N: BaseFloat + Cast<f64>> Mean<N> for $t<N> {
            #[inline]
            fn mean(&self) -> N {
                let normalizer = ::cast(1.0f64 / self.len() as f64);
                self.iter().fold(::zero(), |acc, x| acc + *x * normalizer)
            }
        }


        /*
         *
         * Display
         *
         */
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


macro_rules! basis_impl(
    ($t: ident, $dimension: expr) => (
        impl<N: BaseFloat> Basis for $t<N> {
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

                    if !ApproxEq::approx_eq(&::norm_squared(&elt), &::zero()) {
                        let new_element = ::normalize(&elt);

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


// We need to keep this on a separate macro to retrieve the first component nam.
macro_rules! partial_order_impl(
    ($t: ident, $comp0: ident $(, $compN: ident)*) => (
        /*
         *
         * PartialOrder
         *
         */
        impl<N: BaseFloat> PartialOrder for $t<N> {
            #[inline]
            fn inf(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.min(other.$comp0), $(self.$compN.min(other.$compN)),*)
            }

            #[inline]
            fn sup(&self, other: &$t<N>) -> $t<N> {
                $t::new(self.$comp0.max(other.$comp0), $(self.$compN.max(other.$compN)),*)
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
