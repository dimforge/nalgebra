#![macro_use]

macro_rules! dvec_impl(
    ($dvec: ident) => (
        impl<N: Zero + Copy + Clone> $dvec<N> {
            /// Builds a vector filled with zeros.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the vector.
            #[inline]
            pub fn new_zeros(dim: usize) -> $dvec<N> {
                $dvec::from_elem(dim, ::zero())
            }

            /// Tests if all components of the vector are zeroes.
            #[inline]
            pub fn is_zero(&self) -> bool {
                self.as_ref().iter().all(|e| e.is_zero())
            }
        }

        impl<N> $dvec<N> {
            /// Slices this vector.
            #[inline]
            pub fn as_ref<'a>(&'a self) -> &'a [N] {
                &self.at[.. self.len()]
            }

            /// Mutably slices this vector.
            #[inline]
            pub fn as_mut<'a>(&'a mut self) -> &'a mut [N] {
                let len = self.len();
                &mut self.at[.. len]
            }
        }

        impl<N> Shape<usize> for $dvec<N> {
            #[inline]
            fn shape(&self) -> usize {
                self.len()
            }
        }

        impl<N: Copy> Indexable<usize, N> for $dvec<N> {
            #[inline]
            fn swap(&mut self, i: usize, j: usize) {
                assert!(i < self.len());
                assert!(j < self.len());
                self.as_mut().swap(i, j);
            }

            #[inline]
            unsafe fn unsafe_at(&self, i: usize) -> N {
                *self.at[..].get_unchecked(i)
            }

            #[inline]
            unsafe fn unsafe_set(&mut self, i: usize, val: N) {
                *self.at[..].get_unchecked_mut(i) = val
            }

        }

        impl<N, T> Index<T> for $dvec<N> where [N]: Index<T> {
            type Output = <[N] as Index<T>>::Output;

            fn index(&self, i: T) -> &<[N] as Index<T>>::Output {
                &self.as_ref()[i]
            }
        }

        impl<N, T> IndexMut<T> for $dvec<N> where [N]: IndexMut<T> {
            fn index_mut(&mut self, i: T) -> &mut <[N] as Index<T>>::Output {
                &mut self.as_mut()[i]
            }
        }

        impl<N: One + Zero + Copy + Clone> $dvec<N> {
            /// Builds a vector filled with ones.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the vector.
            #[inline]
            pub fn new_ones(dim: usize) -> $dvec<N> {
                $dvec::from_elem(dim, ::one())
            }
        }

        impl<N: Rand + Zero> $dvec<N> {
            /// Builds a vector filled with random values.
            #[inline]
            pub fn new_random(dim: usize) -> $dvec<N> {
                $dvec::from_fn(dim, |_| rand::random())
            }
        }

        impl<N> Iterable<N> for $dvec<N> {
            #[inline]
            fn iter<'l>(&'l self) -> Iter<'l, N> {
                self.as_ref().iter()
            }
        }

        impl<N> IterableMut<N> for $dvec<N> {
            #[inline]
            fn iter_mut<'l>(&'l mut self) -> IterMut<'l, N> {
                self.as_mut().iter_mut()
            }
        }

        impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N>> Axpy<N> for $dvec<N> {
            fn axpy(&mut self, a: &N, x: &$dvec<N>) {
                assert!(self.len() == x.len());

                for i in 0..x.len() {
                    unsafe {
                        let self_i = self.unsafe_at(i);
                        self.unsafe_set(i, self_i + *a * x.unsafe_at(i))
                    }
                }
            }
        }

        impl<N: BaseFloat + ApproxEq<N>> $dvec<N> {
            /// Computes the canonical basis for the given dimension. A canonical basis is a set of
            /// vectors, mutually orthogonal, with all its component equal to 0.0 except one which is equal
            /// to 1.0.
            pub fn canonical_basis_with_dim(dim: usize) -> Vec<$dvec<N>> {
                let mut res : Vec<$dvec<N>> = Vec::new();

                for i in 0..dim {
                    let mut basis_element : $dvec<N> = $dvec::new_zeros(dim);

                    basis_element[i] = ::one();

                    res.push(basis_element);
                }

                res
            }

            /// Computes a basis of the space orthogonal to the vector. If the input vector is of dimension
            /// `n`, this will return `n - 1` vectors.
            pub fn orthogonal_subspace_basis(&self) -> Vec<$dvec<N>> {
                // compute the basis of the orthogonal subspace using Gram-Schmidt
                // orthogonalization algorithm
                let     dim                 = self.len();
                let mut res : Vec<$dvec<N>> = Vec::new();

                for i in 0..dim {
                    let mut basis_element : $dvec<N> = $dvec::new_zeros(self.len());

                    basis_element[i] = ::one();

                    if res.len() == dim - 1 {
                        break;
                    }

                    let mut elt = basis_element.clone();

                    elt.axpy(&-::dot(&basis_element, self), self);

                    for v in res.iter() {
                        let proj = ::dot(&elt, v);
                        elt.axpy(&-proj, v)
                    };

                    if !ApproxEq::approx_eq(&Norm::sqnorm(&elt), &::zero()) {
                        res.push(Norm::normalize(&elt));
                    }
                }

                assert!(res.len() == dim - 1);

                res
            }
        }

        impl<N: Copy + Mul<N, Output = N> + Zero> Mul<$dvec<N>> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn mul(self, right: $dvec<N>) -> $dvec<N> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left * *right
                }

                res
            }
        }

        impl<N: Copy + Div<N, Output = N> + Zero> Div<$dvec<N>> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn div(self, right: $dvec<N>) -> $dvec<N> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left / *right
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N> + Zero> Add<$dvec<N>> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn add(self, right: $dvec<N>) -> $dvec<N> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left + *right
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N> + Zero> Sub<$dvec<N>> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn sub(self, right: $dvec<N>) -> $dvec<N> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left - *right
                }

                res
            }
        }

        impl<N: Neg<Output = N> + Zero + Copy> Neg for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn neg(self) -> $dvec<N> {
                FromIterator::from_iter(self.as_ref().iter().map(|a| -*a))
            }
        }

        impl<N: BaseNum> Dot<N> for $dvec<N> {
            #[inline]
            fn dot(&self, other: &$dvec<N>) -> N {
                assert!(self.len() == other.len());
                let mut res: N = ::zero();
                for i in 0..self.len() {
                    res = res + unsafe { self.unsafe_at(i) * other.unsafe_at(i) };
                }
                res
            }
        }

        impl<N: BaseFloat> Norm<N> for $dvec<N> {
            #[inline]
            fn sqnorm(&self) -> N {
                Dot::dot(self, self)
            }

            #[inline]
            fn normalize(&self) -> $dvec<N> {
                let mut res : $dvec<N> = self.clone();
                let _ = res.normalize_mut();
                res
            }

            #[inline]
            fn normalize_mut(&mut self) -> N {
                let l = Norm::norm(self);

                for n in self.as_mut().iter_mut() {
                    *n = *n / l;
                }

                l
            }
        }

        impl<N: ApproxEq<N>> ApproxEq<N> for $dvec<N> {
            #[inline]
            fn approx_epsilon(_: Option<$dvec<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_ulps(_: Option<$dvec<N>>) -> u32 {
                ApproxEq::approx_ulps(None::<N>)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$dvec<N>, epsilon: &N) -> bool {
                let mut zip = self.as_ref().iter().zip(other.as_ref().iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$dvec<N>, ulps: u32) -> bool {
                let mut zip = self.as_ref().iter().zip(other.as_ref().iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
            }
        }

        impl<N: Copy + Mul<N, Output = N> + Zero> Mul<N> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn mul(self, right: N) -> $dvec<N> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e * right
                }

                res
            }
        }

        impl<N: Copy + Div<N, Output = N> + Zero> Div<N> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn div(self, right: N) -> $dvec<N> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e / right
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N> + Zero> Add<N> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn add(self, right: N) -> $dvec<N> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e + right
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N> + Zero> Sub<N> for $dvec<N> {
            type Output = $dvec<N>;

            #[inline]
            fn sub(self, right: N) -> $dvec<N> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e - right
                }

                res
            }
        }
    )
);

macro_rules! small_dvec_impl (
    ($dvec: ident, $dim: expr, $($idx: expr),*) => (
        impl<N> $dvec<N> {
            #[inline]
            pub fn len(&self) -> usize {
                self.dim
            }
        }

        impl<N: PartialEq> PartialEq for $dvec<N> {
            #[inline]
            fn eq(&self, other: &$dvec<N>) -> bool {
                if self.len() != other.len() {
                    return false; // FIXME: fail instead?
                }

                for (a, b) in self.as_ref().iter().zip(other.as_ref().iter()) {
                    if *a != *b {
                        return false;
                    }
                }

                true
            }
        }

        impl<N: Clone> Clone for $dvec<N> {
            fn clone(&self) -> $dvec<N> {
                let at: [N; $dim] = [ $( self.at[$idx].clone(), )* ];

                $dvec {
                    at:  at,
                    dim: self.dim
                }
            }
        }

        dvec_impl!($dvec);
    )
);

macro_rules! small_dvec_from_impl (
    ($dvec: ident, $dim: expr, $($zeros: expr),*) => (
        impl<N: Copy + Zero> $dvec<N> {
            /// Builds a vector filled with a constant.
            #[inline]
            pub fn from_elem(dim: usize, elem: N) -> $dvec<N> {
                assert!(dim <= $dim);

                let mut at: [N; $dim] = [ $( $zeros, )* ];

                for n in &mut at[.. dim] {
                    *n = elem;
                }

                $dvec {
                    at:  at,
                    dim: dim
                }
            }
        }

        impl<N: Copy + Zero> $dvec<N> {
            /// Builds a vector filled with the components provided by a vector.
            ///
            /// The vector must have at least `dim` elements.
            #[inline]
            pub fn from_slice(dim: usize, vec: &[N]) -> $dvec<N> {
                assert!(dim <= vec.len() && dim <= $dim);

                // FIXME: not safe.
                let mut at: [N; $dim] = [ $( $zeros, )* ];

                for (curr, other) in vec.iter().zip(at.iter_mut()) {
                    *other = *curr;
                }

                $dvec {
                    at:  at,
                    dim: dim
                }
            }
        }

        impl<N: Zero> $dvec<N> {
            /// Builds a vector filled with the result of a function.
            #[inline(always)]
            pub fn from_fn<F: FnMut(usize) -> N>(dim: usize, mut f: F) -> $dvec<N> {
                assert!(dim <= $dim);

                let mut at: [N; $dim] = [ $( $zeros, )* ];

                for i in 0..dim {
                    at[i] = f(i);
                }

                $dvec {
                    at:  at,
                    dim: dim
                }
            }
        }

        impl<N: Zero> FromIterator<N> for $dvec<N> {
            #[inline]
            fn from_iter<I: IntoIterator<Item = N>>(param: I) -> $dvec<N> {
                let mut at: [N; $dim] = [ $( $zeros, )* ];

                let mut dim = 0;

                for n in param.into_iter() {
                    if dim == $dim {
                        break;
                    }

                    at[dim] = n;

                    dim = dim + 1;
                }

                $dvec {
                    at:  at,
                    dim: dim
                }
            }
        }

        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + Zero> Arbitrary for $dvec<N> {
            #[inline]
            fn arbitrary<G: Gen>(g: &mut G) -> $dvec<N> {
                $dvec::from_fn(g.gen_range(0, $dim), |_| Arbitrary::arbitrary(g))
            }
        }
    )
);
