#![macro_escape]

macro_rules! dvec_impl(
    ($dvec: ident, $mul: ident, $div: ident, $add: ident, $sub: ident) => (
        double_dispatch_binop_decl_trait!($dvec, $mul)
        double_dispatch_binop_decl_trait!($dvec, $div)
        double_dispatch_binop_decl_trait!($dvec, $add)
        double_dispatch_binop_decl_trait!($dvec, $sub)

        mul_redispatch_impl!($dvec, $mul)
        div_redispatch_impl!($dvec, $div)
        add_redispatch_impl!($dvec, $add)
        sub_redispatch_impl!($dvec, $sub)

        impl<N: Zero + Clone> $dvec<N> {
            /// Builds a vector filled with zeros.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the vector.
            #[inline]
            pub fn new_zeros(dim: uint) -> $dvec<N> {
                $dvec::from_elem(dim, Zero::zero())
            }

            /// Tests if all components of the vector are zeroes.
            #[inline]
            pub fn is_zero(&self) -> bool {
                self.as_slice().iter().all(|e| e.is_zero())
            }
        }

        impl<N> $dvec<N> {
            /// Slices this vector.
            #[inline]
            pub fn as_slice<'a>(&'a self) -> &'a [N] {
                self.at.slice_to(self.len())
            }

            /// Mutably slices this vector.
            #[inline]
            pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [N] {
                let len = self.len();
                self.at.slice_to_mut(len)
            }
        }

        impl<N> Shape<uint, N> for $dvec<N> {
            #[inline]
            fn shape(&self) -> uint {
                self.len()
            }
        }

        impl<N: Clone> Indexable<uint, N> for $dvec<N> {
            #[inline]
            fn at(&self, i: uint) -> N {
                assert!(i < self.len());
                unsafe {
                    self.unsafe_at(i)
                }
            }

            #[inline]
            fn set(&mut self, i: uint, val: N) {
                assert!(i < self.len());
                unsafe {
                    self.unsafe_set(i, val);
                }
            }

            #[inline]
            fn swap(&mut self, i: uint, j: uint) {
                assert!(i < self.len());
                assert!(j < self.len());
                self.as_mut_slice().swap(i, j);
            }

            #[inline]
            unsafe fn unsafe_at(&self, i: uint) -> N {
                (*self.at.as_slice().unsafe_get(i)).clone()
            }

            #[inline]
            unsafe fn unsafe_set(&mut self, i: uint, val: N) {
                *self.at.as_mut_slice().unsafe_mut(i) = val
            }

        }

        impl<N> Index<uint, N> for $dvec<N> {
            fn index(&self, i: &uint) -> &N {
                &self.as_slice()[*i]
            }
        }

        impl<N> IndexMut<uint, N> for $dvec<N> {
            fn index_mut(&mut self, i: &uint) -> &mut N {
                &mut self.as_mut_slice()[*i]
            }
        }

        impl<N: One + Zero + Clone> $dvec<N> {
            /// Builds a vector filled with ones.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the vector.
            #[inline]
            pub fn new_ones(dim: uint) -> $dvec<N> {
                $dvec::from_elem(dim, One::one())
            }
        }

        impl<N: Rand + Zero> $dvec<N> {
            /// Builds a vector filled with random values.
            #[inline]
            pub fn new_random(dim: uint) -> $dvec<N> {
                $dvec::from_fn(dim, |_| rand::random())
            }
        }

        impl<N> Iterable<N> for $dvec<N> {
            #[inline]
            fn iter<'l>(&'l self) -> Items<'l, N> {
                self.as_slice().iter()
            }
        }

        impl<N> IterableMut<N> for $dvec<N> {
            #[inline]
            fn iter_mut<'l>(&'l mut self) -> MutItems<'l, N> {
                self.as_mut_slice().iter_mut()
            }
        }

        impl<N: Clone + BaseFloat + ApproxEq<N> + $mul<N, $dvec<N>>> $dvec<N> {
            /// Computes the canonical basis for the given dimension. A canonical basis is a set of
            /// vectors, mutually orthogonal, with all its component equal to 0.0 except one which is equal
            /// to 1.0.
            pub fn canonical_basis_with_dim(dim: uint) -> Vec<$dvec<N>> {
                let mut res : Vec<$dvec<N>> = Vec::new();

                for i in range(0u, dim) {
                    let mut basis_element : $dvec<N> = $dvec::new_zeros(dim);

                    basis_element.set(i, One::one());

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

                for i in range(0u, dim) {
                    let mut basis_element : $dvec<N> = $dvec::new_zeros(self.len());

                    basis_element.set(i, One::one());

                    if res.len() == dim - 1 {
                        break;
                    }

                    let mut elt = basis_element.clone();

                    elt = elt - *self * Dot::dot(&basis_element, self);

                    for v in res.iter() {
                        elt = elt - *v * Dot::dot(&elt, v)
                    };

                    if !ApproxEq::approx_eq(&Norm::sqnorm(&elt), &Zero::zero()) {
                        res.push(Norm::normalize_cpy(&elt));
                    }
                }

                assert!(res.len() == dim - 1);

                res
            }
        }

        impl<N: Mul<N, N> + Zero> $mul<N, $dvec<N>> for $dvec<N> {
            #[inline]
            fn binop(left: &$dvec<N>, right: &$dvec<N>) -> $dvec<N> {
                assert!(left.len() == right.len());
                FromIterator::from_iter(left.as_slice().iter().zip(right.as_slice().iter()).map(|(a, b)| *a * *b))
            }
        }

        impl<N: Div<N, N> + Zero> $div<N, $dvec<N>> for $dvec<N> {
            #[inline]
            fn binop(left: &$dvec<N>, right: &$dvec<N>) -> $dvec<N> {
                assert!(left.len() == right.len());
                FromIterator::from_iter(left.as_slice().iter().zip(right.as_slice().iter()).map(|(a, b)| *a / *b))
            }
        }

        impl<N: Add<N, N> + Zero> $add<N, $dvec<N>> for $dvec<N> {
            #[inline]
            fn binop(left: &$dvec<N>, right: &$dvec<N>) -> $dvec<N> {
                assert!(left.len() == right.len());
                FromIterator::from_iter(left.as_slice().iter().zip(right.as_slice().iter()).map(|(a, b)| *a + *b))
            }
        }

        impl<N: Sub<N, N> + Zero> $sub<N, $dvec<N>> for $dvec<N> {
            #[inline]
            fn binop(left: &$dvec<N>, right: &$dvec<N>) -> $dvec<N> {
                assert!(left.len() == right.len());
                FromIterator::from_iter(left.as_slice().iter().zip(right.as_slice().iter()).map(|(a, b)| *a - *b))
            }
        }

        impl<N: Neg<N> + Zero> Neg<$dvec<N>> for $dvec<N> {
            #[inline]
            fn neg(&self) -> $dvec<N> {
                FromIterator::from_iter(self.as_slice().iter().map(|a| -*a))
            }
        }

        impl<N: Num + Clone> Dot<N> for $dvec<N> {
            #[inline]
            fn dot(a: &$dvec<N>, b: &$dvec<N>) -> N {
                assert!(a.len() == b.len());

                let mut res: N = Zero::zero();

                for i in range(0u, a.len()) {
                    res = res + unsafe { a.unsafe_at(i) * b.unsafe_at(i) };
                }

                res
            }
        }

        impl<N: BaseFloat + Clone> Norm<N> for $dvec<N> {
            #[inline]
            fn sqnorm(v: &$dvec<N>) -> N {
                Dot::dot(v, v)
            }

            #[inline]
            fn norm(v: &$dvec<N>) -> N {
                Norm::sqnorm(v).sqrt()
            }

            #[inline]
            fn normalize_cpy(v: &$dvec<N>) -> $dvec<N> {
                let mut res : $dvec<N> = v.clone();

                let _ = res.normalize();

                res
            }

            #[inline]
            fn normalize(&mut self) -> N {
                let l = Norm::norm(self);

                for n in self.as_mut_slice().iter_mut() {
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
            fn approx_eq(a: &$dvec<N>, b: &$dvec<N>) -> bool {
                let mut zip = a.as_slice().iter().zip(b.as_slice().iter());

                zip.all(|(a, b)| ApproxEq::approx_eq(a, b))
            }

            #[inline]
            fn approx_eq_eps(a: &$dvec<N>, b: &$dvec<N>, epsilon: &N) -> bool {
                let mut zip = a.as_slice().iter().zip(b.as_slice().iter());

                zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
            }
        }

        dvec_scalar_mul_impl!($dvec, f64, $mul)
        dvec_scalar_mul_impl!($dvec, f32, $mul)
        dvec_scalar_mul_impl!($dvec, u64, $mul)
        dvec_scalar_mul_impl!($dvec, u32, $mul)
        dvec_scalar_mul_impl!($dvec, u16, $mul)
        dvec_scalar_mul_impl!($dvec, u8, $mul)
        dvec_scalar_mul_impl!($dvec, i64, $mul)
        dvec_scalar_mul_impl!($dvec, i32, $mul)
        dvec_scalar_mul_impl!($dvec, i16, $mul)
        dvec_scalar_mul_impl!($dvec, i8, $mul)
        dvec_scalar_mul_impl!($dvec, uint, $mul)
        dvec_scalar_mul_impl!($dvec, int, $mul)

        dvec_scalar_div_impl!($dvec, f64, $div)
        dvec_scalar_div_impl!($dvec, f32, $div)
        dvec_scalar_div_impl!($dvec, u64, $div)
        dvec_scalar_div_impl!($dvec, u32, $div)
        dvec_scalar_div_impl!($dvec, u16, $div)
        dvec_scalar_div_impl!($dvec, u8, $div)
        dvec_scalar_div_impl!($dvec, i64, $div)
        dvec_scalar_div_impl!($dvec, i32, $div)
        dvec_scalar_div_impl!($dvec, i16, $div)
        dvec_scalar_div_impl!($dvec, i8, $div)
        dvec_scalar_div_impl!($dvec, uint, $div)
        dvec_scalar_div_impl!($dvec, int, $div)

        dvec_scalar_add_impl!($dvec, f64, $add)
        dvec_scalar_add_impl!($dvec, f32, $add)
        dvec_scalar_add_impl!($dvec, u64, $add)
        dvec_scalar_add_impl!($dvec, u32, $add)
        dvec_scalar_add_impl!($dvec, u16, $add)
        dvec_scalar_add_impl!($dvec, u8, $add)
        dvec_scalar_add_impl!($dvec, i64, $add)
        dvec_scalar_add_impl!($dvec, i32, $add)
        dvec_scalar_add_impl!($dvec, i16, $add)
        dvec_scalar_add_impl!($dvec, i8, $add)
        dvec_scalar_add_impl!($dvec, uint, $add)
        dvec_scalar_add_impl!($dvec, int, $add)

        dvec_scalar_sub_impl!($dvec, f64, $sub)
        dvec_scalar_sub_impl!($dvec, f32, $sub)
        dvec_scalar_sub_impl!($dvec, u64, $sub)
        dvec_scalar_sub_impl!($dvec, u32, $sub)
        dvec_scalar_sub_impl!($dvec, u16, $sub)
        dvec_scalar_sub_impl!($dvec, u8, $sub)
        dvec_scalar_sub_impl!($dvec, i64, $sub)
        dvec_scalar_sub_impl!($dvec, i32, $sub)
        dvec_scalar_sub_impl!($dvec, i16, $sub)
        dvec_scalar_sub_impl!($dvec, i8, $sub)
        dvec_scalar_sub_impl!($dvec, uint, $sub)
        dvec_scalar_sub_impl!($dvec, int, $sub)
    )
)

macro_rules! dvec_scalar_mul_impl (
    ($dvec: ident, $n: ident, $mul: ident) => (
        impl $mul<$n, $dvec<$n>> for $n {
            #[inline]
            fn binop(left: &$dvec<$n>, right: &$n) -> $dvec<$n> {
                FromIterator::from_iter(left.as_slice().iter().map(|a| *a * *right))
            }
        }
    )
)

macro_rules! dvec_scalar_div_impl (
    ($dvec: ident, $n: ident, $div: ident) => (
        impl $div<$n, $dvec<$n>> for $n {
            #[inline]
            fn binop(left: &$dvec<$n>, right: &$n) -> $dvec<$n> {
                FromIterator::from_iter(left.as_slice().iter().map(|a| *a / *right))
            }
        }
    )
)

macro_rules! dvec_scalar_add_impl (
    ($dvec: ident, $n: ident, $add: ident) => (
        impl $add<$n, $dvec<$n>> for $n {
            #[inline]
            fn binop(left: &$dvec<$n>, right: &$n) -> $dvec<$n> {
                FromIterator::from_iter(left.as_slice().iter().map(|a| *a + *right))
            }
        }
    )
)

macro_rules! dvec_scalar_sub_impl (
    ($dvec: ident, $n: ident, $sub: ident) => (
        impl $sub<$n, $dvec<$n>> for $n {
            #[inline]
            fn binop(left: &$dvec<$n>, right: &$n) -> $dvec<$n> {
                FromIterator::from_iter(left.as_slice().iter().map(|a| *a - *right))
            }
        }
    )
)

macro_rules! small_dvec_impl (
    ($dvec: ident, $dim: expr, $mul: ident, $div: ident, $add: ident, $sub: ident $(,$idx: expr)*) => (
        impl<N> $dvec<N> {
            #[inline]
            pub fn len(&self) -> uint {
                self.dim
            }
        }

        impl<N: PartialEq> PartialEq for $dvec<N> {
            #[inline]
            fn eq(&self, other: &$dvec<N>) -> bool {
                if self.len() != other.len() {
                    return false; // FIXME: fail instead?
                }

                for (a, b) in self.as_slice().iter().zip(other.as_slice().iter()) {
                    if *a != *b {
                        return false;
                    }
                }

                true
            }
        }

        impl<N: Clone> Clone for $dvec<N> {
            fn clone(&self) -> $dvec<N> {
                let at: [N, ..$dim] = [ $( self.at[$idx].clone(), )* ];

                $dvec {
                    at:  at,
                    dim: self.dim
                }
            }
        }

        dvec_impl!($dvec, $mul, $div, $add, $sub)
    )
)

macro_rules! small_dvec_from_impl (
    ($dvec: ident, $dim: expr $(,$zeros: expr)*) => (
        impl<N: Clone + Zero> $dvec<N> {
            /// Builds a vector filled with a constant.
            #[inline]
            pub fn from_elem(dim: uint, elem: N) -> $dvec<N> {
                assert!(dim <= $dim);

                let mut at: [N, ..$dim] = [ $( $zeros, )* ];

                for n in at.slice_to_mut(dim).iter_mut() {
                    *n = elem.clone();
                }

                $dvec {
                    at:  at,
                    dim: dim
                }
            }
        }

        impl<N: Clone + Zero> $dvec<N> {
            /// Builds a vector filled with the components provided by a vector.
            ///
            /// The vector must have at least `dim` elements.
            #[inline]
            pub fn from_slice(dim: uint, vec: &[N]) -> $dvec<N> {
                assert!(dim <= vec.len() && dim <= $dim);

                // FIXME: not safe.
                let mut at: [N, ..$dim] = [ $( $zeros, )* ];

                for (curr, other) in vec.iter().zip(at.iter_mut()) {
                    *other = curr.clone();
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
            pub fn from_fn(dim: uint, f: |uint| -> N) -> $dvec<N> {
                assert!(dim <= $dim);

                let mut at: [N, ..$dim] = [ $( $zeros, )* ];

                for i in range(0, dim) {
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
            fn from_iter<I: Iterator<N>>(mut param: I) -> $dvec<N> {
                let mut at: [N, ..$dim] = [ $( $zeros, )* ];

                let mut dim = 0;

                for n in param {
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
    )
)
