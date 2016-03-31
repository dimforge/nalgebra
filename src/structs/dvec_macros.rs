#![macro_use]

macro_rules! dvec_impl(
    ($dvec: ident) => (
        vecn_dvec_common_impl!($dvec);

        impl<N: Zero + Copy + Clone> $dvec<N> {
            /// Builds a vector filled with zeros.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the vector.
            #[inline]
            pub fn new_zeros(dim: usize) -> $dvec<N> {
                $dvec::from_elem(dim, ::zero())
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

        impl<N: BaseFloat + ApproxEq<N>> $dvec<N> {
            /// Computes the canonical basis for the given dimension. A canonical basis is a set of
            /// vectors, mutually orthogonal, with all its component equal to 0.0 except one which is equal
            /// to 1.0.
            pub fn canonical_basis_with_dim(dim: usize) -> Vec<$dvec<N>> {
                let mut res : Vec<$dvec<N>> = Vec::new();

                for i in 0 .. dim {
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

                for i in 0 .. dim {
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
    )
);

macro_rules! small_dvec_impl (
    ($dvec: ident, $dim: expr, $($idx: expr),*) => (
        dvec_impl!($dvec);

        impl<N> $dvec<N> {
            /// The number of elements of this vector.
            #[inline]
            pub fn len(&self) -> usize {
                self.dim
            }

            /// Creates an uninitialized vec of dimension `dim`.
            #[inline]
            pub unsafe fn new_uninitialized(dim: usize) -> $dvec<N> {
                assert!(dim <= $dim, "The chosen dimension is too high for that type of \
                                      stack-allocated dynamic vector. Consider using the \
                                      heap-allocated vector: DVec.");

                $dvec {
                    at:  mem::uninitialized(),
                    dim: dim
                }
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

                for i in 0 .. dim {
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
