#![macro_use]

macro_rules! dvec_impl(
    ($dvector: ident) => (
        vecn_dvec_common_impl!($dvector);

        impl<N: Zero + Copy + Clone> $dvector<N> {
            /// Builds a vector filled with zeros.
            ///
            /// # Arguments
            /// * `dimension` - The dimension of the vector.
            #[inline]
            pub fn new_zeros(dimension: usize) -> $dvector<N> {
                $dvector::from_elem(dimension, ::zero())
            }
        }

        impl<N: One + Zero + Copy + Clone> $dvector<N> {
            /// Builds a vector filled with ones.
            ///
            /// # Arguments
            /// * `dimension` - The dimension of the vector.
            #[inline]
            pub fn new_ones(dimension: usize) -> $dvector<N> {
                $dvector::from_elem(dimension, ::one())
            }
        }

        impl<N: Rand + Zero> $dvector<N> {
            /// Builds a vector filled with random values.
            #[inline]
            pub fn new_random(dimension: usize) -> $dvector<N> {
                $dvector::from_fn(dimension, |_| rand::random())
            }
        }

        impl<N: BaseFloat + ApproxEq<N>> $dvector<N> {
            /// Computes the canonical basis for the given dimension. A canonical basis is a set of
            /// vectors, mutually orthogonal, with all its component equal to 0.0 except one which is equal
            /// to 1.0.
            pub fn canonical_basis_with_dim(dimension: usize) -> Vec<$dvector<N>> {
                let mut res : Vec<$dvector<N>> = Vec::new();

                for i in 0 .. dimension {
                    let mut basis_element : $dvector<N> = $dvector::new_zeros(dimension);

                    basis_element[i] = ::one();

                    res.push(basis_element);
                }

                res
            }

            /// Computes a basis of the space orthogonal to the vector. If the input vector is of dimension
            /// `n`, this will return `n - 1` vectors.
            pub fn orthogonal_subspace_basis(&self) -> Vec<$dvector<N>> {
                // compute the basis of the orthogonal subspace using Gram-Schmidt
                // orthogonalization algorithm
                let     dimension                 = self.len();
                let mut res : Vec<$dvector<N>> = Vec::new();

                for i in 0 .. dimension {
                    let mut basis_element : $dvector<N> = $dvector::new_zeros(self.len());

                    basis_element[i] = ::one();

                    if res.len() == dimension - 1 {
                        break;
                    }

                    let mut elt = basis_element.clone();

                    elt.axpy(&-::dot(&basis_element, self), self);

                    for v in res.iter() {
                        let proj = ::dot(&elt, v);
                        elt.axpy(&-proj, v)
                    };

                    if !ApproxEq::approx_eq(&Norm::norm_squared(&elt), &::zero()) {
                        res.push(Norm::normalize(&elt));
                    }
                }

                assert!(res.len() == dimension - 1);

                res
            }
        }
    )
);

macro_rules! small_dvec_impl (
    ($dvector: ident, $dimension: expr, $($idx: expr),*) => (
        dvec_impl!($dvector);

        impl<N> $dvector<N> {
            /// The number of elements of this vector.
            #[inline]
            pub fn len(&self) -> usize {
                self.dimension
            }

            /// Creates an uninitialized vector of dimension `dimension`.
            #[inline]
            pub unsafe fn new_uninitialized(dimension: usize) -> $dvector<N> {
                assert!(dimension <= $dimension, "The chosen dimension is too high for that type of \
                                      stack-allocated dynamic vector. Consider using the \
                                      heap-allocated vector: DVector.");

                $dvector {
                    at:  mem::uninitialized(),
                    dimension: dimension
                }
            }
        }

        impl<N: PartialEq> PartialEq for $dvector<N> {
            #[inline]
            fn eq(&self, other: &$dvector<N>) -> bool {
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

        impl<N: Clone> Clone for $dvector<N> {
            fn clone(&self) -> $dvector<N> {
                let at: [N; $dimension] = [ $( self.at[$idx].clone(), )* ];

                $dvector {
                    at:  at,
                    dimension: self.dimension
                }
            }
        }
    )
);

macro_rules! small_dvec_from_impl (
    ($dvector: ident, $dimension: expr, $($zeros: expr),*) => (
        impl<N: Copy + Zero> $dvector<N> {
            /// Builds a vector filled with a constant.
            #[inline]
            pub fn from_elem(dimension: usize, elem: N) -> $dvector<N> {
                assert!(dimension <= $dimension);

                let mut at: [N; $dimension] = [ $( $zeros, )* ];

                for n in &mut at[.. dimension] {
                    *n = elem;
                }

                $dvector {
                    at:  at,
                    dimension: dimension
                }
            }
        }

        impl<N: Copy + Zero> $dvector<N> {
            /// Builds a vector filled with the components provided by a vector.
            ///
            /// The vector must have at least `dimension` elements.
            #[inline]
            pub fn from_slice(dimension: usize, vector: &[N]) -> $dvector<N> {
                assert!(dimension <= vector.len() && dimension <= $dimension);

                // FIXME: not safe.
                let mut at: [N; $dimension] = [ $( $zeros, )* ];

                for (curr, other) in vector.iter().zip(at.iter_mut()) {
                    *other = *curr;
                }

                $dvector {
                    at:  at,
                    dimension: dimension
                }
            }
        }

        impl<N: Zero> $dvector<N> {
            /// Builds a vector filled with the result of a function.
            #[inline(always)]
            pub fn from_fn<F: FnMut(usize) -> N>(dimension: usize, mut f: F) -> $dvector<N> {
                assert!(dimension <= $dimension);

                let mut at: [N; $dimension] = [ $( $zeros, )* ];

                for i in 0 .. dimension {
                    at[i] = f(i);
                }

                $dvector {
                    at:  at,
                    dimension: dimension
                }
            }
        }

        impl<N: Zero> FromIterator<N> for $dvector<N> {
            #[inline]
            fn from_iter<I: IntoIterator<Item = N>>(param: I) -> $dvector<N> {
                let mut at: [N; $dimension] = [ $( $zeros, )* ];

                let mut dimension = 0;

                for n in param.into_iter() {
                    if dimension == $dimension {
                        break;
                    }

                    at[dimension] = n;

                    dimension = dimension + 1;
                }

                $dvector {
                    at:  at,
                    dimension: dimension
                }
            }
        }

        #[cfg(feature="arbitrary")]
        impl<N: Arbitrary + Zero> Arbitrary for $dvector<N> {
            #[inline]
            fn arbitrary<G: Gen>(g: &mut G) -> $dvector<N> {
                $dvector::from_fn(g.gen_range(0, $dimension), |_| Arbitrary::arbitrary(g))
            }
        }
    )
);
