use std::num::{Zero, One, Algebraic};
use std::vec::{VecIterator, VecMutIterator};
use std::vec::from_elem;
use std::cmp::ApproxEq;
use std::iterator::{FromIterator, IteratorUtil};
use traits::iterable::{Iterable, IterableMut};
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

/// Vector with a dimension unknown at compile-time.
#[deriving(Eq, Ord, ToStr, Clone)]
pub struct DVec<N> {
    /// Components of the vector. Contains as much elements as the vector dimension.
    at: ~[N]
}

/// Builds a vector filled with zeros.
/// 
/// # Arguments
///   * `dim` - The dimension of the vector.
#[inline]
pub fn zero_vec_with_dim<N: Zero + Clone>(dim: uint) -> DVec<N> {
    DVec { at: from_elem(dim, Zero::zero::<N>()) }
}

/// Tests if all components of the vector are zeroes.
#[inline]
pub fn is_zero_vec<N: Zero>(vec: &DVec<N>) -> bool {
    vec.at.iter().all(|e| e.is_zero())
}

impl<N> Iterable<N> for DVec<N> {
    fn iter<'l>(&'l self) -> VecIterator<'l, N> {
        self.at.iter()
    }
}

impl<N> IterableMut<N> for DVec<N> {
    fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N> {
        self.at.mut_iter()
    }
}

impl<N, Iter: Iterator<N>> FromIterator<N, Iter> for DVec<N> {
    fn from_iterator(mut param: &mut Iter) -> DVec<N> {
        let mut res = DVec { at: ~[] };

        for e in param {
            res.at.push(e)
        }

        res
    }
}

impl<N: Clone + DivisionRing + Algebraic + ApproxEq<N>> DVec<N> {
    /// Computes the canonical basis for the given dimension. A canonical basis is a set of
    /// vectors, mutually orthogonal, with all its component equal to 0.0 exept one which is equal
    /// to 1.0.
    pub fn canonical_basis_with_dim(dim: uint) -> ~[DVec<N>] {
        let mut res : ~[DVec<N>] = ~[];

        for i in range(0u, dim) {
            let mut basis_element : DVec<N> = zero_vec_with_dim(dim);

            basis_element.at[i] = One::one();

            res.push(basis_element);
        }

        res
    }

    /// Computes a basis of the space orthogonal to the vector. If the input vector is of dimension
    /// `n`, this will return `n - 1` vectors.
    pub fn orthogonal_subspace_basis(&self) -> ~[DVec<N>] {
        // compute the basis of the orthogonal subspace using Gram-Schmidt
        // orthogonalization algorithm
        let     dim              = self.at.len();
        let mut res : ~[DVec<N>] = ~[];

        for i in range(0u, dim) {
            let mut basis_element : DVec<N> = zero_vec_with_dim(self.at.len());

            basis_element.at[i] = One::one();

            if res.len() == dim - 1 {
                break;
            }

            let mut elt = basis_element.clone();

            elt = elt - self.scalar_mul(&basis_element.dot(self));

            for v in res.iter() {
                elt = elt - v.scalar_mul(&elt.dot(v))
            };

            if !elt.sqnorm().approx_eq(&Zero::zero()) {
                res.push(elt.normalized());
            }
        }

        assert!(res.len() == dim - 1);

        res
    }
}

impl<N: Add<N,N>> Add<DVec<N>, DVec<N>> for DVec<N> {
    #[inline]
    fn add(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.at.len() == other.at.len());
        DVec {
            at: self.at.iter().zip(other.at.iter()).transform(|(a, b)| *a + *b).collect()
        }
    }
}

impl<N: Sub<N,N>> Sub<DVec<N>, DVec<N>> for DVec<N> {
    #[inline]
    fn sub(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.at.len() == other.at.len());
        DVec {
            at: self.at.iter().zip(other.at.iter()).transform(|(a, b)| *a - *b).collect()
        }
    }
}

impl<N: Neg<N>> Neg<DVec<N>> for DVec<N> {
    #[inline]
    fn neg(&self) -> DVec<N> {
        DVec { at: self.at.iter().transform(|a| -a).collect() }
    }
}

impl<N: Ring> Dot<N> for DVec<N> {
    #[inline]
    fn dot(&self, other: &DVec<N>) -> N {
        assert!(self.at.len() == other.at.len());

        let mut res = Zero::zero::<N>();

        for i in range(0u, self.at.len()) {
            res = res + self.at[i] * other.at[i];
        }

        res
    } 
}

impl<N: Ring> SubDot<N> for DVec<N> {
    #[inline]
    fn sub_dot(&self, a: &DVec<N>, b: &DVec<N>) -> N {
        let mut res = Zero::zero::<N>();

        for i in range(0u, self.at.len()) {
            res = res + (self.at[i] - a.at[i]) * b.at[i];
        }

        res
    } 
}

impl<N: Mul<N, N>> ScalarMul<N> for DVec<N> {
    #[inline]
    fn scalar_mul(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().transform(|a| a * *s).collect() }
    }

    #[inline]
    fn scalar_mul_inplace(&mut self, s: &N) {
        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] * *s;
        }
    }
}


impl<N: Div<N, N>> ScalarDiv<N> for DVec<N> {
    #[inline]
    fn scalar_div(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().transform(|a| a / *s).collect() }
    }

    #[inline]
    fn scalar_div_inplace(&mut self, s: &N) {
        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] / *s;
        }
    }
}

impl<N: Add<N, N>> ScalarAdd<N> for DVec<N> {
    #[inline]
    fn scalar_add(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().transform(|a| a + *s).collect() }
    }

    #[inline]
    fn scalar_add_inplace(&mut self, s: &N) {
        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] + *s;
        }
    }
}

impl<N: Sub<N, N>> ScalarSub<N> for DVec<N> {
    #[inline]
    fn scalar_sub(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().transform(|a| a - *s).collect() }
    }

    #[inline]
    fn scalar_sub_inplace(&mut self, s: &N) {
        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] - *s;
        }
    }
}

impl<N: Add<N, N> + Neg<N> + Clone> Translation<DVec<N>> for DVec<N> {
    #[inline]
    fn translation(&self) -> DVec<N> {
        self.clone()
    }

    #[inline]
    fn inv_translation(&self) -> DVec<N> {
        -self
    }

    #[inline]
    fn translate_by(&mut self, t: &DVec<N>) {
        *self = *self + *t;
    }
}

impl<N: Add<N, N> + Neg<N> + Clone> Translatable<DVec<N>, DVec<N>> for DVec<N> {
    #[inline]
    fn translated(&self, t: &DVec<N>) -> DVec<N> {
        self + *t
    }
}

impl<N: DivisionRing + Algebraic + Clone> Norm<N> for DVec<N> {
    #[inline]
    fn sqnorm(&self) -> N {
        self.dot(self)
    }

    #[inline]
    fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    #[inline]
    fn normalized(&self) -> DVec<N> {
        let mut res : DVec<N> = self.clone();

        res.normalize();

        res
    }

    #[inline]
    fn normalize(&mut self) -> N {
        let l = self.norm();

        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] / l;
        }

        l
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DVec<N> {
    #[inline]
    fn approx_epsilon() -> N {
        ApproxEq::approx_epsilon::<N, N>()
    }

    #[inline]
    fn approx_eq(&self, other: &DVec<N>) -> bool {
        let mut zip = self.at.iter().zip(other.at.iter());

        do zip.all |(a, b)| {
            a.approx_eq(b)
        }
    }

    #[inline]
    fn approx_eq_eps(&self, other: &DVec<N>, epsilon: &N) -> bool {
        let mut zip = self.at.iter().zip(other.at.iter());

        do zip.all |(a, b)| {
            a.approx_eq_eps(b, epsilon)
        }
    }
}
