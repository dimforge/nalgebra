use std::num::{Zero, One, Algebraic};
use std::vec;
use std::vec::{VecIterator, VecMutIterator};
use std::cmp::ApproxEq;
use std::iterator::FromIterator;
use traits::iterable::{Iterable, IterableMut};
use traits::translation::Translation;
use traits::scalar_op::{ScalarAdd, ScalarSub};

/// Vector with a dimension unknown at compile-time.
#[deriving(Eq, ToStr, Clone)]
pub struct DVec<N> {
    /// Components of the vector. Contains as much elements as the vector dimension.
    at: ~[N]
}

impl<N: Zero + Clone> DVec<N> {
    /// Builds a vector filled with zeros.
    /// 
    /// # Arguments
    ///   * `dim` - The dimension of the vector.
    #[inline]
    pub fn new_zeros(dim: uint) -> DVec<N> {
        DVec::from_elem(dim, Zero::zero())
    }

    /// Tests if all components of the vector are zeroes.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.at.iter().all(|e| e.is_zero())
    }
}

impl<N: One + Clone> DVec<N> {
    /// Builds a vector filled with ones.
    /// 
    /// # Arguments
    ///   * `dim` - The dimension of the vector.
    #[inline]
    pub fn new_ones(dim: uint) -> DVec<N> {
        DVec::from_elem(dim, One::one())
    }
}

impl<N: Clone> DVec<N> {
    /// Builds a vector filled with a constant.
    #[inline]
    pub fn from_elem(dim: uint, elem: N) -> DVec<N> {
        DVec { at: vec::from_elem(dim, elem) }
    }
}

impl<N: Clone> DVec<N> {
    /// Builds a vector filled with the result of a function.
    #[inline(always)]
    pub fn from_fn(dim: uint, f: &fn(uint) -> N) -> DVec<N> {
        DVec { at: vec::from_fn(dim, |i| f(i)) }
    }
}

impl<N> Container for DVec<N> {
    #[inline]
    fn len(&self) -> uint {
        self.at.len()
    }
}

impl<N> Iterable<N> for DVec<N> {
    #[inline]
    fn iter<'l>(&'l self) -> VecIterator<'l, N> {
        self.at.iter()
    }
}

impl<N> IterableMut<N> for DVec<N> {
    #[inline]
    fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N> {
        self.at.mut_iter()
    }
}

impl<N> FromIterator<N> for DVec<N> {
    #[inline]
    fn from_iterator<I: Iterator<N>>(mut param: &mut I) -> DVec<N> {
        let mut res = DVec { at: ~[] };

        for e in param {
            res.at.push(e)
        }

        res
    }
}

impl<N: Clone + Num + Algebraic + ApproxEq<N>> DVec<N> {
    /// Computes the canonical basis for the given dimension. A canonical basis is a set of
    /// vectors, mutually orthogonal, with all its component equal to 0.0 exept one which is equal
    /// to 1.0.
    pub fn canonical_basis_with_dim(dim: uint) -> ~[DVec<N>] {
        let mut res : ~[DVec<N>] = ~[];

        for i in range(0u, dim) {
            let mut basis_element : DVec<N> = DVec::new_zeros(dim);

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
            let mut basis_element : DVec<N> = DVec::new_zeros(self.at.len());

            basis_element.at[i] = One::one();

            if res.len() == dim - 1 {
                break;
            }

            let mut elt = basis_element.clone();

            elt = elt - self * basis_element.dot(self);

            for v in res.iter() {
                elt = elt - v * elt.dot(v)
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
            at: self.at.iter().zip(other.at.iter()).map(|(a, b)| *a + *b).collect()
        }
    }
}

impl<N: Sub<N,N>> Sub<DVec<N>, DVec<N>> for DVec<N> {
    #[inline]
    fn sub(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.at.len() == other.at.len());
        DVec {
            at: self.at.iter().zip(other.at.iter()).map(|(a, b)| *a - *b).collect()
        }
    }
}

impl<N: Neg<N>> Neg<DVec<N>> for DVec<N> {
    #[inline]
    fn neg(&self) -> DVec<N> {
        DVec { at: self.at.iter().map(|a| -a).collect() }
    }
}

impl<N: Num> DVec<N> {
    /// Will change soon.
    #[inline]
    pub fn dot(&self, other: &DVec<N>) -> N {
        assert!(self.at.len() == other.at.len());

        let mut res: N = Zero::zero();

        for i in range(0u, self.at.len()) {
            res = res + self.at[i] * other.at[i];
        }

        res
    } 

    /// Will change soon.
    #[inline]
    pub fn sub_dot(&self, a: &DVec<N>, b: &DVec<N>) -> N {
        let mut res: N = Zero::zero();

        for i in range(0u, self.at.len()) {
            res = res + (self.at[i] - a.at[i]) * b.at[i];
        }

        res
    } 
}

impl<N: Mul<N, N>> Mul<N, DVec<N>> for DVec<N> {
    #[inline]
    fn mul(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().map(|a| a * *s).collect() }
    }
}


impl<N: Div<N, N>> Div<N, DVec<N>> for DVec<N> {
    #[inline]
    fn div(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().map(|a| a / *s).collect() }
    }
}

impl<N: Add<N, N>> ScalarAdd<N> for DVec<N> {
    #[inline]
    fn scalar_add(&self, s: &N) -> DVec<N> {
        DVec { at: self.at.iter().map(|a| a + *s).collect() }
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
        DVec { at: self.at.iter().map(|a| a - *s).collect() }
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

    #[inline]
    fn translated(&self, t: &DVec<N>) -> DVec<N> {
        self + *t
    }

    #[inline]
    fn set_translation(&mut self, t: DVec<N>) {
        *self = t
    }
}

impl<N: Num + Algebraic + Clone> DVec<N> {
    /// Will change soon.
    #[inline]
    pub fn sqnorm(&self) -> N {
        self.dot(self)
    }

    /// Will change soon.
    #[inline]
    pub fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    /// Will change soon.
    #[inline]
    pub fn normalized(&self) -> DVec<N> {
        let mut res : DVec<N> = self.clone();

        res.normalize();

        res
    }

    /// Will change soon.
    #[inline]
    pub fn normalize(&mut self) -> N {
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
        fail!("Fix me.")
        // let res: N = ApproxEq::<N>::approx_epsilon();

        // res
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
