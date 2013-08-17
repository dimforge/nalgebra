use std::num::{One, Zero};
use std::vec::from_elem;
use std::cmp::ApproxEq;
use traits::inv::Inv;
use traits::ring::DivisionRing;
use traits::transpose::Transpose;
use traits::rlmul::{RMul, LMul};
use dvec::{DVec, zero_vec_with_dim};

/// Square matrix with a dimension unknown at compile-time.
#[deriving(Eq, ToStr, Clone)]
pub struct DMat<N> {
    priv dim: uint, // FIXME: handle more than just square matrices
    priv mij: ~[N]
}

/// Builds a matrix filled with zeros.
/// 
/// # Arguments
///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
///   components.
#[inline]
pub fn zero_mat_with_dim<N: Zero + Clone>(dim: uint) -> DMat<N> {
    DMat { dim: dim, mij: from_elem(dim * dim, Zero::zero()) }
}

/// Tests if all components of the matrix are zeroes.
#[inline]
pub fn is_zero_mat<N: Zero>(mat: &DMat<N>) -> bool {
    mat.mij.iter().all(|e| e.is_zero())
}

/// Builds an identity matrix.
/// 
/// # Arguments
///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
///   components.
#[inline]
pub fn one_mat_with_dim<N: Clone + One + Zero>(dim: uint) -> DMat<N> {
    let mut res = zero_mat_with_dim(dim);
    let     _1  = One::one::<N>();

    for i in range(0u, dim) {
        res.set(i, i, &_1);
    }

    res
}

impl<N: Clone> DMat<N> {
    #[inline]
    fn offset(&self, i: uint, j: uint) -> uint {
        i * self.dim + j
    }

    /// Changes the value of a component of the matrix.
    ///
    /// # Arguments
    ///   * `i` - 0-based index of the line to be changed
    ///   * `j` - 0-based index of the column to be changed
    #[inline]
    pub fn set(&mut self, i: uint, j: uint, t: &N) {
        assert!(i < self.dim);
        assert!(j < self.dim);
        self.mij[self.offset(i, j)] = t.clone()
    }

    /// Reads the value of a component of the matrix.
    ///
    /// # Arguments
    ///   * `i` - 0-based index of the line to be read
    ///   * `j` - 0-based index of the column to be read
    #[inline]
    pub fn at(&self, i: uint, j: uint) -> N {
        assert!(i < self.dim);
        assert!(j < self.dim);
        self.mij[self.offset(i, j)].clone()
    }
}

impl<N: Clone> Index<(uint, uint), N> for DMat<N> {
    #[inline]
    fn index(&self, &(i, j): &(uint, uint)) -> N {
        self.at(i, j)
    }
}

impl<N: Clone + Mul<N, N> + Add<N, N> + Zero>
Mul<DMat<N>, DMat<N>> for DMat<N> {
    fn mul(&self, other: &DMat<N>) -> DMat<N> {
        assert!(self.dim == other.dim);

        let     dim = self.dim;
        let mut res = zero_mat_with_dim(dim);

        for i in range(0u, dim) {
            for j in range(0u, dim) {
                let mut acc = Zero::zero::<N>();

                for k in range(0u, dim) {
                    acc = acc + self.at(i, k) * other.at(k, j);
                }

                res.set(i, j, &acc);
            }
        }

        res
    }
}

impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
RMul<DVec<N>> for DMat<N> {
    fn rmul(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.dim == other.at.len());

        let     dim           = self.dim;
        let mut res : DVec<N> = zero_vec_with_dim(dim);

        for i in range(0u, dim) {
            for j in range(0u, dim) {
                res.at[i] = res.at[i] + other.at[j] * self.at(i, j);
            }
        }

        res
    }
}

impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
LMul<DVec<N>> for DMat<N> {
    fn lmul(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.dim == other.at.len());

        let     dim           = self.dim;
        let mut res : DVec<N> = zero_vec_with_dim(dim);

        for i in range(0u, dim) {
            for j in range(0u, dim) {
                res.at[i] = res.at[i] + other.at[j] * self.at(j, i);
            }
        }

        res
    }
}

impl<N: Clone + Eq + DivisionRing>
Inv for DMat<N> {
    #[inline]
    fn inverse(&self) -> Option<DMat<N>> {
        let mut res : DMat<N> = self.clone();

        if res.inplace_inverse() {
            Some(res)
        }
        else {
            None
        }
    }

    fn inplace_inverse(&mut self) -> bool {
        let     dim = self.dim;
        let mut res = one_mat_with_dim::<N>(dim);
        let     _0T = Zero::zero::<N>();

        // inversion using Gauss-Jordan elimination
        for k in range(0u, dim) {
            // search a non-zero value on the k-th column
            // FIXME: would it be worth it to spend some more time searching for the
            // max instead?

            let mut n0 = k; // index of a non-zero entry

            while (n0 != dim) {
                if self.at(n0, k) != _0T {
                    break;
                }

                n0 = n0 + 1;
            }

            if n0 == dim { 
                return false
            }

            // swap pivot line
            if n0 != k {
                for j in range(0u, dim) {
                    let off_n0_j = self.offset(n0, j);
                    let off_k_j  = self.offset(k, j);

                    self.mij.swap(off_n0_j, off_k_j);
                    res.mij.swap(off_n0_j, off_k_j);
                }
            }

            let pivot = self.at(k, k);

            for j in range(k, dim) {
                let selfval = &(self.at(k, j) / pivot);
                self.set(k, j, selfval);
            }

            for j in range(0u, dim) {
                let resval  = &(res.at(k, j)   / pivot);
                res.set(k, j, resval);
            }

            for l in range(0u, dim) {
                if l != k {
                    let normalizer = self.at(l, k);

                    for j in range(k, dim) {
                        let selfval = &(self.at(l, j) - self.at(k, j) * normalizer);
                        self.set(l, j, selfval);
                    }

                    for j in range(0u, dim) {
                        let resval  = &(res.at(l, j)   - res.at(k, j)   * normalizer);
                        res.set(l, j, resval);
                    }
                }
            }
        }

        *self = res;

        true
    }
}

impl<N: Clone> Transpose for DMat<N> {
    #[inline]
    fn transposed(&self) -> DMat<N> {
        let mut res = self.clone();

        res.transpose();

        res
    }

    fn transpose(&mut self) {
        let dim = self.dim;

        for i in range(1u, dim) {
            for j in range(0u, dim - 1) {
                let off_i_j = self.offset(i, j);
                let off_j_i = self.offset(j, i);

                self.mij.swap(off_i_j, off_j_i);
            }
        }
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DMat<N> {
    #[inline]
    fn approx_epsilon() -> N {
        ApproxEq::approx_epsilon::<N, N>()
    }

    #[inline]
    fn approx_eq(&self, other: &DMat<N>) -> bool {
        let mut zip = self.mij.iter().zip(other.mij.iter());

        do zip.all |(a, b)| {
            a.approx_eq(b)
        }
    }

    #[inline]
    fn approx_eq_eps(&self, other: &DMat<N>, epsilon: &N) -> bool {
        let mut zip = self.mij.iter().zip(other.mij.iter());

        do zip.all |(a, b)| {
            a.approx_eq_eps(b, epsilon)
        }
    }
}
