use std::rand::Rand;
use std::rand;
use std::num::{One, Zero};
use std::vec;
use std::cmp::ApproxEq;
use std::util;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::rlmul::{RMul, LMul};
use dvec::DVec;

/// Matrix with dimensions unknown at compile-time.
#[deriving(Eq, ToStr, Clone)]
pub struct DMat<N> {
    priv nrows: uint,
    priv ncols: uint,
    priv mij: ~[N]
}

impl<N> DMat<N> {
    /// Creates an uninitialized matrix.
    #[inline]
    pub unsafe fn new_uninitialized(nrows: uint, ncols: uint) -> DMat<N> {
        let mut vec = vec::with_capacity(nrows * ncols);
        vec::raw::set_len(&mut vec, nrows * ncols);

        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec
        }
    }
}

impl<N: Zero + Clone> DMat<N> {
    /// Builds a matrix filled with zeros.
    /// 
    /// # Arguments
    ///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
    ///   components.
    #[inline]
    pub fn new_zeros(nrows: uint, ncols: uint) -> DMat<N> {
        DMat::from_elem(nrows, ncols, Zero::zero())
    }

    /// Tests if all components of the matrix are zeroes.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mij.iter().all(|e| e.is_zero())
    }
}

impl<N: Rand> DMat<N> {
    /// Builds a matrix filled with random values.
    #[inline]
    pub fn new_random(nrows: uint, ncols: uint) -> DMat<N> {
        DMat::from_fn(nrows, ncols, |_, _| rand::random())
    }
}

impl<N: One + Clone> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn new_ones(nrows: uint, ncols: uint) -> DMat<N> {
        DMat::from_elem(nrows, ncols, One::one())
    }
}

impl<N: Clone> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn from_elem(nrows: uint, ncols: uint, val: N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec::from_elem(nrows * ncols, val)
        }
    }
}

impl<N> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline(always)]
    pub fn from_fn(nrows: uint, ncols: uint, f: &fn(uint, uint) -> N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec::from_fn(nrows * ncols, |i| { let m = i % ncols; f(m, m - i * ncols) })
        }
    }

    /// The number of row on the matrix.
    pub fn nrows(&self) -> uint {
        self.nrows
    }

    /// The number of columns on the matrix.
    pub fn ncols(&self) -> uint {
        self.ncols
    }
}

// FIXME: add a function to modify the dimension (to avoid useless allocations)?

impl<N: One + Zero + Clone> DMat<N> {
    /// Builds an identity matrix.
    /// 
    /// # Arguments
    ///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
    ///   components.
    #[inline]
    pub fn new_identity(dim: uint) -> DMat<N> {
        let mut res = DMat::new_zeros(dim, dim);

        for i in range(0u, dim) {
            let _1: N  = One::one();
            res.set(i, i, _1);
        }

        res
    }
}


impl<N: Clone> DMat<N> {
    #[inline]
    fn offset(&self, i: uint, j: uint) -> uint {
        i * self.ncols + j
    }

    /// Changes the value of a component of the matrix.
    ///
    /// # Arguments
    ///   * `row` - 0-based index of the line to be changed
    ///   * `col` - 0-based index of the column to be changed
    #[inline]
    pub fn set(&mut self, row: uint, col: uint, val: N) {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.mij[self.offset(row, col)] = val
    }

    /// Reads the value of a component of the matrix.
    ///
    /// # Arguments
    ///   * `row` - 0-based index of the line to be read
    ///   * `col` - 0-based index of the column to be read
    #[inline]
    pub fn at(&self, row: uint, col: uint) -> N {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.mij[self.offset(row, col)].clone()
    }
}

impl<N: Clone + Mul<N, N> + Add<N, N> + Zero> Mul<DMat<N>, DMat<N>> for DMat<N> {
    fn mul(&self, other: &DMat<N>) -> DMat<N> {
        assert!(self.ncols == other.nrows);

        let mut res = unsafe { DMat::new_uninitialized(self.nrows, other.ncols) };

        for i in range(0u, self.nrows) {
            for j in range(0u, other.ncols) {
                let mut acc: N = Zero::zero();

                for k in range(0u, self.ncols) {
                    acc = acc + self.at(i, k) * other.at(k, j);
                }

                res.set(i, j, acc);
            }
        }

        res
    }
}

impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
RMul<DVec<N>> for DMat<N> {
    fn rmul(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.ncols == other.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(self.nrows) };

        for i in range(0u, self.nrows) {
            let mut acc: N = Zero::zero();

            for j in range(0u, self.ncols) {
                acc = acc + other.at[j] * self.at(i, j);
            }

            res.at[i] = acc;
        }

        res
    }
}

impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
LMul<DVec<N>> for DMat<N> {
    fn lmul(&self, other: &DVec<N>) -> DVec<N> {
        assert!(self.nrows == other.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(self.ncols) };

        for i in range(0u, self.ncols) {
            let mut acc: N = Zero::zero();

            for j in range(0u, self.nrows) {
                acc = acc + other.at[j] * self.at(j, i);
            }

            res.at[i] = acc;
        }

        res
    }
}

impl<N: Clone + Num>
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
        assert!(self.nrows == self.ncols);

        let dim              = self.nrows;
        let mut res: DMat<N> = DMat::new_identity(dim);
        let     _0T: N       = Zero::zero();

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
                let selfval = self.at(k, j) / pivot;
                self.set(k, j, selfval);
            }

            for j in range(0u, dim) {
                let resval = res.at(k, j) / pivot;
                res.set(k, j, resval);
            }

            for l in range(0u, dim) {
                if l != k {
                    let normalizer = self.at(l, k);

                    for j in range(k, dim) {
                        let selfval = self.at(l, j) - self.at(k, j) * normalizer;
                        self.set(l, j, selfval);
                    }

                    for j in range(0u, dim) {
                        let resval = res.at(l, j) - res.at(k, j) * normalizer;
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
        for i in range(1u, self.nrows) {
            for j in range(0u, self.ncols - 1) {
                let off_i_j = self.offset(i, j);
                let off_j_i = self.offset(j, i);

                self.mij.swap(off_i_j, off_j_i);
            }
        }

        util::swap(&mut self.nrows, &mut self.ncols);
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DMat<N> {
    #[inline]
    fn approx_epsilon() -> N {
        fail!("This function cannot work due to a compiler bug.")
        // let res: N = ApproxEq::<N>::approx_epsilon();

        // res
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
