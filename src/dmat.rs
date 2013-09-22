//! Matrix with dimensions unknown at compile-time.

use std::rand::Rand;
use std::rand;
use std::num::{One, Zero};
use std::vec;
use std::cmp::ApproxEq;
use std::util;
use dvec::{DVec, DVecMulRhs};
use traits::operations::{Inv, Transpose};

/// Matrix with dimensions unknown at compile-time.
#[deriving(Eq, ToStr, Clone)]
pub struct DMat<N> {
    priv nrows: uint,
    priv ncols: uint,
    priv mij: ~[N]
}

/// Trait of object `o` which can be multiplied by a `DMat` `d`: `d * o`.
pub trait DMatMulRhs<N, Res> {
    /// Multiplies a `DMat` by `Self`.
    fn binop(left: &DMat<N>, right: &Self) -> Res;
}

impl<N, Rhs: DMatMulRhs<N, Res>, Res> Mul<Rhs, Res> for DMat<N> {
    #[inline(always)]
    fn mul(&self, other: &Rhs) -> Res {
        DMatMulRhs::binop(self, other)
    }
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
    #[inline]
    pub fn nrows(&self) -> uint {
        self.nrows
    }

    /// The number of columns on the matrix.
    #[inline]
    pub fn ncols(&self) -> uint {
        self.ncols
    }

    /// Transforms this matrix into an array. This consumes the matrix and is O(1).
    #[inline]
    pub fn to_array(self) -> ~[N] {
        self.mij
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

    /// Just like `set` without bounds checking.
    #[inline]
    pub unsafe fn set_fast(&mut self, row: uint, col: uint, val: N) {
        let off = self.offset(row, col);
        *self.mij.unsafe_mut_ref(off) = val
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
        unsafe { self.at_fast(row, col) }
    }

    /// Just like `at` without bounds checking.
    #[inline]
    pub unsafe fn at_fast(&self, row: uint, col: uint) -> N {
        vec::raw::get(self.mij, self.offset(row, col))
    }
}

impl<N: Clone + Mul<N, N> + Add<N, N> + Zero> DMatMulRhs<N, DMat<N>> for DMat<N> {
    fn binop(left: &DMat<N>, right: &DMat<N>) -> DMat<N> {
        assert!(left.ncols == right.nrows);

        let mut res = unsafe { DMat::new_uninitialized(left.nrows, right.ncols) };

        for i in range(0u, left.nrows) {
            for j in range(0u, right.ncols) {
                let mut acc: N = Zero::zero();

                unsafe {
                    for k in range(0u, left.ncols) {
                        acc = acc + left.at_fast(i, k) * right.at_fast(k, j);
                    }

                    res.set_fast(i, j, acc);
                }
            }
        }

        res
    }
}

impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
DMatMulRhs<N, DVec<N>> for DVec<N> {
    fn binop(left: &DMat<N>, right: &DVec<N>) -> DVec<N> {
        assert!(left.ncols == right.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(left.nrows) };

        for i in range(0u, left.nrows) {
            let mut acc: N = Zero::zero();

            for j in range(0u, left.ncols) {
                unsafe {
                    acc = acc + left.at_fast(i, j) * right.at_fast(j);
                }
            }

            res.at[i] = acc;
        }

        res
    }
}


impl<N: Clone + Add<N, N> + Mul<N, N> + Zero>
DVecMulRhs<N, DVec<N>> for DMat<N> {
    fn binop(left: &DVec<N>, right: &DMat<N>) -> DVec<N> {
        assert!(right.nrows == left.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(right.ncols) };

        for i in range(0u, right.ncols) {
            let mut acc: N = Zero::zero();

            for j in range(0u, right.nrows) {
                unsafe {
                    acc = acc + left.at_fast(j) * right.at_fast(j, i);
                }
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
                if unsafe { self.at_fast(n0, k) } != _0T {
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

            unsafe {
                let pivot = self.at_fast(k, k);

                for j in range(k, dim) {
                    let selfval = self.at_fast(k, j) / pivot;
                    self.set_fast(k, j, selfval);
                }

                for j in range(0u, dim) {
                    let resval = res.at_fast(k, j) / pivot;
                    res.set_fast(k, j, resval);
                }

                for l in range(0u, dim) {
                    if l != k {
                        let normalizer = self.at_fast(l, k);

                        for j in range(k, dim) {
                            let selfval = self.at_fast(l, j) - self.at_fast(k, j) * normalizer;
                            self.set_fast(l, j, selfval);
                        }

                        for j in range(0u, dim) {
                            let resval = res.at_fast(l, j) - res.at_fast(k, j) * normalizer;
                            res.set_fast(l, j, resval);
                        }
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
