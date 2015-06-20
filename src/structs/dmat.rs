//! Matrix with dimensions unknown at compile-time.

#![allow(missing_docs)] // we hide doc to not have to document the $trhs double dispatch trait.

use std::{cmp, mem};
use std::iter::repeat;
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};
use std::fmt::{Debug, Display, Formatter, Result};
use rand::{self, Rand};
use num::{Zero, One};
use structs::dvec::DVec;
use traits::operations::{ApproxEq, Inv, Transpose, Mean, Cov};
use traits::structure::{Cast, ColSlice, RowSlice, Diag, DiagMut, Eye, Indexable, Shape, BaseNum};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Matrix with dimensions unknown at compile-time.
#[derive(Eq, PartialEq, Clone)]
pub struct DMat<N> {
    nrows: usize,
    ncols: usize,
    mij:   Vec<N>
}

impl<N> DMat<N> {
    /// Creates an uninitialized matrix.
    #[inline]
    pub unsafe fn new_uninitialized(nrows: usize, ncols: usize) -> DMat<N> {
        let mut vec = Vec::with_capacity(nrows * ncols);
        vec.set_len(nrows * ncols);

        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec
        }
    }
}

impl<N: Zero + Clone + Copy> DMat<N> {
    /// Builds a matrix filled with zeros.
    ///
    /// # Arguments
    ///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
    ///   components.
    #[inline]
    pub fn new_zeros(nrows: usize, ncols: usize) -> DMat<N> {
        DMat::from_elem(nrows, ncols, ::zero())
    }

    /// Tests if all components of the matrix are zeroes.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mij.iter().all(|e| e.is_zero())
    }

    #[inline]
    pub fn reset(&mut self) {
        for mij in self.mij.iter_mut() {
            *mij = ::zero();
        }
    }
}

impl<N: Rand> DMat<N> {
    /// Builds a matrix filled with random values.
    #[inline]
    pub fn new_random(nrows: usize, ncols: usize) -> DMat<N> {
        DMat::from_fn(nrows, ncols, |_, _| rand::random())
    }
}

impl<N: One + Clone + Copy> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn new_ones(nrows: usize, ncols: usize) -> DMat<N> {
        DMat::from_elem(nrows, ncols, ::one())
    }
}

impl<N: Clone + Copy> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn from_elem(nrows: usize, ncols: usize, val: N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   repeat(val).take(nrows * ncols).collect()
        }
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in row-major order.
    /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_row_vec(nrows: usize, ncols: usize, vec: &[N]) -> DMat<N> {
        let mut res = DMat::from_col_vec(ncols, nrows, vec);

        // we transpose because the buffer is row_major
        res.transpose_mut();

        res
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in column-major order.
    /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_col_vec(nrows: usize, ncols: usize, vec: &[N]) -> DMat<N> {
        assert!(nrows * ncols == vec.len());

        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec.to_vec()
        }
    }
}

impl<N> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline(always)]
    pub fn from_fn<F: FnMut(usize, usize) -> N>(nrows: usize, ncols: usize, mut f: F) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   (0..nrows * ncols).map(|i| { let m = i / nrows; f(i - m * nrows, m) }).collect()
        }
    }

    /// The number of row on the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// The number of columns on the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Transforms this matrix isizeo an array. This consumes the matrix and is O(1).
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn to_vec(self) -> Vec<N> {
        self.mij
    }

    /// Gets a reference to this matrix data.
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn as_vec(&self) -> &[N] {
        &self.mij
    }

    /// Gets a mutable reference to this matrix data.
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut [N] {
         &mut self.mij[..]
    }
}

// FIXME: add a function to modify the dimension (to avoid useless allocations)?

impl<N: One + Zero + Clone + Copy> Eye for DMat<N> {
    /// Builds an identity matrix.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
    /// components.
    #[inline]
    fn new_identity(dim: usize) -> DMat<N> {
        let mut res = DMat::new_zeros(dim, dim);

        for i in 0..dim {
            let _1: N  = ::one();
            res[(i, i)]  = _1;
        }

        res
    }
}

impl<N> DMat<N> {
    #[inline(always)]
    fn offset(&self, i: usize, j: usize) -> usize {
        i + j * self.nrows
    }

}

impl<N: Copy> Indexable<(usize, usize), N> for DMat<N> {
    /// Just like `set` without bounds checking.
    #[inline]
    unsafe fn unsafe_set(&mut self, rowcol: (usize, usize), val: N) {
        let (row, col) = rowcol;
        let offset = self.offset(row, col);
        *self.mij[..].get_unchecked_mut(offset) = val
    }

    /// Just like `at` without bounds checking.
    #[inline]
    unsafe fn unsafe_at(&self, rowcol: (usize,  usize)) -> N {
        let (row, col) = rowcol;

        *self.mij.get_unchecked(self.offset(row, col))
    }

    #[inline]
    fn swap(&mut self, rowcol1: (usize, usize), rowcol2: (usize, usize)) {
        let (row1, col1) = rowcol1;
        let (row2, col2) = rowcol2;
        let offset1 = self.offset(row1, col1);
        let offset2 = self.offset(row2, col2);
        let count = self.mij.len();
        assert!(offset1 < count);
        assert!(offset2 < count);
        self.mij[..].swap(offset1, offset2);
    }

}

impl<N> Shape<(usize, usize)> for DMat<N> {
    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl<N> Index<(usize, usize)> for DMat<N> {
    type Output = N;

    fn index(&self, (i, j): (usize, usize)) -> &N {
        assert!(i < self.nrows);
        assert!(j < self.ncols);

        unsafe {
            self.mij.get_unchecked(self.offset(i, j))
        }
    }
}

impl<N> IndexMut<(usize, usize)> for DMat<N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut N {
        assert!(i < self.nrows);
        assert!(j < self.ncols);

        let offset = self.offset(i, j);

        unsafe {
            self.mij[..].get_unchecked_mut(offset)
        }
    }
}

impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn mul(self, right: DMat<N>) -> DMat<N> {
        (&self) * (&right)
    }
}

impl<'a, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<&'a DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn mul(self, right: &'a DMat<N>) -> DMat<N> {
        (&self) * right
    }
}

impl<'a, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<DMat<N>> for &'a DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn mul(self, right: DMat<N>) -> DMat<N> {
        right * self
    }
}

impl<'a, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<&'a DMat<N>> for &'a DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn mul(self, right: &DMat<N>) -> DMat<N> {
        assert!(self.ncols == right.nrows);

        let mut res = unsafe { DMat::new_uninitialized(self.nrows, right.ncols) };

        for i in 0..self.nrows {
            for j in 0..right.ncols {
                let mut acc: N = ::zero();

                unsafe {
                    for k in 0..self.ncols {
                        acc = acc
                            + self.unsafe_at((i, k)) * right.unsafe_at((k, j));
                    }

                    res.unsafe_set((i, j), acc);
                }
            }
        }

        res
    }
}

impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero> Mul<DVec<N>> for DMat<N> {
    type Output = DVec<N>;

    fn mul(self, right: DVec<N>) -> DVec<N> {
        assert!(self.ncols == right.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(self.nrows) };

        for i in 0..self.nrows {
            let mut acc: N = ::zero();

            for j in 0..self.ncols {
                unsafe {
                    acc = acc + self.unsafe_at((i, j)) * right.unsafe_at(j);
                }
            }

            res.at[i] = acc;
        }

        res
    }
}


impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero> Mul<DMat<N>> for DVec<N> {
    type Output = DVec<N>;

    fn mul(self, right: DMat<N>) -> DVec<N> {
        assert!(right.nrows == self.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(right.ncols) };

        for i in 0..right.ncols {
            let mut acc: N = ::zero();

            for j in 0..right.nrows {
                unsafe {
                    acc = acc + self.unsafe_at(j) * right.unsafe_at((j, i));
                }
            }

            res.at[i] = acc;
        }

        res
    }
}

impl<N: BaseNum + Clone> Inv for DMat<N> {
    #[inline]
    fn inv(&self) -> Option<DMat<N>> {
        let mut res: DMat<N> = self.clone();
        if res.inv_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    fn inv_mut(&mut self) -> bool {
        assert!(self.nrows == self.ncols);

        let dim              = self.nrows;
        let mut res: DMat<N> = Eye::new_identity(dim);

        // inversion using Gauss-Jordan elimination
        for k in 0..dim {
            // search a non-zero value on the k-th column
            // FIXME: would it be worth it to spend some more time searching for the
            // max instead?

            let mut n0 = k; // index of a non-zero entry

            while n0 != dim {
                if unsafe { self.unsafe_at((n0, k)) } != ::zero() {
                    break;
                }

                n0 = n0 + 1;
            }

            if n0 == dim {
                return false
            }

            // swap pivot line
            if n0 != k {
                for j in 0..dim {
                    let off_n0_j = self.offset(n0, j);
                    let off_k_j  = self.offset(k, j);

                    self.mij[..].swap(off_n0_j, off_k_j);
                    res.mij[..].swap(off_n0_j, off_k_j);
                }
            }

            unsafe {
                let pivot = self.unsafe_at((k, k));

                for j in k..dim {
                    let selfval = self.unsafe_at((k, j)) / pivot;
                    self.unsafe_set((k, j), selfval);
                }

                for j in 0..dim {
                    let resval = res.unsafe_at((k, j)) / pivot;
                    res.unsafe_set((k, j), resval);
                }

                for l in 0..dim {
                    if l != k {
                        let normalizer = self.unsafe_at((l, k));

                        for j in k..dim {
                            let selfval = self.unsafe_at((l, j)) - self.unsafe_at((k, j)) * normalizer;
                            self.unsafe_set((l, j), selfval);
                        }

                        for j in 0..dim {
                            let resval = res.unsafe_at((l, j)) - res.unsafe_at((k, j)) * normalizer;
                            res.unsafe_set((l, j), resval);
                        }
                    }
                }
            }
        }

        *self = res;

        true
    }
}

impl<N: Clone + Copy> Transpose for DMat<N> {
    #[inline]
    fn transpose(&self) -> DMat<N> {
        if self.nrows == self.ncols {
            let mut res = self.clone();

            res.transpose_mut();

            res
        }
        else {
            let mut res = unsafe { DMat::new_uninitialized(self.ncols, self.nrows) };

            for i in 0..self.nrows {
                for j in 0..self.ncols {
                    unsafe {
                        res.unsafe_set((j, i), self.unsafe_at((i, j)))
                    }
                }
            }

            res
        }
    }

    #[inline]
    fn transpose_mut(&mut self) {
        if self.nrows == self.ncols {
            for i in 1..self.nrows {
                for j in 0..self.ncols - 1 {
                    let off_i_j = self.offset(i, j);
                    let off_j_i = self.offset(j, i);

                    self.mij[..].swap(off_i_j, off_j_i);
                }
            }

            mem::swap(&mut self.nrows, &mut self.ncols);
        }
        else {
            // FIXME: implement a better algorithm which does that in-place.
            *self = Transpose::transpose(self);
        }
    }
}

impl<N: BaseNum + Cast<f64> + Clone> Mean<DVec<N>> for DMat<N> {
    fn mean(&self) -> DVec<N> {
        let mut res: DVec<N> = DVec::new_zeros(self.ncols);
        let normalizer: N    = Cast::from(1.0f64 / self.nrows as f64);

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                unsafe {
                    let acc = res.unsafe_at(j) + self.unsafe_at((i, j)) * normalizer;
                    res.unsafe_set(j, acc);
                }
            }
        }

        res
    }
}

impl<N: BaseNum + Cast<f64> + Clone> Cov<DMat<N>> for DMat<N> {
    // FIXME: this could be heavily optimized, removing all temporaries by merging loops.
    fn cov(&self) -> DMat<N> {
        assert!(self.nrows > 1);

        let mut centered = unsafe { DMat::new_uninitialized(self.nrows, self.ncols) };
        let mean = self.mean();

        // FIXME: use the rows iterator when available
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                unsafe {
                    centered.unsafe_set((i, j), self.unsafe_at((i, j)) - mean.unsafe_at(j));
                }
            }
        }

        // FIXME: return a triangular matrix?
        let fnormalizer: f64 = Cast::from(self.nrows() - 1);
        let normalizer: N    = Cast::from(fnormalizer);
        // FIXME: this will do 2 allocations for temporaries!
        (Transpose::transpose(&centered) * centered) / normalizer
    }
}

impl<N: Copy + Clone> ColSlice<DVec<N>> for DMat<N> {
    fn col_slice(&self, col_id :usize, row_start: usize, row_end: usize) -> DVec<N> {
        assert!(col_id < self.ncols);
        assert!(row_start < row_end);
        assert!(row_end <= self.nrows);
        // we can init from slice thanks to the matrix being column major
        let start= self.offset(row_start, col_id);
        let stop = self.offset(row_end, col_id);
        let slice = DVec::from_slice(row_end - row_start, &self.mij[start .. stop]);
        slice
    }
}

impl<N: Copy> RowSlice<DVec<N>> for DMat<N> {
    fn row_slice(&self, row_id :usize, col_start: usize, col_end: usize) -> DVec<N> {
        assert!(row_id < self.nrows);
        assert!(col_start < col_end);
        assert!(col_end <= self.ncols);
        let mut slice : DVec<N> = unsafe {
            DVec::new_uninitialized(self.nrows)
        };
        let mut slice_idx = 0;
        for col_id in col_start..col_end {
            unsafe {
                slice.unsafe_set(slice_idx, self.unsafe_at((row_id, col_id)));
            }
            slice_idx += 1;
        }
        slice
    }
}

impl<N: Copy + Clone + Zero>  Diag<DVec<N>> for DMat<N> {
    #[inline]
    fn from_diag(diag: &DVec<N>) -> DMat<N> {
        let mut res = DMat::new_zeros(diag.len(), diag.len());

        res.set_diag(diag);

        res
    }

    #[inline]
    fn diag(&self) -> DVec<N> {
        let smallest_dim = cmp::min(self.nrows, self.ncols);

        let mut diag: DVec<N> = DVec::new_zeros(smallest_dim);

        for i in 0..smallest_dim {
            unsafe { diag.unsafe_set(i, self.unsafe_at((i, i))) }
        }

        diag
    }
}

impl<N: Copy + Clone + Zero>  DiagMut<DVec<N>> for DMat<N> {
    #[inline]
    fn set_diag(&mut self, diag: &DVec<N>) {
        let smallest_dim = cmp::min(self.nrows, self.ncols);

        assert!(diag.len() == smallest_dim);

        for i in 0..smallest_dim {
            unsafe { self.unsafe_set((i, i), diag.unsafe_at(i)) }
        }
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DMat<N> {
    #[inline]
    fn approx_epsilon(_: Option<DMat<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_ulps(_: Option<DMat<N>>) -> u32 {
        ApproxEq::approx_ulps(None::<N>)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &DMat<N>, epsilon: &N) -> bool {
        let mut zip = self.mij.iter().zip(other.mij.iter());
        zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
    }

    #[inline]
    fn approx_eq_ulps(&self, other: &DMat<N>, ulps: u32) -> bool {
        let mut zip = self.mij.iter().zip(other.mij.iter());
        zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
    }
}

impl<N: Debug + Copy + Display> Debug for DMat<N> {
    fn fmt(&self, form:&mut Formatter) -> Result {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let _ = write!(form, "{} ", self[(i, j)]);
            }
            let _ = write!(form, "\n");
        }
        write!(form, "\n")
    }
}

impl<N: Copy + Mul<N, Output = N>> Mul<N> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn mul(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij * right;
        }

        res
    }
}

impl<N: Copy + Div<N, Output = N>> Div<N> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn div(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij / right;
        }

        res
    }
}

impl<N: Copy + Add<N, Output = N>> Add<N> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn add(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij + right;
        }

        res
    }
}

impl<N: Copy + Add<N, Output = N>> Add<DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn add(self, right: DMat<N>) -> DMat<N> {
        self + (&right)
    }
}

impl<'a, N: Copy + Add<N, Output = N>> Add<DMat<N>> for &'a DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn add(self, right: DMat<N>) -> DMat<N> {
        right + self
    }
}

impl<'a, N: Copy + Add<N, Output = N>> Add<&'a DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn add(self, right: &'a DMat<N>) -> DMat<N> {
        assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                "Unable to add matrices with different dimensions.");

        let mut res = self;

        for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
            *mij = *mij + *right_ij;
        }

        res
    }
}

impl<N: Copy + Sub<N, Output = N>> Sub<N> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn sub(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij - right;
        }

        res
    }
}

impl<N: Copy + Sub<N, Output = N>> Sub<DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn sub(self, right: DMat<N>) -> DMat<N> {
        self - (&right)
    }
}

impl<'a, N: Copy + Sub<N, Output = N>> Sub<DMat<N>> for &'a DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn sub(self, right: DMat<N>) -> DMat<N> {
        right - self
    }
}

impl<'a, N: Copy + Sub<N, Output = N>> Sub<&'a DMat<N>> for DMat<N> {
    type Output = DMat<N>;

    #[inline]
    fn sub(self, right: &'a DMat<N>) -> DMat<N> {
        assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                "Unable to subtract matrices with different dimensions.");

        let mut res = self;

        for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
            *mij = *mij - *right_ij;
        }

        res
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary> Arbitrary for DMat<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> DMat<N> {
        DMat::from_fn(
            Arbitrary::arbitrary(g), Arbitrary::arbitrary(g),
            |_, _| Arbitrary::arbitrary(g)
        )
    }
}
