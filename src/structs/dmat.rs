//! Matrix with dimensions unknown at compile-time.

#![allow(missing_docs)] // we hide doc to not have to document the $trhs double dispatch trait.

use std::cmp;
use std::rand::Rand;
use std::rand;
use traits::operations::ApproxEq;
use std::mem;
use structs::dvec::DVec;
use traits::operations::{Inv, Transpose, Mean, Cov};
use traits::structure::{Cast, ColSlice, RowSlice, Diag, Eye, Indexable, Shape, Zero, One, BaseNum};
use std::fmt::{Show, Formatter, Result};


/// Matrix with dimensions unknown at compile-time.
#[deriving(Eq, PartialEq, Clone)]
pub struct DMat<N> {
    nrows: uint,
    ncols: uint,
    mij:   Vec<N>
}

impl<N> DMat<N> {
    /// Creates an uninitialized matrix.
    #[inline]
    pub unsafe fn new_uninitialized(nrows: uint, ncols: uint) -> DMat<N> {
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
    pub fn new_zeros(nrows: uint, ncols: uint) -> DMat<N> {
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
    pub fn new_random(nrows: uint, ncols: uint) -> DMat<N> {
        DMat::from_fn(nrows, ncols, |_, _| rand::random())
    }
}

impl<N: One + Clone + Copy> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn new_ones(nrows: uint, ncols: uint) -> DMat<N> {
        DMat::from_elem(nrows, ncols, ::one())
    }
}

impl<N: Clone + Copy> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn from_elem(nrows: uint, ncols: uint, val: N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   Vec::from_elem(nrows * ncols, val)
        }
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in row-major order.
    /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_row_vec(nrows: uint, ncols: uint, vec: &[N]) -> DMat<N> {
        let mut res = DMat::from_col_vec(ncols, nrows, vec);

        // we transpose because the buffer is row_major
        res.transpose();

        res
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in column-major order.
    /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_col_vec(nrows: uint, ncols: uint, vec: &[N]) -> DMat<N> {
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
    pub fn from_fn(nrows: uint, ncols: uint, f: |uint, uint| -> N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   Vec::from_fn(nrows * ncols, |i| { let m = i / nrows; f(i - m * nrows, m) })
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
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn to_vec(self) -> Vec<N> {
        self.mij
    }

    /// Gets a reference to this matrix data.
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn as_vec<'r>(&'r self) -> &'r [N] {
        self.mij.as_slice()
    }

    /// Gets a mutable reference to this matrix data.
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn as_mut_vec<'r>(&'r mut self) -> &'r mut [N] {
         self.mij.as_mut_slice()
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
    fn new_identity(dim: uint) -> DMat<N> {
        let mut res = DMat::new_zeros(dim, dim);

        for i in range(0u, dim) {
            let _1: N  = ::one();
            res[(i, i)]  = _1;
        }

        res
    }
}

impl<N> DMat<N> {
    #[inline(always)]
    fn offset(&self, i: uint, j: uint) -> uint {
        i + j * self.nrows
    }

}

impl<N: Copy> Indexable<(uint, uint), N> for DMat<N> {
    /// Changes the value of a component of the matrix.
    ///
    /// # Arguments
    /// * `rowcol` - 0-based tuple (row, col)  to be changed
    #[inline]
    fn set(&mut self, rowcol: (uint,  uint), val: N) {
        let (row, col) = rowcol;
        assert!(row < self.nrows);
        assert!(col < self.ncols);

        let offset = self.offset(row, col);
        self.mij[offset] = val
    }

    /// Just like `set` without bounds checking.
    #[inline]
    unsafe fn unsafe_set(&mut self, rowcol: (uint, uint), val: N) {
        let (row, col) = rowcol;
        let offset = self.offset(row, col);
        *self.mij.as_mut_slice().unsafe_mut(offset) = val
    }

    /// Reads the value of a component of the matrix.
    ///
    /// # Arguments
    /// * `rowcol` - 0-based tuple (row, col)  to be read
    #[inline]
    fn at(&self, rowcol: (uint, uint)) -> N {
        let (row, col) = rowcol;
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        unsafe { self.unsafe_at((row, col)) }
    }

    /// Just like `at` without bounds checking.
    #[inline]
    unsafe fn unsafe_at(&self, rowcol: (uint,  uint)) -> N {
        let (row, col) = rowcol;

        *self.mij.as_slice().unsafe_get(self.offset(row, col))
    }

    #[inline]
    fn swap(&mut self, rowcol1: (uint, uint), rowcol2: (uint, uint)) {
        let (row1, col1) = rowcol1;
        let (row2, col2) = rowcol2;
        let offset1 = self.offset(row1, col1);
        let offset2 = self.offset(row2, col2);
        let count = self.mij.len();
        assert!(offset1 < count);
        assert!(offset2 < count);
        self.mij.as_mut_slice().swap(offset1, offset2);
    }

}

impl<N> Shape<(uint, uint), N> for DMat<N> {
    #[inline]
    fn shape(&self) -> (uint, uint) {
        (self.nrows, self.ncols)
    }
}

impl<N> Index<(uint, uint), N> for DMat<N> {
    fn index(&self, &(i, j): &(uint, uint)) -> &N {
        assert!(i < self.nrows);
        assert!(j < self.ncols);

        unsafe {
            self.mij.as_slice().unsafe_get(self.offset(i, j))
        }
    }
}

impl<N> IndexMut<(uint, uint), N> for DMat<N> {
    fn index_mut(&mut self, &(i, j): &(uint, uint)) -> &mut N {
        assert!(i < self.nrows);
        assert!(j < self.ncols);

        let offset = self.offset(i, j);

        unsafe {
            self.mij.as_mut_slice().unsafe_mut(offset)
        }
    }
}

impl<N: Copy + Mul<N, N> + Add<N, N> + Zero> Mul<DMat<N>, DMat<N>> for DMat<N> {
    fn mul(self, right: DMat<N>) -> DMat<N> {
        assert!(self.ncols == right.nrows);

        let mut res = unsafe { DMat::new_uninitialized(self.nrows, right.ncols) };

        for i in range(0u, self.nrows) {
            for j in range(0u, right.ncols) {
                let mut acc: N = ::zero();

                unsafe {
                    for k in range(0u, self.ncols) {
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

impl<N: Copy + Add<N, N> + Mul<N, N> + Zero> Mul<DVec<N>, DVec<N>> for DMat<N> {
    fn mul(self, right: DVec<N>) -> DVec<N> {
        assert!(self.ncols == right.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(self.nrows) };

        for i in range(0u, self.nrows) {
            let mut acc: N = ::zero();

            for j in range(0u, self.ncols) {
                unsafe {
                    acc = acc + self.unsafe_at((i, j)) * right.unsafe_at(j);
                }
            }

            res.at[i] = acc;
        }

        res
    }
}


impl<N: Copy + Add<N, N> + Mul<N, N> + Zero> Mul<DMat<N>, DVec<N>> for DVec<N> {
    fn mul(self, right: DMat<N>) -> DVec<N> {
        assert!(right.nrows == self.at.len());

        let mut res : DVec<N> = unsafe { DVec::new_uninitialized(right.ncols) };

        for i in range(0u, right.ncols) {
            let mut acc: N = ::zero();

            for j in range(0u, right.nrows) {
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
    fn inv_cpy(&self) -> Option<DMat<N>> {
        let mut res: DMat<N> = self.clone();
        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    fn inv(&mut self) -> bool {
        assert!(self.nrows == self.ncols);

        let dim              = self.nrows;
        let mut res: DMat<N> = Eye::new_identity(dim);

        // inversion using Gauss-Jordan elimination
        for k in range(0u, dim) {
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
                for j in range(0u, dim) {
                    let off_n0_j = self.offset(n0, j);
                    let off_k_j  = self.offset(k, j);

                    self.mij.as_mut_slice().swap(off_n0_j, off_k_j);
                    res.mij.as_mut_slice().swap(off_n0_j, off_k_j);
                }
            }

            unsafe {
                let pivot = self.unsafe_at((k, k));

                for j in range(k, dim) {
                    let selfval = self.unsafe_at((k, j)) / pivot;
                    self.unsafe_set((k, j), selfval);
                }

                for j in range(0u, dim) {
                    let resval = res.unsafe_at((k, j)) / pivot;
                    res.unsafe_set((k, j), resval);
                }

                for l in range(0u, dim) {
                    if l != k {
                        let normalizer = self.unsafe_at((l, k));

                        for j in range(k, dim) {
                            let selfval = self.unsafe_at((l, j)) - self.unsafe_at((k, j)) * normalizer;
                            self.unsafe_set((l, j), selfval);
                        }

                        for j in range(0u, dim) {
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
    fn transpose_cpy(&self) -> DMat<N> {
        if self.nrows == self.ncols {
            let mut res = self.clone();

            res.transpose();

            res
        }
        else {
            let mut res = unsafe { DMat::new_uninitialized(self.ncols, self.nrows) };

            for i in range(0u, self.nrows) {
                for j in range(0u, self.ncols) {
                    unsafe {
                        res.unsafe_set((j, i), self.unsafe_at((i, j)))
                    }
                }
            }

            res
        }
    }

    #[inline]
    fn transpose(&mut self) {
        if self.nrows == self.ncols {
            for i in range(1u, self.nrows) {
                for j in range(0u, self.ncols - 1) {
                    let off_i_j = self.offset(i, j);
                    let off_j_i = self.offset(j, i);

                    self.mij.as_mut_slice().swap(off_i_j, off_j_i);
                }
            }

            mem::swap(&mut self.nrows, &mut self.ncols);
        }
        else {
            // FIXME: implement a better algorithm which does that in-place.
            *self = Transpose::transpose_cpy(self);
        }
    }
}

impl<N: BaseNum + Cast<f64> + Clone> Mean<DVec<N>> for DMat<N> {
    fn mean(&self) -> DVec<N> {
        let mut res: DVec<N> = DVec::new_zeros(self.ncols);
        let normalizer: N    = Cast::from(1.0f64 / Cast::from(self.nrows));

        for i in range(0u, self.nrows) {
            for j in range(0u, self.ncols) {
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
        for i in range(0u, self.nrows) {
            for j in range(0u, self.ncols) {
                unsafe {
                    centered.unsafe_set((i, j), self.unsafe_at((i, j)) - mean.unsafe_at(j));
                }
            }
        }

        // FIXME: return a triangular matrix?
        let fnormalizer: f64 = Cast::from(self.nrows() - 1);
        let normalizer: N    = Cast::from(fnormalizer);
        // FIXME: this will do 2 allocations for temporaries!
        (Transpose::transpose_cpy(&centered) * centered) / normalizer
    }
}

impl<N: Copy + Clone> ColSlice<DVec<N>> for DMat<N> {
    fn col_slice(&self, col_id :uint, row_start: uint, row_end: uint) -> DVec<N> {
        assert!(col_id < self.ncols);
        assert!(row_start < row_end);
        assert!(row_end <= self.nrows);
        // we can init from slice thanks to the matrix being column major
        let start= self.offset(row_start, col_id);
        let stop = self.offset(row_end, col_id);
        let slice = DVec::from_slice(
            row_end - row_start, self.mij.slice(start, stop));
        slice
    }
}

impl<N: Copy> RowSlice<DVec<N>> for DMat<N> {
    fn row_slice(&self, row_id :uint, col_start: uint, col_end: uint) -> DVec<N> {
        assert!(row_id < self.nrows);
        assert!(col_start < col_end);
        assert!(col_end <= self.ncols);
        let mut slice : DVec<N> = unsafe {
            DVec::new_uninitialized(self.nrows)
        };
        let mut slice_idx = 0u;
        for col_id in range(col_start, col_end) {
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
    fn set_diag(&mut self, diag: &DVec<N>) {
        let smallest_dim = cmp::min(self.nrows, self.ncols);

        assert!(diag.len() == smallest_dim);

        for i in range(0, smallest_dim) {
            unsafe { self.unsafe_set((i, i), diag.unsafe_at(i)) }
        }
    }

    #[inline]
    fn diag(&self) -> DVec<N> {
        let smallest_dim = cmp::min(self.nrows, self.ncols);

        let mut diag: DVec<N> = DVec::new_zeros(smallest_dim);

        for i in range(0, smallest_dim) {
            unsafe { diag.unsafe_set(i, self.unsafe_at((i, i))) }
        }

        diag
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DMat<N> {
    #[inline]
    fn approx_epsilon(_: Option<DMat<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &DMat<N>, epsilon: &N) -> bool {
        let zip = self.mij.iter().zip(other.mij.iter());
        zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
    }
}

impl<N: Show + Copy> Show for DMat<N> {
    fn fmt(&self, form:&mut Formatter) -> Result {
        for i in range(0u, self.nrows()) {
            for j in range(0u, self.ncols()) {
                let _ = write!(form, "{} ", self[(i, j)]);
            }
            let _ = write!(form, "\n");
        }
        write!(form, "\n")
    }
}

impl<N: Copy + Mul<N, N>> Mul<N, DMat<N>> for DMat<N> {
    #[inline]
    fn mul(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij * right;
        }

        res
    }
}

impl<N: Copy + Div<N, N>> Div<N, DMat<N>> for DMat<N> {
    #[inline]
    fn div(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij / right;
        }

        res
    }
}

impl<N: Copy + Add<N, N>> Add<N, DMat<N>> for DMat<N> {
    #[inline]
    fn add(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij + right;
        }

        res
    }
}

impl<N: Copy + Sub<N, N>> Sub<N, DMat<N>> for DMat<N> {
    #[inline]
    fn sub(self, right: N) -> DMat<N> {
        let mut res = self;

        for mij in res.mij.iter_mut() {
            *mij = *mij - right;
        }

        res
    }
}
