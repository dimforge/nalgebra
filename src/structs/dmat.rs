//! Matrix with dimensions unknown at compile-time.

#![allow(missing_docs)] // we hide doc to not have to document the $trhs double dispatch trait.

use std::cmp;
use std::rand::Rand;
use std::rand;
use std::num::{One, Zero, Num};
use traits::operations::ApproxEq;
use std::mem;
use structs::dvec::{DVec, DVecMulRhs};
use traits::operations::{Inv, Transpose, Mean, Cov};
use traits::structure::{Cast, ColSlice, RowSlice, Diag, Eye, Indexable, Shape};
use std::fmt::{Show, Formatter, Result};


/// Matrix with dimensions unknown at compile-time.
#[deriving(Eq, PartialEq, Clone)]
pub struct DMat<N> {
    nrows: uint,
    ncols: uint,
    mij:   Vec<N>
}

double_dispatch_binop_decl_trait!(DMat, DMatMulRhs)
double_dispatch_binop_decl_trait!(DMat, DMatDivRhs)
double_dispatch_binop_decl_trait!(DMat, DMatAddRhs)
double_dispatch_binop_decl_trait!(DMat, DMatSubRhs)

mul_redispatch_impl!(DMat, DMatMulRhs)
div_redispatch_impl!(DMat, DMatDivRhs)
add_redispatch_impl!(DMat, DMatAddRhs)
sub_redispatch_impl!(DMat, DMatSubRhs)

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

    #[inline]
    pub fn reset(&mut self) {
        for mij in self.mij.iter_mut() {
            *mij = Zero::zero();
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

impl<N: One + Zero + Clone> Eye for DMat<N> {
    /// Builds an identity matrix.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
    /// components.
    #[inline]
    fn new_identity(dim: uint) -> DMat<N> {
        let mut res = DMat::new_zeros(dim, dim);

        for i in range(0u, dim) {
            let _1: N  = One::one();
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

impl<N: Clone> Indexable<(uint, uint), N> for DMat<N> {
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
        (*self.mij.as_slice().unsafe_get(self.offset(row, col))).clone()
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

impl<N: Clone + Mul<N, N> + Add<N, N> + Zero> DMatMulRhs<N, DMat<N>> for DMat<N> {
    fn binop(left: &DMat<N>, right: &DMat<N>) -> DMat<N> {
        assert!(left.ncols == right.nrows);

        let mut res = unsafe { DMat::new_uninitialized(left.nrows, right.ncols) };

        for i in range(0u, left.nrows) {
            for j in range(0u, right.ncols) {
                let mut acc: N = Zero::zero();

                unsafe {
                    for k in range(0u, left.ncols) {
                        acc = acc
                            + left.unsafe_at((i, k)) * right.unsafe_at((k, j));
                    }

                    res.unsafe_set((i, j), acc);
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
                    acc = acc + left.unsafe_at((i, j)) * right.unsafe_at(j);
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
                    acc = acc + left.unsafe_at(j) * right.unsafe_at((j, i));
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
    fn inv_cpy(m: &DMat<N>) -> Option<DMat<N>> {
        let mut res : DMat<N> = m.clone();

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
                if unsafe { self.unsafe_at((n0, k)) } != Zero::zero() {
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

impl<N: Clone> Transpose for DMat<N> {
    #[inline]
    fn transpose_cpy(m: &DMat<N>) -> DMat<N> {
        if m.nrows == m.ncols {
            let mut res = m.clone();

            res.transpose();

            res
        }
        else {
            let mut res = unsafe { DMat::new_uninitialized(m.ncols, m.nrows) };

            for i in range(0u, m.nrows) {
                for j in range(0u, m.ncols) {
                    unsafe {
                        res.unsafe_set((j, i), m.unsafe_at((i, j)))
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

impl<N: Num + Cast<f64> + Clone> Mean<DVec<N>> for DMat<N> {
    fn mean(m: &DMat<N>) -> DVec<N> {
        let mut res: DVec<N> = DVec::new_zeros(m.ncols);
        let normalizer: N    = Cast::from(1.0f64 / Cast::from(m.nrows));

        for i in range(0u, m.nrows) {
            for j in range(0u, m.ncols) {
                unsafe {
                    let acc = res.unsafe_at(j) + m.unsafe_at((i, j)) * normalizer;
                    res.unsafe_set(j, acc);
                }
            }
        }

        res
    }
}

impl<N: Clone + Num + Cast<f64> + DMatDivRhs<N, DMat<N>>> Cov<DMat<N>> for DMat<N> {
    // FIXME: this could be heavily optimized, removing all temporaries by merging loops.
    fn cov(m: &DMat<N>) -> DMat<N> {
        assert!(m.nrows > 1);

        let mut centered = unsafe { DMat::new_uninitialized(m.nrows, m.ncols) };
        let mean = Mean::mean(m);

        // FIXME: use the rows iterator when available
        for i in range(0u, m.nrows) {
            for j in range(0u, m.ncols) {
                unsafe {
                    centered.unsafe_set((i, j), m.unsafe_at((i, j)) - mean.unsafe_at(j));
                }
            }
        }

        // FIXME: return a triangular matrix?
        let fnormalizer: f64 = Cast::from(m.nrows() - 1);
        let normalizer: N    = Cast::from(fnormalizer);
        // FIXME: this will do 2 allocations for temporaries!
        (Transpose::transpose_cpy(&centered) * centered) / normalizer
    }
}

impl<N: Clone> ColSlice<DVec<N>> for DMat<N> {
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

impl<N: Clone> RowSlice<DVec<N>> for DMat<N> {
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

impl<N: Clone + Zero>  Diag<DVec<N>> for DMat<N> {
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
    fn approx_eq(a: &DMat<N>, b: &DMat<N>) -> bool {
        let mut zip = a.mij.iter().zip(b.mij.iter());

        zip.all(|(a, b)| ApproxEq::approx_eq(a, b))
    }

    #[inline]
    fn approx_eq_eps(a: &DMat<N>, b: &DMat<N>, epsilon: &N) -> bool {
        let mut zip = a.mij.iter().zip(b.mij.iter());

        zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
    }
}

impl<N: Show + Clone> Show for DMat<N> {
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

macro_rules! scalar_mul_impl (
    ($n: ident) => (
        impl DMatMulRhs<$n, DMat<$n>> for $n {
            #[inline]
            fn binop(left: &DMat<$n>, right: &$n) -> DMat<$n> {
                DMat {
                    nrows: left.nrows,
                    ncols: left.ncols,
                    mij:   left.mij.iter().map(|a| *a * *right).collect()
                }
            }
        }
    )
)

macro_rules! scalar_div_impl (
    ($n: ident) => (
        impl DMatDivRhs<$n, DMat<$n>> for $n {
            #[inline]
            fn binop(left: &DMat<$n>, right: &$n) -> DMat<$n> {
                DMat {
                    nrows: left.nrows,
                    ncols: left.ncols,
                    mij:   left.mij.iter().map(|a| *a / *right).collect()
                }
            }
        }
    )
)

macro_rules! scalar_add_impl (
    ($n: ident) => (
        impl DMatAddRhs<$n, DMat<$n>> for $n {
            #[inline]
            fn binop(left: &DMat<$n>, right: &$n) -> DMat<$n> {
                DMat {
                    nrows: left.nrows,
                    ncols: left.ncols,
                    mij:   left.mij.iter().map(|a| *a + *right).collect()
                }
            }
        }
    )
)

macro_rules! scalar_sub_impl (
    ($n: ident) => (
        impl DMatSubRhs<$n, DMat<$n>> for $n {
            #[inline]
            fn binop(left: &DMat<$n>, right: &$n) -> DMat<$n> {
                DMat {
                    nrows: left.nrows,
                    ncols: left.ncols,
                    mij:   left.mij.iter().map(|a| *a - *right).collect()
                }
            }
        }
    )
)

scalar_mul_impl!(f64)
scalar_mul_impl!(f32)
scalar_mul_impl!(u64)
scalar_mul_impl!(u32)
scalar_mul_impl!(u16)
scalar_mul_impl!(u8)
scalar_mul_impl!(i64)
scalar_mul_impl!(i32)
scalar_mul_impl!(i16)
scalar_mul_impl!(i8)
scalar_mul_impl!(uint)
scalar_mul_impl!(int)

scalar_div_impl!(f64)
scalar_div_impl!(f32)
scalar_div_impl!(u64)
scalar_div_impl!(u32)
scalar_div_impl!(u16)
scalar_div_impl!(u8)
scalar_div_impl!(i64)
scalar_div_impl!(i32)
scalar_div_impl!(i16)
scalar_div_impl!(i8)
scalar_div_impl!(uint)
scalar_div_impl!(int)

scalar_add_impl!(f64)
scalar_add_impl!(f32)
scalar_add_impl!(u64)
scalar_add_impl!(u32)
scalar_add_impl!(u16)
scalar_add_impl!(u8)
scalar_add_impl!(i64)
scalar_add_impl!(i32)
scalar_add_impl!(i16)
scalar_add_impl!(i8)
scalar_add_impl!(uint)
scalar_add_impl!(int)

scalar_sub_impl!(f64)
scalar_sub_impl!(f32)
scalar_sub_impl!(u64)
scalar_sub_impl!(u32)
scalar_sub_impl!(u16)
scalar_sub_impl!(u8)
scalar_sub_impl!(i64)
scalar_sub_impl!(i32)
scalar_sub_impl!(i16)
scalar_sub_impl!(i8)
scalar_sub_impl!(uint)
scalar_sub_impl!(int)
