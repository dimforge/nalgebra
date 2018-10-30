use alga::general::{ClosedAdd, ClosedMul};
use num::{One, Zero};
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Range};
use std::slice;

use allocator::Allocator;
use constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use storage::{Storage, StorageMut};
use {DefaultAllocator, Dim, Matrix, MatrixMN, Real, Scalar, Vector, VectorN, U1};

// FIXME: this structure exists for now only because impl trait
// cannot be used for trait method return types.
pub trait CsStorageIter<'a, N, R, C = U1> {
    type ColumnEntries: Iterator<Item = (usize, N)>;
    type ColumnRowIndices: Iterator<Item = usize>;

    fn column_row_indices(&'a self, j: usize) -> Self::ColumnRowIndices;
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries;
}

pub trait CsStorage<N, R, C = U1>: for<'a> CsStorageIter<'a, N, R, C> {
    fn shape(&self) -> (R, C);
    unsafe fn row_index_unchecked(&self, i: usize) -> usize;
    unsafe fn get_value_unchecked(&self, i: usize) -> &N;
    fn get_value(&self, i: usize) -> &N;
    fn row_index(&self, i: usize) -> usize;
    fn column_range(&self, i: usize) -> Range<usize>;
    fn len(&self) -> usize;
}

pub trait CsStorageMut<N, R, C = U1>: CsStorage<N, R, C> {
    /*
    /// Sets the length of this column without initializing its values and row indices.
    ///
    /// If the given length is larger than the current one, uninitialized entries are
    /// added at the end of the column `i`. This will effectively shift all the matrix entries
    /// of the columns at indices `j` with `j > i`.
    fn set_column_len(&mut self, i: usize, len: usize);
    */
}

#[derive(Clone, Debug)]
pub struct CsVecStorage<N: Scalar, R: Dim, C: Dim>
where
    DefaultAllocator: Allocator<usize, C>,
{
    pub(crate) shape: (R, C),
    pub(crate) p: VectorN<usize, C>,
    pub(crate) i: Vec<usize>,
    pub(crate) vals: Vec<N>,
}

impl<N: Scalar, R: Dim, C: Dim> CsVecStorage<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    pub fn values(&self) -> &[N] {
        &self.vals
    }
}

impl<N: Scalar, R: Dim, C: Dim> CsVecStorage<N, R, C> where DefaultAllocator: Allocator<usize, C> {}

impl<'a, N: Scalar, R: Dim, C: Dim> CsStorageIter<'a, N, R, C> for CsVecStorage<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    type ColumnEntries =
        iter::Zip<iter::Cloned<slice::Iter<'a, usize>>, iter::Cloned<slice::Iter<'a, N>>>;
    type ColumnRowIndices = iter::Cloned<slice::Iter<'a, usize>>;

    #[inline]
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries {
        let rng = self.column_range(j);
        self.i[rng.clone()]
            .iter()
            .cloned()
            .zip(self.vals[rng].iter().cloned())
    }

    #[inline]
    fn column_row_indices(&'a self, j: usize) -> Self::ColumnRowIndices {
        let rng = self.column_range(j);
        self.i[rng.clone()].iter().cloned()
    }
}

impl<N: Scalar, R: Dim, C: Dim> CsStorage<N, R, C> for CsVecStorage<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    #[inline]
    fn shape(&self) -> (R, C) {
        self.shape
    }

    #[inline]
    fn len(&self) -> usize {
        self.vals.len()
    }

    #[inline]
    fn row_index(&self, i: usize) -> usize {
        self.i[i]
    }

    #[inline]
    unsafe fn row_index_unchecked(&self, i: usize) -> usize {
        *self.i.get_unchecked(i)
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, i: usize) -> &N {
        self.vals.get_unchecked(i)
    }

    #[inline]
    fn get_value(&self, i: usize) -> &N {
        &self.vals[i]
    }

    #[inline]
    fn column_range(&self, j: usize) -> Range<usize> {
        let end = if j + 1 == self.p.len() {
            self.len()
        } else {
            self.p[j + 1]
        };

        self.p[j]..end
    }
}

/*
pub struct CsSliceStorage<'a, N: Scalar, R: Dim, C: DimAdd<U1>> {
    shape: (R, C),
    p: VectorSlice<usize, DimSum<C, U1>>,
    i: VectorSlice<usize, Dynamic>,
    vals: VectorSlice<N, Dynamic>,
}*/

/// A compressed sparse column matrix.
#[derive(Clone, Debug)]
pub struct CsMatrix<N: Scalar, R: Dim, C: Dim, S: CsStorage<N, R, C> = CsVecStorage<N, R, C>> {
    pub data: S,
    _phantoms: PhantomData<(N, R, C)>,
}

pub type CsVector<N, R, S = CsVecStorage<N, R, U1>> = CsMatrix<N, R, U1, S>;

impl<N: Scalar, R: Dim, C: Dim> CsMatrix<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    pub fn new_uninitialized_generic(nrows: R, ncols: C, nvals: usize) -> Self {
        let mut i = Vec::with_capacity(nvals);
        unsafe {
            i.set_len(nvals);
        }
        i.shrink_to_fit();

        let mut vals = Vec::with_capacity(nvals);
        unsafe {
            vals.set_len(nvals);
        }
        vals.shrink_to_fit();

        CsMatrix {
            data: CsVecStorage {
                shape: (nrows, ncols),
                p: VectorN::zeros_generic(ncols, U1),
                i,
                vals,
            },
            _phantoms: PhantomData,
        }
    }
}

fn cumsum<D: Dim>(a: &mut VectorN<usize, D>, b: &mut VectorN<usize, D>) -> usize
where
    DefaultAllocator: Allocator<usize, D>,
{
    assert!(a.len() == b.len());
    let mut sum = 0;

    for i in 0..a.len() {
        b[i] = sum;
        sum += a[i];
        a[i] = b[i];
    }

    sum
}

impl<N: Scalar, R: Dim, C: Dim, S: CsStorage<N, R, C>> CsMatrix<N, R, C, S> {
    pub fn from_data(data: S) -> Self {
        CsMatrix {
            data,
            _phantoms: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn nrows(&self) -> usize {
        self.data.shape().0.value()
    }

    pub fn ncols(&self) -> usize {
        self.data.shape().1.value()
    }

    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    pub fn is_square(&self) -> bool {
        let (nrows, ncols) = self.data.shape();
        nrows.value() == ncols.value()
    }

    pub fn transpose(&self) -> CsMatrix<N, C, R>
    where
        DefaultAllocator: Allocator<usize, R>,
    {
        let (nrows, ncols) = self.data.shape();

        let nvals = self.len();
        let mut res = CsMatrix::new_uninitialized_generic(ncols, nrows, nvals);
        let mut workspace = Vector::zeros_generic(nrows, U1);

        // Compute p.
        for i in 0..nvals {
            let row_id = self.data.row_index(i);
            workspace[row_id] += 1;
        }

        let _ = cumsum(&mut workspace, &mut res.data.p);

        // Fill the result.
        for j in 0..ncols.value() {
            for (row_id, value) in self.data.column_entries(j) {
                let shift = workspace[row_id];

                res.data.vals[shift] = value;
                res.data.i[shift] = j;
                workspace[row_id] += 1;
            }
        }

        res
    }
}
