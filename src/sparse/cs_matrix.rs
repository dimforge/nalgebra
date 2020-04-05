use num::Zero;
use simba::scalar::ClosedAdd;
use std::iter;
use std::marker::PhantomData;
use std::ops::Range;
use std::slice;

use crate::allocator::Allocator;
use crate::sparse::cs_utils;
use crate::{DefaultAllocator, Dim, Dynamic, Scalar, Vector, VectorN, U1};

pub struct ColumnEntries<'a, N> {
    curr: usize,
    i: &'a [usize],
    v: &'a [N],
}

impl<'a, N> ColumnEntries<'a, N> {
    #[inline]
    pub fn new(i: &'a [usize], v: &'a [N]) -> Self {
        assert_eq!(i.len(), v.len());
        Self { curr: 0, i, v }
    }
}

impl<'a, N: Clone> Iterator for ColumnEntries<'a, N> {
    type Item = (usize, N);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.i.len() {
            None
        } else {
            let res = Some(
                (unsafe { self.i.get_unchecked(self.curr).clone() }, unsafe {
                    self.v.get_unchecked(self.curr).clone()
                }),
            );
            self.curr += 1;
            res
        }
    }
}

// FIXME: this structure exists for now only because impl trait
// cannot be used for trait method return types.
/// Trait for iterable compressed-column matrix storage.
pub trait CsStorageIter<'a, N, R, C = U1> {
    /// Iterator through all the rows of a specific columns.
    ///
    /// The elements are given as a tuple (row_index, value).
    type ColumnEntries: Iterator<Item = (usize, N)>;
    /// Iterator through the row indices of a specific column.
    type ColumnRowIndices: Iterator<Item = usize>;

    /// Iterates through all the row indices of the j-th column.
    fn column_row_indices(&'a self, j: usize) -> Self::ColumnRowIndices;
    /// Iterates through all the entries of the j-th column.
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries;
}

/// Trait for mutably iterable compressed-column sparse matrix storage.
pub trait CsStorageIterMut<'a, N: 'a, R, C = U1> {
    /// Mutable iterator through all the values of the sparse matrix.
    type ValuesMut: Iterator<Item = &'a mut N>;
    /// Mutable iterator through all the rows of a specific columns.
    ///
    /// The elements are given as a tuple (row_index, value).
    type ColumnEntriesMut: Iterator<Item = (usize, &'a mut N)>;

    /// A mutable iterator through the values buffer of the sparse matrix.
    fn values_mut(&'a mut self) -> Self::ValuesMut;
    /// Iterates mutably through all the entries of the j-th column.
    fn column_entries_mut(&'a mut self, j: usize) -> Self::ColumnEntriesMut;
}

/// Trait for compressed column sparse matrix storage.
pub trait CsStorage<N, R, C = U1>: for<'a> CsStorageIter<'a, N, R, C> {
    /// The shape of the stored matrix.
    fn shape(&self) -> (R, C);
    /// Retrieve the i-th row index of the underlying row index buffer.
    ///
    /// No bound-checking is performed.
    unsafe fn row_index_unchecked(&self, i: usize) -> usize;
    /// The i-th value on the contiguous value buffer of this storage.
    ///
    /// No bound-checking is performed.
    unsafe fn get_value_unchecked(&self, i: usize) -> &N;
    /// The i-th value on the contiguous value buffer of this storage.
    fn get_value(&self, i: usize) -> &N;
    /// Retrieve the i-th row index of the underlying row index buffer.
    fn row_index(&self, i: usize) -> usize;
    /// The value indices for the `i`-th column.
    fn column_range(&self, i: usize) -> Range<usize>;
    /// The size of the value buffer (i.e. the entries known as possibly being non-zero).
    fn len(&self) -> usize;
}

/// Trait for compressed column sparse matrix mutable storage.
pub trait CsStorageMut<N, R, C = U1>:
    CsStorage<N, R, C> + for<'a> CsStorageIterMut<'a, N, R, C>
{
}

/// A storage of column-compressed sparse matrix based on a Vec.
#[derive(Clone, Debug, PartialEq)]
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
    /// The value buffer of this storage.
    pub fn values(&self) -> &[N] {
        &self.vals
    }

    /// The column shifts buffer.
    pub fn p(&self) -> &[usize] {
        self.p.as_slice()
    }

    /// The row index buffers.
    pub fn i(&self) -> &[usize] {
        &self.i
    }
}

impl<N: Scalar, R: Dim, C: Dim> CsVecStorage<N, R, C> where DefaultAllocator: Allocator<usize, C> {}

impl<'a, N: Scalar, R: Dim, C: Dim> CsStorageIter<'a, N, R, C> for CsVecStorage<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    type ColumnEntries = ColumnEntries<'a, N>;
    type ColumnRowIndices = iter::Cloned<slice::Iter<'a, usize>>;

    #[inline]
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries {
        let rng = self.column_range(j);
        ColumnEntries::new(&self.i[rng.clone()], &self.vals[rng])
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

impl<'a, N: Scalar, R: Dim, C: Dim> CsStorageIterMut<'a, N, R, C> for CsVecStorage<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    type ValuesMut = slice::IterMut<'a, N>;
    type ColumnEntriesMut = iter::Zip<iter::Cloned<slice::Iter<'a, usize>>, slice::IterMut<'a, N>>;

    #[inline]
    fn values_mut(&'a mut self) -> Self::ValuesMut {
        self.vals.iter_mut()
    }

    #[inline]
    fn column_entries_mut(&'a mut self, j: usize) -> Self::ColumnEntriesMut {
        let rng = self.column_range(j);
        self.i[rng.clone()]
            .iter()
            .cloned()
            .zip(self.vals[rng].iter_mut())
    }
}

impl<N: Scalar, R: Dim, C: Dim> CsStorageMut<N, R, C> for CsVecStorage<N, R, C> where
    DefaultAllocator: Allocator<usize, C>
{
}

/*
pub struct CsSliceStorage<'a, N: Scalar, R: Dim, C: DimAdd<U1>> {
    shape: (R, C),
    p: VectorSlice<usize, DimSum<C, U1>>,
    i: VectorSlice<usize, Dynamic>,
    vals: VectorSlice<N, Dynamic>,
}*/

/// A compressed sparse column matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CsMatrix<
    N: Scalar,
    R: Dim = Dynamic,
    C: Dim = Dynamic,
    S: CsStorage<N, R, C> = CsVecStorage<N, R, C>,
> {
    pub(crate) data: S,
    _phantoms: PhantomData<(N, R, C)>,
}

/// A column compressed sparse vector.
pub type CsVector<N, R = Dynamic, S = CsVecStorage<N, R, U1>> = CsMatrix<N, R, U1, S>;

impl<N: Scalar, R: Dim, C: Dim> CsMatrix<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    /// Creates a new compressed sparse column matrix with the specified dimension and
    /// `nvals` possible non-zero values.
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

    /*
    pub(crate) fn from_parts_generic(
        nrows: R,
        ncols: C,
        p: VectorN<usize, C>,
        i: Vec<usize>,
        vals: Vec<N>,
    ) -> Self
    where
        N: Zero + ClosedAdd,
        DefaultAllocator: Allocator<N, R>,
    {
        assert_eq!(ncols.value(), p.len(), "Invalid inptr size.");
        assert_eq!(i.len(), vals.len(), "Invalid value size.");

        // Check p.
        for ptr in &p {
            assert!(*ptr < i.len(), "Invalid inptr value.");
        }

        for ptr in p.as_slice().windows(2) {
            assert!(ptr[0] <= ptr[1], "Invalid inptr ordering.");
        }

        // Check i.
        for i in &i {
            assert!(*i < nrows.value(), "Invalid row ptr value.")
        }

        let mut res = CsMatrix {
            data: CsVecStorage {
                shape: (nrows, ncols),
                p,
                i,
                vals,
            },
            _phantoms: PhantomData,
        };

        // Sort and remove duplicates.
        res.sort();
        res.dedup();

        res
    }*/
}

/*
impl<N: Scalar + Zero + ClosedAdd> CsMatrix<N> {
    pub(crate) fn from_parts(
        nrows: usize,
        ncols: usize,
        p: Vec<usize>,
        i: Vec<usize>,
        vals: Vec<N>,
    ) -> Self
    {
        let nrows = Dynamic::new(nrows);
        let ncols = Dynamic::new(ncols);
        let p = DVector::from_data(VecStorage::new(ncols, U1, p));
        Self::from_parts_generic(nrows, ncols, p, i, vals)
    }
}
*/

impl<N: Scalar, R: Dim, C: Dim, S: CsStorage<N, R, C>> CsMatrix<N, R, C, S> {
    pub(crate) fn from_data(data: S) -> Self {
        CsMatrix {
            data,
            _phantoms: PhantomData,
        }
    }

    /// The size of the data buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// The number of rows of this matrix.
    pub fn nrows(&self) -> usize {
        self.data.shape().0.value()
    }

    /// The number of rows of this matrix.
    pub fn ncols(&self) -> usize {
        self.data.shape().1.value()
    }

    /// The shape of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    /// Whether this matrix is square or not.
    pub fn is_square(&self) -> bool {
        let (nrows, ncols) = self.data.shape();
        nrows.value() == ncols.value()
    }

    /// Should always return `true`.
    ///
    /// This method is generally used for debugging and should typically not be called in user code.
    /// This checks that the row inner indices of this matrix are sorted. It takes `O(n)` time,
    /// where n` is `self.len()`.
    /// All operations of CSC matrices on nalgebra assume, and will return, sorted indices.
    /// If at any time this `is_sorted` method returns `false`, then, something went wrong
    /// and an issue should be open on the nalgebra repository with details on how to reproduce
    /// this.
    pub fn is_sorted(&self) -> bool {
        for j in 0..self.ncols() {
            let mut curr = None;
            for idx in self.data.column_row_indices(j) {
                if let Some(curr) = curr {
                    if idx <= curr {
                        return false;
                    }
                }

                curr = Some(idx);
            }
        }

        true
    }

    /// Computes the transpose of this sparse matrix.
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

        let _ = cs_utils::cumsum(&mut workspace, &mut res.data.p);

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

impl<N: Scalar, R: Dim, C: Dim, S: CsStorageMut<N, R, C>> CsMatrix<N, R, C, S> {
    /// Iterator through all the mutable values of this sparse matrix.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut N> {
        self.data.values_mut()
    }
}

impl<N: Scalar, R: Dim, C: Dim> CsMatrix<N, R, C>
where
    DefaultAllocator: Allocator<usize, C>,
{
    pub(crate) fn sort(&mut self)
    where
        DefaultAllocator: Allocator<N, R>,
    {
        // Size = R
        let nrows = self.data.shape().0;
        let mut workspace = unsafe { VectorN::new_uninitialized_generic(nrows, U1) };
        self.sort_with_workspace(workspace.as_mut_slice());
    }

    pub(crate) fn sort_with_workspace(&mut self, workspace: &mut [N]) {
        assert!(
            workspace.len() >= self.nrows(),
            "Workspace must be able to hold at least self.nrows() elements."
        );

        for j in 0..self.ncols() {
            // Scatter the row in the workspace.
            for (irow, val) in self.data.column_entries(j) {
                workspace[irow] = val;
            }

            // Sort the index vector.
            let range = self.data.column_range(j);
            self.data.i[range.clone()].sort();

            // Permute the values too.
            for (i, irow) in range.clone().zip(self.data.i[range].iter().cloned()) {
                self.data.vals[i] = workspace[irow].inlined_clone();
            }
        }
    }

    // Remove dupliate entries on a sorted CsMatrix.
    pub(crate) fn dedup(&mut self)
    where
        N: Zero + ClosedAdd,
    {
        let mut curr_i = 0;

        for j in 0..self.ncols() {
            let range = self.data.column_range(j);
            self.data.p[j] = curr_i;

            if range.start != range.end {
                let mut value = N::zero();
                let mut irow = self.data.i[range.start];

                for idx in range {
                    let curr_irow = self.data.i[idx];

                    if curr_irow == irow {
                        value += self.data.vals[idx].inlined_clone();
                    } else {
                        self.data.i[curr_i] = irow;
                        self.data.vals[curr_i] = value;
                        value = self.data.vals[idx].inlined_clone();
                        irow = curr_irow;
                        curr_i += 1;
                    }
                }

                // Handle the last entry.
                self.data.i[curr_i] = irow;
                self.data.vals[curr_i] = value;
                curr_i += 1;
            }
        }

        self.data.i.truncate(curr_i);
        self.data.i.shrink_to_fit();
        self.data.vals.truncate(curr_i);
        self.data.vals.shrink_to_fit();
    }
}
