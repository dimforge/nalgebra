use num::Zero;
use simba::scalar::ClosedAddAssign;
use std::iter;
use std::marker::PhantomData;
use std::ops::Range;
use std::slice;

use crate::allocator::Allocator;
use crate::sparse::cs_utils;
use crate::{Const, DefaultAllocator, Dim, Dyn, Matrix, OVector, Scalar, U1, Vector};

pub struct ColumnEntries<'a, T> {
    curr: usize,
    i: &'a [usize],
    v: &'a [T],
}

impl<'a, T> ColumnEntries<'a, T> {
    #[inline]
    pub fn new(i: &'a [usize], v: &'a [T]) -> Self {
        assert_eq!(i.len(), v.len());
        Self { curr: 0, i, v }
    }
}

impl<T: Clone> Iterator for ColumnEntries<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.i.len() {
            None
        } else {
            let res = Some((unsafe { *self.i.get_unchecked(self.curr) }, unsafe {
                self.v.get_unchecked(self.curr).clone()
            }));
            self.curr += 1;
            res
        }
    }
}

// TODO: this structure exists for now only because impl trait
// cannot be used for trait method return types.
/// Trait for iterable compressed-column matrix storage.
pub trait CsStorageIter<'a, T, R, C = U1> {
    /// Iterator through all the rows of a specific columns.
    ///
    /// The elements are given as a tuple (`row_index`, value).
    type ColumnEntries: Iterator<Item = (usize, T)>;
    /// Iterator through the row indices of a specific column.
    type ColumnRowIndices: Iterator<Item = usize>;

    /// Iterates through all the row indices of the j-th column.
    fn column_row_indices(&'a self, j: usize) -> Self::ColumnRowIndices;
    /// Iterates through all the entries of the j-th column.
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries;
}

/// Trait for mutably iterable compressed-column sparse matrix storage.
pub trait CsStorageIterMut<'a, T: 'a, R, C = U1> {
    /// Mutable iterator through all the values of the sparse matrix.
    type ValuesMut: Iterator<Item = &'a mut T>;
    /// Mutable iterator through all the rows of a specific columns.
    ///
    /// The elements are given as a tuple (`row_index`, value).
    type ColumnEntriesMut: Iterator<Item = (usize, &'a mut T)>;

    /// A mutable iterator through the values buffer of the sparse matrix.
    fn values_mut(&'a mut self) -> Self::ValuesMut;
    /// Iterates mutably through all the entries of the j-th column.
    fn column_entries_mut(&'a mut self, j: usize) -> Self::ColumnEntriesMut;
}

/// Trait for compressed column sparse matrix storage.
pub trait CsStorage<T, R, C = U1>: for<'a> CsStorageIter<'a, T, R, C> {
    /// The shape of the stored matrix.
    fn shape(&self) -> (R, C);
    /// Retrieve the i-th row index of the underlying row index buffer.
    ///
    /// # Safety
    /// No bound-checking is performed.
    unsafe fn row_index_unchecked(&self, i: usize) -> usize;
    /// The i-th value on the contiguous value buffer of this storage.
    ///
    /// # Safety
    /// No bound-checking is performed.
    unsafe fn get_value_unchecked(&self, i: usize) -> &T;
    /// The i-th value on the contiguous value buffer of this storage.
    fn get_value(&self, i: usize) -> &T;
    /// Retrieve the i-th row index of the underlying row index buffer.
    fn row_index(&self, i: usize) -> usize;
    /// The value indices for the `i`-th column.
    fn column_range(&self, i: usize) -> Range<usize>;
    /// The size of the value buffer (i.e. the entries known as possibly being non-zero).
    fn len(&self) -> usize;
}

/// Trait for compressed column sparse matrix mutable storage.
pub trait CsStorageMut<T, R, C = U1>:
    CsStorage<T, R, C> + for<'a> CsStorageIterMut<'a, T, R, C>
{
}

/// A storage of column-compressed sparse matrix based on a Vec.
#[derive(Clone, Debug, PartialEq)]
pub struct CsVecStorage<T: Scalar, R: Dim, C: Dim>
where
    DefaultAllocator: Allocator<C>,
{
    pub(crate) shape: (R, C),
    pub(crate) p: OVector<usize, C>,
    pub(crate) i: Vec<usize>,
    pub(crate) vals: Vec<T>,
}

impl<T: Scalar, R: Dim, C: Dim> CsVecStorage<T, R, C>
where
    DefaultAllocator: Allocator<C>,
{
    /// Returns a reference to the value buffer containing all non-zero elements.
    ///
    /// In a Compressed Sparse Column (CSC) matrix, only non-zero values are stored.
    /// This method returns a slice containing all these stored values in column-major order.
    /// The values are organized such that for each column, its non-zero elements appear
    /// consecutively in this buffer.
    ///
    /// # Understanding Sparse Storage
    ///
    /// Unlike dense matrices that store every element (including zeros), sparse matrices
    /// only store non-zero values along with their positions. For a matrix with millions
    /// of elements but only thousands of non-zeros, this saves enormous amounts of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a 3x3 sparse matrix from triplet format (row, col, value)
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    ///     (2, 2, 3.0),
    /// ];
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Get the values buffer - contains only non-zero elements
    /// let values = m.data.values();
    /// assert_eq!(values.len(), 3); // Only 3 non-zero values stored
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsVecStorage::i`] - Returns the row indices corresponding to these values
    /// - [`CsVecStorage::p`] - Returns column pointer offsets to locate values by column
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.vals
    }

    /// Returns the column pointer buffer for the compressed storage format.
    ///
    /// In Compressed Sparse Column (CSC) format, the column pointer array indicates where
    /// each column's data begins in the values and row indices arrays. Specifically, for column `j`:
    /// - `p[j]` gives the starting index in the values/row indices arrays
    /// - `p[j+1]` gives the ending index (exclusive)
    /// - Thus, column `j` has `p[j+1] - p[j]` non-zero elements
    ///
    /// # Format Details
    ///
    /// - Length: `ncols` (number of columns)
    /// - `p[j]` ≤ `p[j+1]` for all valid `j` (monotonically increasing)
    /// - The values and row indices for column `j` are in ranges `[p[j]..p[j+1]]`
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a sparse matrix:
    /// // [1.0  0.0  2.0]
    /// // [0.0  3.0  0.0]
    /// // [0.0  0.0  4.0]
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 3.0),
    ///     (0, 2, 2.0),
    ///     (2, 2, 4.0),
    /// ];
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Column pointers tell us where each column's data starts
    /// let p = m.data.p();
    /// // Column 0: indices p[0]..p[1] (1 element)
    /// // Column 1: indices p[1]..p[2] (1 element)
    /// // Column 2: indices p[2]..p[3] (2 elements)
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsVecStorage::i`] - Returns the row indices for each stored value
    /// - [`CsVecStorage::values`] - Returns the actual non-zero values
    #[must_use]
    pub fn p(&self) -> &[usize] {
        self.p.as_slice()
    }

    /// Returns the row index buffer for the compressed storage format.
    ///
    /// This array stores the row indices of each non-zero element in the matrix.
    /// For each value in the values buffer, there's a corresponding entry in this
    /// array indicating which row that value belongs to.
    ///
    /// # Format Details
    ///
    /// - Length: Same as the values buffer (number of non-zero elements)
    /// - For column `j`, the row indices are in the range `[p[j]..p[j+1]]`
    /// - Within each column, row indices are sorted in ascending order
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a sparse matrix:
    /// // [1.0  0.0]
    /// // [2.0  0.0]
    /// // [0.0  3.0]
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 0, 2.0),
    ///     (2, 1, 3.0),
    /// ];
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 2, &triplets);
    ///
    /// // Row indices tell us which row each value belongs to
    /// let row_indices = m.data.i();
    /// let values = m.data.values();
    ///
    /// // The non-zero elements with their row indices:
    /// // values[0]=1.0 is in row row_indices[0]=0
    /// // values[1]=2.0 is in row row_indices[1]=1
    /// // values[2]=3.0 is in row row_indices[2]=2
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsVecStorage::p`] - Returns column pointers to segment this array by column
    /// - [`CsVecStorage::values`] - Returns the values corresponding to these row indices
    #[must_use]
    pub fn i(&self) -> &[usize] {
        &self.i
    }
}

impl<T: Scalar, R: Dim, C: Dim> CsVecStorage<T, R, C> where DefaultAllocator: Allocator<C> {}

impl<'a, T: Scalar, R: Dim, C: Dim> CsStorageIter<'a, T, R, C> for CsVecStorage<T, R, C>
where
    DefaultAllocator: Allocator<C>,
{
    type ColumnEntries = ColumnEntries<'a, T>;
    type ColumnRowIndices = iter::Cloned<slice::Iter<'a, usize>>;

    #[inline]
    fn column_entries(&'a self, j: usize) -> Self::ColumnEntries {
        let rng = self.column_range(j);
        ColumnEntries::new(&self.i[rng.clone()], &self.vals[rng])
    }

    #[inline]
    fn column_row_indices(&'a self, j: usize) -> Self::ColumnRowIndices {
        let rng = self.column_range(j);
        self.i[rng].iter().cloned()
    }
}

impl<T: Scalar, R: Dim, C: Dim> CsStorage<T, R, C> for CsVecStorage<T, R, C>
where
    DefaultAllocator: Allocator<C>,
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
        unsafe { *self.i.get_unchecked(i) }
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, i: usize) -> &T {
        unsafe { self.vals.get_unchecked(i) }
    }

    #[inline]
    fn get_value(&self, i: usize) -> &T {
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

impl<'a, T: Scalar, R: Dim, C: Dim> CsStorageIterMut<'a, T, R, C> for CsVecStorage<T, R, C>
where
    DefaultAllocator: Allocator<C>,
{
    type ValuesMut = slice::IterMut<'a, T>;
    type ColumnEntriesMut = iter::Zip<iter::Cloned<slice::Iter<'a, usize>>, slice::IterMut<'a, T>>;

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

impl<T: Scalar, R: Dim, C: Dim> CsStorageMut<T, R, C> for CsVecStorage<T, R, C> where
    DefaultAllocator: Allocator<C>
{
}

/*
pub struct CsSliceStorage<'a, T: Scalar, R: Dim, C: DimAdd<U1>> {
    shape: (R, C),
    p: VectorSlice<usize, DimSum<C, U1>>,
    i: VectorSlice<usize, Dyn>,
    vals: VectorSlice<T, Dyn>,
}*/

/// A compressed sparse column matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CsMatrix<
    T: Scalar,
    R: Dim = Dyn,
    C: Dim = Dyn,
    S: CsStorage<T, R, C> = CsVecStorage<T, R, C>,
> {
    pub(crate) data: S,
    _phantoms: PhantomData<(T, R, C)>,
}

/// A column compressed sparse vector.
pub type CsVector<T, R = Dyn, S = CsVecStorage<T, R, U1>> = CsMatrix<T, R, U1, S>;

impl<T: Scalar, R: Dim, C: Dim> CsMatrix<T, R, C>
where
    DefaultAllocator: Allocator<C>,
{
    /// Creates a new uninitialized compressed sparse column matrix with pre-allocated storage.
    ///
    /// This method allocates storage for a sparse matrix with the given dimensions and
    /// a specified capacity for non-zero values. The storage is uninitialized, meaning
    /// the values and row indices are not set to any particular values.
    ///
    /// # Parameters
    ///
    /// - `nrows`: Number of rows in the matrix
    /// - `ncols`: Number of columns in the matrix
    /// - `nvals`: Capacity for non-zero values (pre-allocates storage)
    ///
    /// # Understanding the Parameters
    ///
    /// The `nvals` parameter is crucial for performance. If you know approximately how many
    /// non-zero elements your matrix will contain, pre-allocating this space avoids
    /// expensive reallocations later. For example, a 1000×1000 tridiagonal matrix has
    /// approximately 3000 non-zero values.
    ///
    /// # Safety Note
    ///
    /// This method creates uninitialized storage. The matrix should be properly filled
    /// before use. This is typically used internally by other construction methods.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Dyn};
    ///
    /// // Create a sparse matrix structure for a 100x100 matrix
    /// // with space for approximately 500 non-zero elements
    /// let matrix = CsMatrix::<f64, Dyn, Dyn>::new_uninitialized_generic(
    ///     Dyn(100),
    ///     Dyn(100),
    ///     500
    /// );
    ///
    /// assert_eq!(matrix.nrows(), 100);
    /// assert_eq!(matrix.ncols(), 100);
    /// ```
    ///
    /// # Typical Use Cases
    ///
    /// - **Finite Element Assembly**: Pre-allocate for the expected number of entries
    ///   per element times the number of elements
    /// - **Graph Adjacency Matrices**: Allocate for the number of edges in the graph
    /// - **Sparse Matrix Construction**: Used internally by conversion methods
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::from_triplet`] - Higher-level method to create a sparse matrix from data
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
                p: OVector::zeros_generic(ncols, Const::<1>),
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
        p: OVector<usize, C>,
        i: Vec<usize>,
        vals: Vec<T>,
    ) -> Self
    where
        T: Zero + ClosedAddAssign,
        DefaultAllocator: Allocator<R>,
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
impl<T: Scalar + Zero + ClosedAddAssign> CsMatrix<T> {
    pub(crate) fn from_parts(
        nrows: usize,
        ncols: usize,
        p: Vec<usize>,
        i: Vec<usize>,
        vals: Vec<T>,
    ) -> Self
    {
        let nrows = Dyn(nrows);
        let ncols = Dyn(ncols);
        let p = DVector::from_data(VecStorage::new(ncols, U1, p));
        Self::from_parts_generic(nrows, ncols, p, i, vals)
    }
}
*/

impl<T: Scalar, R: Dim, C: Dim, S: CsStorage<T, R, C>> CsMatrix<T, R, C, S> {
    pub(crate) fn from_data(data: S) -> Self {
        CsMatrix {
            data,
            _phantoms: PhantomData,
        }
    }

    /// Returns the number of stored non-zero elements in this sparse matrix.
    ///
    /// This is the length of the internal storage buffers, representing how many
    /// non-zero values are currently stored. This is different from the total
    /// number of elements in the matrix (which would be `nrows() * ncols()`).
    ///
    /// # Sparse vs Dense Storage
    ///
    /// For a sparse matrix, `len()` typically returns a much smaller number than
    /// `nrows() * ncols()`. This is the key memory advantage of sparse storage:
    /// only non-zero elements are stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a 1000x1000 identity matrix (sparse)
    /// let triplets: Vec<_> = (0..1000).map(|i| (i, i, 1.0)).collect();
    /// let identity = CsMatrix::<f64, Dyn, Dyn>::from_triplet(1000, 1000, &triplets);
    ///
    /// // Only 1000 non-zero elements stored
    /// assert_eq!(identity.len(), 1000);
    ///
    /// // But the matrix represents 1,000,000 elements total
    /// assert_eq!(identity.nrows() * identity.ncols(), 1_000_000);
    /// ```
    ///
    /// # Applications
    ///
    /// - **Memory Estimation**: `len() * size_of::<T>()` gives approximate memory usage
    /// - **Sparsity Analysis**: Compare `len()` to `nrows() * ncols()` to measure sparsity
    /// - **Performance Hints**: Operations scale with `len()`, not matrix dimensions
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::nrows`] - Returns the number of rows
    /// - [`CsMatrix::ncols`] - Returns the number of columns
    /// - [`CsMatrix::shape`] - Returns both dimensions as a tuple
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of rows in this sparse matrix.
    ///
    /// This returns the logical number of rows, not the number of stored elements.
    /// A sparse matrix can have many rows even if most elements are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a sparse 100x50 matrix
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (99, 49, 2.0),
    /// ];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(100, 50, &triplets);
    ///
    /// assert_eq!(m.nrows(), 100);
    /// assert_eq!(m.ncols(), 50);
    /// assert_eq!(m.len(), 2); // Only 2 non-zero elements stored
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::ncols`] - Returns the number of columns
    /// - [`CsMatrix::shape`] - Returns both dimensions
    /// - [`CsMatrix::is_square`] - Checks if nrows equals ncols
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.data.shape().0.value()
    }

    /// Returns the number of columns in this sparse matrix.
    ///
    /// This returns the logical number of columns, not the number of stored elements.
    /// In CSC format, each column may contain zero or more non-zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a sparse tridiagonal matrix (useful for differential equations)
    /// let mut triplets = vec![];
    /// for i in 0..100 {
    ///     triplets.push((i, i, 2.0)); // diagonal
    ///     if i > 0 {
    ///         triplets.push((i, i-1, -1.0)); // lower diagonal
    ///     }
    ///     if i < 99 {
    ///         triplets.push((i, i+1, -1.0)); // upper diagonal
    ///     }
    /// }
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(100, 100, &triplets);
    ///
    /// assert_eq!(m.ncols(), 100);
    /// assert_eq!(m.nrows(), 100);
    /// assert_eq!(m.len(), 298); // 100 + 99 + 99 non-zeros
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::nrows`] - Returns the number of rows
    /// - [`CsMatrix::shape`] - Returns both dimensions
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.data.shape().1.value()
    }

    /// Returns the shape of this sparse matrix as a tuple `(nrows, ncols)`.
    ///
    /// This is a convenience method that returns both dimensions at once.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a rectangular sparse matrix
    /// let triplets = vec![(5, 10, 1.0), (8, 15, 2.0)];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(20, 30, &triplets);
    ///
    /// let (nrows, ncols) = m.shape();
    /// assert_eq!(nrows, 20);
    /// assert_eq!(ncols, 30);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::nrows`] - Returns just the number of rows
    /// - [`CsMatrix::ncols`] - Returns just the number of columns
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    /// Checks whether this matrix is square (has the same number of rows and columns).
    ///
    /// Many linear algebra operations, such as matrix decompositions and eigenvalue
    /// computations, require square matrices. This method provides a quick check.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Square matrix
    /// let square = CsMatrix::<f64, Dyn, Dyn>::from_triplet(
    ///     10, 10,
    ///     &[(0, 0, 1.0)]
    /// );
    /// assert!(square.is_square());
    ///
    /// // Rectangular matrix
    /// let rect = CsMatrix::<f64, Dyn, Dyn>::from_triplet(
    ///     10, 20,
    ///     &[(0, 0, 1.0)]
    /// );
    /// assert!(!rect.is_square());
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Precondition Checking**: Before calling operations that require square matrices
    /// - **Matrix Decompositions**: LU, Cholesky, and eigenvalue methods need square matrices
    /// - **Linear Systems**: Square matrices are needed for solving Ax = b
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::nrows`] - Returns the number of rows
    /// - [`CsMatrix::ncols`] - Returns the number of columns
    #[must_use]
    pub fn is_square(&self) -> bool {
        let (nrows, ncols) = self.data.shape();
        nrows.value() == ncols.value()
    }

    /// Verifies that row indices within each column are sorted in ascending order.
    ///
    /// This is a debugging method that checks the internal consistency of the sparse
    /// matrix structure. In the CSC (Compressed Sparse Column) format, row indices
    /// within each column must be sorted for correct operation of algorithms.
    ///
    /// # When to Use
    ///
    /// This method is primarily for debugging and testing. All standard operations
    /// in nalgebra maintain the sorted property automatically, so this should always
    /// return `true` in normal usage. If it returns `false`, there's likely a bug
    /// in the library or in custom code that modifies the matrix structure directly.
    ///
    /// # Performance
    ///
    /// This method takes O(n) time where n is `self.len()` (number of stored elements).
    /// It should not be called in performance-critical code.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a sparse matrix using standard methods
    /// let triplets = vec![
    ///     (5, 0, 1.0),
    ///     (1, 0, 2.0),
    ///     (3, 1, 3.0),
    /// ];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(10, 5, &triplets);
    ///
    /// // Row indices should always be sorted within columns
    /// assert!(m.is_sorted());
    /// ```
    ///
    /// # Implementation Note
    ///
    /// All nalgebra operations that construct or modify sparse matrices maintain
    /// the sorted invariant. If this method ever returns `false` for a matrix
    /// created using standard nalgebra operations, please report it as a bug.
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::len`] - Returns the number of stored elements
    #[must_use]
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

    /// Computes the transpose of this sparse matrix, swapping rows and columns.
    ///
    /// The transpose operation flips a matrix over its diagonal, converting rows to
    /// columns and columns to rows. For a sparse matrix, this also converts between
    /// CSC (Compressed Sparse Column) and CSR (Compressed Sparse Row) format.
    ///
    /// # Algorithm
    ///
    /// The implementation uses a counting-based algorithm that:
    /// 1. Counts non-zeros in each row to determine column pointers for the result
    /// 2. Redistributes values from column-major to row-major order
    /// 3. Maintains sorted row indices within each output column
    ///
    /// Time complexity: O(n + m) where n is the number of non-zeros and m is the number of rows.
    ///
    /// # Examples
    ///
    /// ## Basic Transpose
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a 2x3 sparse matrix:
    /// // [1.0  0.0  2.0]
    /// // [0.0  3.0  0.0]
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 3.0),
    ///     (0, 2, 2.0),
    /// ];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 3, &triplets);
    ///
    /// // Transpose to get 3x2 matrix:
    /// // [1.0  0.0]
    /// // [0.0  3.0]
    /// // [2.0  0.0]
    /// let mt = m.transpose();
    ///
    /// assert_eq!(mt.nrows(), 3);
    /// assert_eq!(mt.ncols(), 2);
    /// assert_eq!(mt.len(), 3);
    /// ```
    ///
    /// ## Symmetric Matrix Operations
    ///
    /// For symmetric matrices (common in finite element analysis), the transpose
    /// equals the original, but you still need to compute it to convert storage format:
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// // Create a symmetric graph adjacency matrix
    /// let triplets = vec![
    ///     (0, 1, 1.0), (1, 0, 1.0),  // edge 0-1
    ///     (1, 2, 1.0), (2, 1, 1.0),  // edge 1-2
    ///     (0, 2, 1.0), (2, 0, 1.0),  // edge 0-2
    /// ];
    /// let graph = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    /// let graph_t = graph.transpose();
    ///
    /// // Same logical matrix, but different storage format
    /// assert_eq!(graph.len(), graph_t.len());
    /// ```
    ///
    /// # Applications
    ///
    /// - **Linear Algebra**: Computing A^T A or A A^T for normal equations
    /// - **Graph Algorithms**: Converting from outgoing to incoming edge representation
    /// - **Optimization**: Some algorithms prefer CSR format, others CSC
    /// - **Numerical Methods**: Many algorithms need both A and A^T
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::shape`] - The transpose swaps the dimensions
    #[must_use = "This function does not mutate the matrix. Consider using the return value or removing the function call. There's also transpose_mut() for square matrices."]
    pub fn transpose(&self) -> CsMatrix<T, C, R>
    where
        DefaultAllocator: Allocator<R>,
    {
        let (nrows, ncols) = self.data.shape();

        let nvals = self.len();
        let mut res = CsMatrix::new_uninitialized_generic(ncols, nrows, nvals);
        let mut workspace = Vector::zeros_generic(nrows, Const::<1>);

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

impl<T: Scalar, R: Dim, C: Dim, S: CsStorageMut<T, R, C>> CsMatrix<T, R, C, S> {
    /// Returns a mutable iterator over all non-zero values in this sparse matrix.
    ///
    /// This allows you to modify the stored values in-place without changing the
    /// sparsity pattern (which elements are non-zero). The iteration order follows
    /// the internal storage order: column by column, with elements within each
    /// column ordered by increasing row index.
    ///
    /// # Use Cases
    ///
    /// - **Scaling**: Multiply all elements by a constant
    /// - **Thresholding**: Set small values to zero (without removing them from storage)
    /// - **Value Updates**: Modify matrix entries while preserving structure
    /// - **In-place Operations**: Apply functions to each element
    ///
    /// # Important Notes
    ///
    /// - This does NOT allow changing which elements are non-zero
    /// - Setting values to zero still keeps them in storage (they remain explicit zeros)
    /// - To actually remove zeros from storage, reconstruction is required
    /// - The sparsity pattern (positions of non-zeros) remains unchanged
    ///
    /// # Examples
    ///
    /// ## Scale All Values
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    ///     (2, 2, 3.0),
    /// ];
    /// let mut m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Scale all values by 2.0
    /// for value in m.values_mut() {
    ///     *value *= 2.0;
    /// }
    /// ```
    ///
    /// ## Apply Threshold
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// let triplets = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 0.001),
    ///     (2, 2, 5.0),
    /// ];
    /// let mut m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Set small values to zero (they remain in storage as explicit zeros)
    /// let threshold = 0.01;
    /// for value in m.values_mut() {
    ///     if value.abs() < threshold {
    ///         *value = 0.0;
    ///     }
    /// }
    /// ```
    ///
    /// ## Numerical Damping
    ///
    /// Common in iterative solvers - add damping to diagonal elements:
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    /// use nalgebra::Dyn;
    ///
    /// let triplets = vec![
    ///     (0, 0, 4.0),
    ///     (1, 1, 6.0),
    ///     (2, 2, 8.0),
    /// ];
    /// let mut m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Add regularization/damping (in practice you'd identify diagonal elements)
    /// for value in m.values_mut() {
    ///     *value += 0.1;
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsVecStorage::values`] - Immutable access to values
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.values_mut()
    }
}

impl<T: Scalar, R: Dim, C: Dim> CsMatrix<T, R, C>
where
    DefaultAllocator: Allocator<C>,
{
    pub(crate) fn sort(&mut self)
    where
        T: Zero,
        DefaultAllocator: Allocator<R>,
    {
        // Size = R
        let nrows = self.data.shape().0;
        let mut workspace = Matrix::zeros_generic(nrows, Const::<1>);
        self.sort_with_workspace(workspace.as_mut_slice());
    }

    pub(crate) fn sort_with_workspace(&mut self, workspace: &mut [T]) {
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
            self.data.i[range.clone()].sort_unstable();

            // Permute the values too.
            for (i, irow) in range.clone().zip(self.data.i[range].iter().cloned()) {
                self.data.vals[i] = workspace[irow].clone();
            }
        }
    }

    // Remove duplicate entries on a sorted CsMatrix.
    pub(crate) fn dedup(&mut self)
    where
        T: Zero + ClosedAddAssign,
    {
        let mut curr_i = 0;

        for j in 0..self.ncols() {
            let range = self.data.column_range(j);
            self.data.p[j] = curr_i;

            if range.start != range.end {
                let mut value = T::zero();
                let mut irow = self.data.i[range.start];

                for idx in range {
                    let curr_irow = self.data.i[idx];

                    if curr_irow == irow {
                        value += self.data.vals[idx].clone();
                    } else {
                        self.data.i[curr_i] = irow;
                        self.data.vals[curr_i] = value;
                        value = self.data.vals[idx].clone();
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
