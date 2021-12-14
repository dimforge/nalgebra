//! A type for representing compressed sparse (row-major / column-major) matrices.

use super::{
    error::{SparseFormatError, SparsityPatternFormatError},
    factorization::CsCholesky,
    SparseEntry,
};
use nalgebra::{RealField, Scalar};
use num_traits::One;
use std::{borrow::Borrow, cmp::Ord, cmp::Ordering, marker::PhantomData};

/// An empty type to represent CSC-like storage convention.
#[derive(Debug, Clone, Copy)]
pub struct CompressedColumnStorage {}

/// An empty type to represent CSR-like storage convention.
#[derive(Debug, Clone, Copy)]
pub struct CompressedRowStorage {}

/// A trait denoting a type that represents a compression format for `CsMatrix` types.
///
/// The trait is sealed, as there are only two types (at present) to denote CSC and CSR style
/// compression.
pub trait Compression: private::Sealed {
    /// The type of compression of the transpose of a `CsMatrix` with the given `Compression`.
    type Transpose: Compression;

    /// The number of rows or columns along the major dimension for this compressed type, given a
    /// shape.
    fn nmajor(rows: usize, cols: usize) -> usize;

    /// The number of rows or columns along the minor dimension for this compressed type, given a
    /// shape.
    fn nminor(rows: usize, cols: usize) -> usize;
}

impl Compression for CompressedColumnStorage {
    type Transpose = CompressedRowStorage;

    fn nmajor(_rows: usize, cols: usize) -> usize {
        cols
    }

    fn nminor(rows: usize, _cols: usize) -> usize {
        rows
    }
}

impl Compression for CompressedRowStorage {
    type Transpose = CompressedColumnStorage;

    fn nmajor(rows: usize, _cols: usize) -> usize {
        rows
    }

    fn nminor(_rows: usize, cols: usize) -> usize {
        cols
    }
}

/// Module for adding a trait to "seal" the implementation of `Compression`.
mod private {
    /// Private trait to seal impl of `Compression`.
    pub trait Sealed {}

    impl Sealed for super::CompressedColumnStorage {}
    impl Sealed for super::CompressedRowStorage {}
}

/// An abstract compressed matrix for representing sparse matrix data.
///
/// This type is the preferred format for performing sparse-matrix operations due to its
/// compact representation. Constructing this type directly requires a deep knowledge of the
/// internals of the compressed matrix format, so it is typically advised that one constructs their
/// matrix using the `CooMatrix` type, and then converting to a compressed representation for
/// linear algebra operations after the fact.
///
/// This type is parameterized by the following types:
///
/// - `T`: The data type of each element
/// - `MajorOffsets`: A type for holding the offsets for each major axis lane.
/// - `MinorIndices`: A type for holding the indices for each minor axis entry.
/// - `Data`: A type for holding the explicit non-zero element values of the sparse matrix.
/// - `CompressionKind`: A type describing the compression (CSR or CSC).
///
/// ## Storage Format
///
/// In a compressed sparse matrix, values of a sparse matrix are stored as the row and column
/// locations of what are referred to as explicit values, or explicit non-zero values.
///
/// **Caveat**: while we may call them 'non-zero' values, nothing stops one from adding a `0.0` or
/// some such to the matrix. In such a case, this would be called an explicit zero, rather than an
/// implicit zero, which isn't directly stored (i.e. implicitly there). Nonetheless, when referring
/// to "explicit non-zero" values, we are also including explicit zeros into that grouping.
///
/// The difference between the CSC and CSR formats is fundamentally about which axis is the major
/// and minor axis. In a column format (CSC), the data is stored in column-major form, which means
/// the `MajorOffsets` correspond to columns, and `MinorIndices` directly to row indices of
/// explicit values in the matrix. CSR is similar, except for rows.
///
/// This API denotes the major storage as "lanes." So if your compression kind is CSC, then your
/// storage layout means that 1 lane is 1 column. Conversely, if your compression kind is CSR, then
/// your storage layout means that 1 lane is 1 row.
///
/// ## Construction & Use
///
/// The `CsMatrix` type limits the ways in which it can be constructed for the most part, because
/// it relies on some invariants. The main things that need to be maintained for this type are:
///
/// 1. **Sorting**. The offsets, indices, and data need to be sorted such that the data is in major
///    -> minor ordering. Specifically, all indices in `MinorIndices` should be monotonically
///    increasing for a given lane. Obviously, lanes should be ordered according to the matrix
///    layout.
/// 2. **Immutability**. One may notice that there are no associated `impl` functions that take the
///    type as `&mut self`. The type therefore enforces that the owned data is immutable. That is
///    not to say that no mutability is allowed, but in almost every case borrowed mutability does
///    not make a lot of sense. e.g. you cannot add a new non-zero element to the matrix without
///    fully re-computing the offsets and indices, so you are better off owning the type as `self`
///    and constructing a new `CsMatrix` rather than borrowing as `&mut self`.
/// 3. **Shape** and **Sizes**. Matrix shape and offsets / indices / data lengths are validated
///    upon construction. Thanks to the previous point (immutability), it is possible to guarantee
///    correctness of the data layout for the lifetime of the object, until it is consumed.
///
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressionKind>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
    CompressionKind: Compression,
{
    /// The shape of the matrix `(nrows, ncols)`
    ///
    /// The shape is always rows then columns, regardless of whether the `CompressionKind` is CSR
    /// or CSC.
    shape: (usize, usize),

    /// A collection of the offsets for each row of the matrix.
    ///
    /// `MajorOffsets` is a type that can be borrowed as `&[usize]`, which should represent an
    /// slice of length `nmajor`, where `nmajor` is the number of rows in e.g. a CSR matrix, or the
    /// number of columns in e.g. a CSC matrix.
    ///
    /// Each element of the slice contains an offset, which is a number denoting the number of
    /// explicit non-zero elements in `Data` to skip in order to get to that major axis lane. So
    /// for example, if we had the following matrix in CSC (column-major) format:
    ///
    /// ```text
    /// | 2 | 0 | 0 |
    /// |---|---|---|
    /// | 0 | 1 | 0 |
    /// |---|---|---|
    /// | 0 | 3 | 0 |
    /// ```
    ///
    /// There is one non-zero element in the first column, two in the second, and zero in the
    /// third. So our offsets would:
    ///
    /// ```rust
    /// // One non-zero element in lane with index 0
    /// // Two non-zero elements in lane with index 1
    /// // Zero non-zero elements in lane with index 2
    /// let counts = vec![1, 2, 0];
    ///
    /// // Zero non-zero elements before lane with index 0
    /// // One non-zero element before lane with index 1
    /// // Three non-zero elements before lane with index 2
    /// let offsets = vec![0, 1, 3];
    /// ```
    ///
    /// One can view the offsets as an accumulated sum of all the previous counts.
    ///
    /// ```rust
    /// // Pretend you have a matrix where these offsets are valid
    /// let offsets = vec![0, 2, 7, 11, 11, 13, 13, 17, 17, 17, 23];
    /// // Assume we want to get data at column 4
    /// let i = 4;
    ///
    /// let offset_column_i = offsets[i];
    /// ```
    offsets: MajorOffsets,

    /// A collection of the minor axis indices directly corresponding to the values in `data`.
    ///
    /// Unlike `offsets`, this holds a collection of index positions in the matrix for each value
    /// in `data`. `MinorIndices` is a type that can be borrowed as `&[usize]`, which should
    /// represent the minor-axis indices of each explicit non-zero element in `data`.
    ///
    /// If you have the following matrix in CSC (column-major) format:
    /// ```text
    /// | 7 | 0 | 0 |
    /// |---|---|---|
    /// | 0 | 4 | 0 |
    /// |---|---|---|
    /// | 0 | 9 | 0 |
    /// |---|---|---|
    /// | 7 | 0 | 6 |
    /// ```
    ///
    /// Then your offsets and indices would look like:
    ///
    /// ```rust
    /// let offsets = vec![0, 2, 4];
    /// let indices = vec![0, 3, 1, 2, 3];
    /// let data = vec![7, 7, 4, 9, 6];
    /// ```
    ///
    /// The length of indices directly corresponds to the number of explicit non-zero elements in
    /// the matrix.
    indices: MinorIndices,

    /// The explicit non-zero values contained within the sparse matrix, sorted in major -> minor
    /// order.
    ///
    /// `Data` is a type that can be borrowed as `&[T]`. The length of that slice should be equal
    /// to the number of explicit non-zero elements in the matrix.
    data: Data,

    /// Phantom type for the extra generic parameters that allow us to specify the `CsMatrix` type,
    /// but otherwise are not directly held by the struct itself.
    _phantom: PhantomData<(T, CompressionKind)>,
}

/// An alias for producing an owned, row-major compressed sparse matrix.
pub type CsrMatrix<T> = CsMatrix<T, Vec<usize>, Vec<usize>, Vec<T>, CompressedRowStorage>;

/// An alias for producing an owned, column-major compressed sparse matrix.
pub type CscMatrix<T> = CsMatrix<T, Vec<usize>, Vec<usize>, Vec<T>, CompressedColumnStorage>;

impl<T, MajorOffsets, MinorIndices, Data, CompressionKind>
    CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressionKind>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
    CompressionKind: Compression,
{
    /// The shape of the matrix, as (nrows, ncols).
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The number of rows in this matrix.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        let (rows, _) = self.shape;
        rows
    }

    /// The number of columns in this matrix.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        let (_, cols) = self.shape;
        cols
    }

    /// The number of lanes along the major dimension of this matrix.
    #[inline]
    pub fn nmajor(&self) -> usize {
        let (rows, cols) = self.shape;
        CompressionKind::nmajor(rows, cols)
    }

    /// The number of lanes along the minor dimension of this matrix.
    #[inline]
    pub fn nminor(&self) -> usize {
        let (rows, cols) = self.shape;
        CompressionKind::nminor(rows, cols)
    }

    /// Returns the number of non-zero entries in the sparse matrix.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.borrow().len()
    }

    pub(crate) unsafe fn from_parts_unchecked(
        nrows: usize,
        ncols: usize,
        offsets: MajorOffsets,
        indices: MinorIndices,
        data: Data,
    ) -> Self {
        Self {
            shape: (nrows, ncols),
            offsets,
            indices,
            data,
            _phantom: PhantomData,
        }
    }

    /// Constructor for the `CsMatrix` type that checks for shape / size / compression consistency.
    ///
    /// # Errors
    ///
    /// This function will error out in any of the following (non-exhaustive) scenarios:
    ///
    /// - `offsets` does not have a length equal to the major dimension length of the matrix plus
    /// one.
    /// - The first entry in `offsets` is not zero.
    /// - The elements of `offsets` are not monotonically increasing.
    /// - The elements of `indices` are not monotonically increasing per lane.
    /// - `indices` has an element that is greater than the minor dimension length of the matrix.
    /// - Duplicate entries were detected (i.e. two entries have the same major and minor index).
    pub fn try_from_parts(
        nrows: usize,
        ncols: usize,
        offsets: MajorOffsets,
        indices: MinorIndices,
        data: Data,
    ) -> Result<Self, SparseFormatError> {
        let nmajor = CompressionKind::nmajor(nrows, ncols);
        let nminor = CompressionKind::nminor(nrows, ncols);
        let borrowed_offsets = offsets.borrow();
        let borrowed_indices = indices.borrow();

        if borrowed_offsets.len() != nmajor {
            // size mismatch
            return Err(SparsityPatternFormatError::InvalidOffsetArrayLength.into());
        }

        if let Some(first) = borrowed_offsets.first() {
            if *first != 0 {
                // First entry exists and is not zero
                return Err(SparsityPatternFormatError::InvalidFirstOffset.into());
            }
        }

        if borrowed_indices.len() != data.borrow().len() {
            // size mismatch
            return Err(SparsityPatternFormatError::DataAndIndicesSizeMismatch.into());
        }

        if borrowed_indices.iter().any(|&index| index >= nminor) {
            // Index out-of-bounds
            return Err(SparsityPatternFormatError::MinorIndexOutOfBounds.into());
        }

        for major_index in 0..nmajor {
            let lower = borrowed_offsets[major_index];

            let lane_indices = if major_index + 1 < nmajor {
                let upper = borrowed_offsets[major_index + 1];

                if lower > upper {
                    // Offsets do not monotonically increase
                    return Err(SparsityPatternFormatError::NonmonotonicOffsets.into());
                }

                &borrowed_indices[lower..upper]
            } else {
                &borrowed_indices[lower..]
            };

            if !lane_indices.is_empty() {
                if let Some(err) = lane_indices
                    .iter()
                    .zip(&lane_indices[1..])
                    .filter_map(|(lower_index, upper_index)| {
                        match lower_index.cmp(upper_index) {
                            Ordering::Less => None,
                            Ordering::Equal => {
                                // Duplicates detected
                                Some(Err(SparsityPatternFormatError::DuplicateEntry.into()))
                            }
                            Ordering::Greater => {
                                // Indices in lane do not monotonically increase
                                Some(Err(
                                    SparsityPatternFormatError::NonmonotonicMinorIndices.into()
                                ))
                            }
                        }
                    })
                    .next()
                {
                    return err;
                }
            }
        }

        Ok(unsafe { Self::from_parts_unchecked(nrows, ncols, offsets, indices, data) })
    }

    /// Consumes self and returns the underlying major offsets, minor indices, and data contained
    /// within the sparse matrix format.
    pub fn disassemble(self) -> (MajorOffsets, MinorIndices, Data) {
        (self.offsets, self.indices, self.data)
    }

    /// Borrows self and returns three slices to the major offsets, minor indices, and data
    /// contained within the sparse matrix format.
    pub fn cs_data(&self) -> (&[usize], &[usize], &[T]) {
        (
            self.offsets.borrow(),
            self.indices.borrow(),
            self.data.borrow(),
        )
    }

    /// Produces an immutable view of the data by borrowing the underlying lanes and sparsity
    /// pattern data.
    ///
    /// This function can be useful when needing to pass an "owned" copy of the data around, but
    /// not wanting to make a true copy of the data. For example, standard operations like `Add`,
    /// `Mul`, `Sub`, etc. require that the types passed in are owned.
    ///
    /// This can be used in most places where `.clone()` is used for the same semantic effect while
    /// reducing the overall memory footprint of your program.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use nalgebra_sparse::cs::CsrMatrix;
    /// #
    /// # fn add_mat(rows: usize, cols: usize, offsets: Vec<usize>, indices: Vec<usize>, data:
    /// # Vec<f64>)
    /// # {
    /// let A = CsrMatrix::<f64>::try_from_parts(rows, cols, offsets, indices, data).unwrap();
    ///
    /// // In order to do this, we need to perform at least one full copy of A.
    /// let C1 = A.clone() + A.clone();
    ///
    /// // Instead we use a view.
    /// //
    /// // Here, rather than do two full copies + allocs, we instead take views of the matrix,
    /// // which won't allocate extra memory for the operation.
    /// let C2 = A.to_view() + A.to_view();
    /// # }
    /// ```
    pub fn to_view(&self) -> CsMatrix<T, &[usize], &[usize], &[T], CompressionKind> {
        let shape = self.shape();
        let (offsets, indices, data) = self.cs_data();

        CsMatrix {
            shape,
            offsets,
            indices,
            data,
            _phantom: PhantomData,
        }
    }

    /// Produces an immutable view of the transpose of the data by borrowing the underlying lanes
    /// and sparsity pattern data.
    pub fn transpose(&self) -> CsMatrix<T, &[usize], &[usize], &[T], CompressionKind::Transpose> {
        let (nrows, ncols) = self.shape();
        let shape = (ncols, nrows);

        CsMatrix {
            shape,
            offsets: self.offsets.borrow(),
            indices: self.indices.borrow(),
            data: self.data.borrow(),
            _phantom: PhantomData,
        }
    }

    /// Gets `Some` immutable reference to the sparse entry at `(major_index, minor_index)` if it is
    /// explicitly non-zero, otherwise returns `Some(SparseEntry::Zero)`.
    ///
    /// This function will return `None` if and only if the requested entry is out-of-bounds of the
    /// underlying matrix.
    fn get_entry_major_minor(
        &self,
        major_index: usize,
        minor_index: usize,
    ) -> Option<SparseEntry<'_, T>> {
        let nmajor = self.nmajor();
        let nminor = self.nminor();

        if major_index >= nmajor || minor_index >= nminor {
            return None;
        }

        let offset = self.offsets.borrow()[major_index];

        let (indices, data) = if major_index + 1 < self.nmajor() {
            let offset_upper = self.offsets.borrow()[major_index + 1];

            let indices = &self.indices.borrow()[offset..offset_upper];
            let data = &self.data.borrow()[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices.borrow()[offset..];
            let data = &self.data.borrow()[offset..];

            (indices, data)
        };

        let entry = if let Ok(local_index) = indices.binary_search_by(|&x| x.cmp(&minor_index)) {
            SparseEntry::NonZero(&data[local_index])
        } else {
            SparseEntry::Zero
        };

        Some(entry)
    }

    /// An iterator that iterates through every implicit and explicit entry in the matrix.
    pub fn all_entries(&self) -> AllElementsIter<'_, T> {
        let minor_length = self.nminor();
        let (offsets, indices, data) = self.cs_data();

        AllElementsIter {
            current_major_index: 0,
            current_minor_index: 0,
            minor_length,
            offsets,
            indices,
            data,
        }
    }

    /// An iterator that iterates through every explicit non-zero triplet `(major_index,
    /// minor_index, value)` in the Matrix, in major -> minor (i.e. sorted) order.
    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.iter().enumerate().flat_map(|(major_index, lane)| {
            lane.map(move |(minor_index, value)| (major_index, minor_index, value))
        })
    }

    /// Gets a major-axis lane of the data given a major index.
    ///
    /// Returns `None` iff the major index does not correspond to a lane in the `CsMatrix` (e.g. if
    /// you ask for lane 100 on a 3x3 sparse matrix).
    pub fn get_lane(&self, major_index: usize) -> Option<CsLaneIter<'_, T>> {
        if major_index >= self.nmajor() {
            return None;
        }

        let offset = self.offsets.borrow()[major_index];

        let (indices, data) = if major_index + 1 < self.nmajor() {
            let offset_upper = self.offsets.borrow()[major_index + 1];

            let indices = &self.indices.borrow()[offset..offset_upper];
            let data = &self.data.borrow()[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices.borrow()[offset..];
            let data = &self.data.borrow()[offset..];

            (indices, data)
        };

        Some(CsLaneIter {
            current_local_index: 0,
            indices,
            data,
        })
    }

    /// An iterator that iterates across every major lane of the `CsMatrix`, in order.
    pub fn iter(&self) -> CsMatrixIter<'_, T> {
        let (offsets, indices, data) = self.cs_data();

        CsMatrixIter {
            current_major_index: 0,
            number_of_lanes: offsets.len(),
            offsets,
            indices,
            data,
        }
    }

    /// An iterator that iterates across every minor lane of the `CsMatrix`, in order.
    ///
    /// NOTE: From a performance perspective, this iterator and [`CsMinorLaneIter`] is not necessarily
    /// ideal. Compressed-Sparse formats are most effective in algorithms where the major ordering is
    /// exploited. Because the matrix is compressed along the opposite dimension (the major dimension),
    /// it is necessary to search backwards through the compression in order to produce the right major
    /// indices for the minor lane.
    pub fn minor_lane_iter(&self) -> CsMatrixMinorLaneIter<'_, T> {
        let (offsets, indices, data) = self.cs_data();

        CsMatrixMinorLaneIter {
            current_minor_index: 0,
            minor_dim: self.nminor(),
            offsets,
            indices,
            data,
        }
    }
}

impl<T, MajorOffsets, MinorIndices, Data, CompressionKind>
    CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressionKind>
where
    T: Scalar + RealField,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
    CompressionKind: Compression,
{
    /// Gets the Cholesky factorization of the matrix, if it exists.
    ///
    /// The factorization will exist only if this matrix is symmetric and positive definite.
    pub fn cholesky(&self) -> Option<CsCholesky<T>> {
        CsCholesky::factor(&self).ok()
    }
}

impl<T, MajorOffsets, MinorIndices, Data>
    CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedRowStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    /// Gets a value in the sparse matrix from a `(row, column)` index pair.
    ///
    /// This function will return `None` if and only if the requested entry is out-of-bounds of the
    /// underlying matrix.
    #[inline]
    pub fn get_entry(&self, row: usize, column: usize) -> Option<SparseEntry<'_, T>> {
        self.get_entry_major_minor(row, column)
    }
}

impl<T, MajorOffsets, MinorIndices, Data>
    CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedColumnStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    /// Gets a value in the sparse matrix from a `(row, column)` index pair.
    ///
    /// This function will return `None` if and only if the requested entry is out-of-bounds of the
    /// underlying matrix.
    #[inline]
    pub fn get_entry(&self, row: usize, column: usize) -> Option<SparseEntry<'_, T>> {
        self.get_entry_major_minor(column, row)
    }
}

impl<T, C> CsMatrix<T, Vec<usize>, Vec<usize>, Vec<T>, C>
where
    T: Scalar,
    C: Compression,
{
    /// Returns an owned `CsMatrix` of shape `(nrows, ncols)` entirely comprised of implicit zeros,
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let nmajor = C::nmajor(nrows, ncols);

        Self {
            shape: (nrows, ncols),
            offsets: vec![0; nmajor],
            indices: Vec::new(),
            data: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Takes the transpose of the current matrix by taking ownership of the underlying data.
    ///
    /// Behaves like [`CsMatrix::transpose`], but takes `self` instead of `&self`.
    pub fn transpose_owned(self) -> CsMatrix<T, Vec<usize>, Vec<usize>, Vec<T>, C::Transpose> {
        let (nrows, ncols) = self.shape;

        CsMatrix {
            shape: (ncols, nrows),
            offsets: self.offsets,
            indices: self.indices,
            data: self.data,
            _phantom: PhantomData,
        }
    }
}

impl<T, C> CsMatrix<T, Vec<usize>, Vec<usize>, Vec<T>, C>
where
    T: Scalar + One,
    C: Compression,
{
    /// Produces an owned identity matrix of shape `(n, n)` in CSC format.
    #[inline]
    pub fn identity(n: usize) -> Self {
        let offsets = (0..n).collect();
        let indices = (0..n).collect();
        let data = vec![T::one(); n];

        Self {
            shape: (n, n),
            offsets,
            indices,
            data,
            _phantom: PhantomData,
        }
    }
}

/// A type to represent iteration through all the elements (zeros and explicit non-zeros) of a
/// `CsMatrix`.
///
/// As an iterator yields `(major_index, minor_index, value)` as `(usize, usize, SparseEntry<'_,
/// T>)`, for every index. Note that for most matrices you probably don't want this, since this
/// will include implicit zeros as well (returned as `SparseEntry::Zero`). However, this can be
/// useful if one is trying to pretty-print a matrix to the screen, or checking the explicit
/// structure of the matrix in a test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllElementsIter<'a, T> {
    current_major_index: usize,
    current_minor_index: usize,
    minor_length: usize,
    offsets: &'a [usize],
    indices: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Iterator for AllElementsIter<'a, T> {
    type Item = (usize, usize, SparseEntry<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_major_index >= self.offsets.len()
            || self.current_minor_index >= self.minor_length
        {
            return None;
        }

        let major_index = self.current_major_index;
        let minor_index = self.current_minor_index;

        let offset = self.offsets[major_index];

        let (indices, data) = if major_index + 1 < self.offsets.len() {
            let offset_upper = self.offsets[major_index + 1];

            let indices = &self.indices[offset..offset_upper];
            let data = &self.data[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices[offset..];
            let data = &self.data[offset..];

            (indices, data)
        };

        let entry = if let Ok(local_index) = indices.binary_search_by(|&x| x.cmp(&minor_index)) {
            SparseEntry::NonZero(&data[local_index])
        } else {
            SparseEntry::Zero
        };

        self.current_minor_index += 1;

        if self.current_minor_index >= self.minor_length {
            self.current_minor_index = 0;
            self.current_major_index += 1;
        }

        // We don't use `self.current_XXXXX_index` here because those were just modified.
        Some((major_index, minor_index, entry))
    }
}

impl<'a, T> ExactSizeIterator for AllElementsIter<'a, T> {
    fn len(&self) -> usize {
        let nelems = self.minor_length * self.offsets.len();
        let ntraversed = self.current_major_index * self.minor_length + self.current_minor_index;

        if nelems > ntraversed {
            nelems - ntraversed
        } else {
            0
        }
    }
}

/// An iterator through each of the major lanes of a `CsMatrix`.
///
/// This yields `CsLaneIter<'_, T>` for every lane. If you want the major index of each lane
/// alongside it, we suggest that users use `.enumerate()` on the resulting iterator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrixIter<'a, T> {
    current_major_index: usize,
    number_of_lanes: usize,
    offsets: &'a [usize],
    indices: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Iterator for CsMatrixIter<'a, T> {
    type Item = CsLaneIter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_major_index >= self.number_of_lanes {
            return None;
        }

        let offset = self.offsets[self.current_major_index];

        let (indices, data) = if self.current_major_index + 1 < self.offsets.len() {
            let offset_upper = self.offsets[self.current_major_index + 1];

            let indices = &self.indices[offset..offset_upper];
            let data = &self.data[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices[offset..];
            let data = &self.data[offset..];

            (indices, data)
        };

        self.current_major_index += 1;

        Some(CsLaneIter {
            current_local_index: 0,
            indices,
            data,
        })
    }
}

impl<'a, T> ExactSizeIterator for CsMatrixIter<'a, T> {
    fn len(&self) -> usize {
        if self.number_of_lanes > self.current_major_index {
            self.number_of_lanes - self.current_major_index
        } else {
            0
        }
    }
}

impl<'a, T> DoubleEndedIterator for CsMatrixIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_major_index >= self.number_of_lanes {
            return None;
        }

        self.number_of_lanes -= 1;

        let offset = self.offsets[self.number_of_lanes];

        let (indices, data) = if self.number_of_lanes + 1 < self.offsets.len() {
            let offset_upper = self.offsets[self.number_of_lanes + 1];

            let indices = &self.indices[offset..offset_upper];
            let data = &self.data[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices[offset..];
            let data = &self.data[offset..];

            (indices, data)
        };

        Some(CsLaneIter {
            current_local_index: 0,
            indices,
            data,
        })
    }
}

/// An iterator through each of the minor lanes of a `CsMatrix`.
///
/// "Minor lane" here refers to a lane along the minor axis, i.e. if you have a CSC matrix, you get
/// lanes over rows; conversely, if you have a CSR matrix you get lanes along columns. This is the
/// opposite of the default iterator which iterates through major lanes of the data.
///
/// This yields `CsMinorLaneIter<'_, T, usize>` for every lane. If you want the minor index of each
/// lane alongside it, we suggest that users use `.enumerate()` on the resulting iterator.
///
/// NOTE: From a performance perspective, this iterator and [`CsMinorLaneIter`] is not necessarily
/// ideal. Compressed-Sparse formats are most effective in algorithms where the major ordering is
/// exploited. Because the matrix is compressed along the opposite dimension (the major dimension),
/// it is necessary to search backwards through the compression in order to produce the right major
/// indices for the minor lane.
///
/// All that said, this is still faster in most cases than re-allocating / re-computing the
/// opposite compression strategy, so it is provided for algorithms that need it (such as
/// sparse-matrix-multiply).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrixMinorLaneIter<'a, T> {
    current_minor_index: usize,
    minor_dim: usize,
    offsets: &'a [usize],
    indices: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Iterator for CsMatrixMinorLaneIter<'a, T> {
    type Item = CsMinorLaneIter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_minor_index < self.minor_dim {
            let minor_index = self.current_minor_index;
            self.current_minor_index += 1;

            Some(CsMinorLaneIter {
                current_major_index: 0,
                minor_index,
                offsets: self.offsets,
                indices: self.indices,
                data: self.data,
            })
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for CsMatrixMinorLaneIter<'a, T> {
    fn len(&self) -> usize {
        let nlanes = self.minor_dim;

        if nlanes > self.current_minor_index {
            nlanes - self.current_minor_index
        } else {
            0
        }
    }
}

/// An iterator representing a single lane in a `CsMatrix`.
///
/// For CSC matrices, this represents a column. For CSR matrices, this represents a row.
///
/// As an iterator yields `(usize, &T)` pairs for every element in the lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsLaneIter<'a, T> {
    current_local_index: usize,
    indices: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Iterator for CsLaneIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_local_index >= self.indices.len() {
            return None;
        }

        let local_index = self.current_local_index;

        let index = &self.indices[local_index];
        let value = &self.data[local_index];

        self.current_local_index += 1;

        Some((index.clone(), value))
    }
}

impl<'a, T> ExactSizeIterator for CsLaneIter<'a, T> {
    fn len(&self) -> usize {
        let nnz = self.indices.len();

        if nnz > self.current_local_index {
            nnz - self.current_local_index
        } else {
            0
        }
    }
}

/// An iterator representing a single minor lane in a `CsMatrix`.
///
/// For CSC matrices, this represents a row. For CSR matrices, this represents a column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMinorLaneIter<'a, T> {
    current_major_index: usize,
    minor_index: usize,
    offsets: &'a [usize],
    indices: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Iterator for CsMinorLaneIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = None;

        let nmajor = self.offsets.len();

        while self.current_major_index < nmajor {
            let offset = self.offsets[self.current_major_index];

            let (indices, data) = if self.current_major_index + 1 < nmajor {
                let offset_upper = self.offsets[self.current_major_index + 1];

                let indices = &self.indices[offset..offset_upper];
                let data = &self.data[offset..offset_upper];

                (indices, data)
            } else {
                let indices = &self.indices[offset..];
                let data = &self.data[offset..];

                (indices, data)
            };

            if let Ok(local_index) = indices.binary_search_by(|&x| x.cmp(&self.minor_index)) {
                let entry = &data[local_index];
                result = Some((self.current_major_index, entry));

                self.current_major_index += 1;
                break;
            } else {
                self.current_major_index += 1;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::*, proptest::*};
    use nalgebra::{DMatrix, SMatrix};
    use proptest::prelude::*;

    #[test]
    fn matrix_has_valid_data() {
        const NROWS: usize = 6;
        const NCOLS: usize = 3;
        const NNZ: usize = 5;

        const OFFSETS: [usize; NCOLS] = [0, 2, 2];
        const INDICES: [usize; NNZ] = [0, 5, 1, 2, 3];
        const DATA: [usize; NNZ] = [0, 1, 2, 3, 4];

        let mat = CscMatrix::try_from_parts(
            NROWS,
            NCOLS,
            OFFSETS.to_vec(),
            INDICES.to_vec(),
            DATA.to_vec(),
        )
        .unwrap();

        assert_eq!(NROWS, mat.nrows());
        assert_eq!(NCOLS, mat.ncols());
        assert_eq!(5, mat.nnz());

        let (mo, mi, d) = mat.cs_data();

        assert_eq!(mo, &OFFSETS);
        assert_eq!(mi, &INDICES);
        assert_eq!(d, &DATA);

        assert_eq!(mi.len(), mat.nnz());
        assert_eq!(d.len(), mat.nnz());

        const EXPECTED_TRIPLETS: [(usize, usize, usize); NNZ] =
            [(0, 0, 0), (0, 5, 1), (2, 1, 2), (2, 2, 3), (2, 3, 4)];

        assert!(mat.triplet_iter().zip(EXPECTED_TRIPLETS).all(
            |((major, minor, &val), (expected_major, expected_minor, expected_value))| {
                major == expected_major && minor == expected_minor && val == expected_value
            }
        ));

        let mat_iter = mat.iter();

        assert_eq!(NCOLS, mat_iter.len());

        for lane in mat_iter {
            assert!(lane.len() <= NROWS);
        }

        assert_eq!(NROWS, mat.minor_lane_iter().len());

        let (mo, mi, d) = mat.disassemble();

        assert_eq!(&mo, &OFFSETS);
        assert_eq!(&mi, &INDICES);
        assert_eq!(&d, &DATA);
    }

    #[test]
    fn empty_matrix_does_not_panic() {
        // An empty 0x0 matrix doesn't make a lot of sense in practical usage but there's no reason
        // it can't exist.
        let mat =
            CscMatrix::try_from_parts(0, 0, Vec::new(), Vec::new(), Vec::<u32>::new()).unwrap();

        assert_eq!(0, mat.nrows());
        assert_eq!(0, mat.ncols());
        assert_eq!(0, mat.nmajor());
        assert_eq!(0, mat.nminor());
        assert_eq!(0, mat.nnz());

        assert_eq!((0, 0), mat.shape());

        assert_eq!(0, mat.all_entries().len());
        assert!(mat.triplet_iter().next().is_none());
        assert_eq!(0, mat.iter().len());
        assert_eq!(0, mat.minor_lane_iter().len());
    }

    #[test]
    fn invalid_first_offset_fails_with_invalid_structure() {
        // Invalid first entry in offsets array; should be zero
        let offsets = vec![1, 2, 2];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    #[test]
    fn offsets_larger_than_ncols_fails_with_invalid_structure() {
        // Offsets has length 1 larger than the number of columns in the matrix.
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    #[test]
    fn offsets_smaller_than_ncols_fails_with_invalid_structure() {
        // Offsets has length 1 smaller than the number of columns in the matrix.
        let offsets = vec![0, 2];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    #[test]
    fn nonmonotonic_offsets_fails_with_invalid_structure() {
        let offsets = vec![0, 3, 2];
        let indices = vec![0, 1, 2, 3, 4];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    #[test]
    fn nonmonotonic_minor_indices_fails_with_invalid_structure() {
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 2, 3, 1, 4];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::InvalidStructure);
    }

    #[test]
    fn minor_index_out_of_bounds_fails_with_index_out_of_bounds() {
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 6, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::IndexOutOfBounds);
    }

    #[test]
    fn duplicate_entry_in_minor_indices_fails_with_duplicate_entry() {
        let offsets = vec![0, 2, 2];
        let indices = vec![0, 5, 2, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let error = CscMatrix::try_from_parts(6, 3, offsets, indices, values).unwrap_err();

        assert_eq!(error.kind(), &SparseFormatErrorKind::DuplicateEntry);
    }

    #[test]
    fn csc_matrix_get_entry() {
        #[rustfmt::skip]
        let dense = SMatrix::<usize, 2, 3>::from_row_slice(&[
            1, 0, 3,
            0, 5, 6
        ]);

        let csc = CscMatrix::from(&dense);

        assert_eq!(csc.get_entry(0, 0), Some(SparseEntry::NonZero(&1)));
        assert_eq!(csc.get_entry(0, 1), Some(SparseEntry::Zero));
        assert_eq!(csc.get_entry(0, 2), Some(SparseEntry::NonZero(&3)));
        assert_eq!(csc.get_entry(1, 0), Some(SparseEntry::Zero));
        assert_eq!(csc.get_entry(1, 1), Some(SparseEntry::NonZero(&5)));
        assert_eq!(csc.get_entry(1, 2), Some(SparseEntry::NonZero(&6)));

        // Check some out of bounds with .get_entry
        assert_eq!(csc.get_entry(0, 3), None);
        assert_eq!(csc.get_entry(0, 4), None);
        assert_eq!(csc.get_entry(1, 3), None);
        assert_eq!(csc.get_entry(1, 4), None);
        assert_eq!(csc.get_entry(2, 0), None);
        assert_eq!(csc.get_entry(2, 1), None);
        assert_eq!(csc.get_entry(2, 2), None);
        assert_eq!(csc.get_entry(2, 3), None);
        assert_eq!(csc.get_entry(2, 4), None);
    }

    #[test]
    fn csr_matrix_get_entry() {
        #[rustfmt::skip]
        let dense = SMatrix::<usize, 2, 3>::from_row_slice(&[
            1, 0, 3,
            0, 5, 6
        ]);

        let csr = CsrMatrix::from(&dense);

        assert_eq!(csr.get_entry(0, 0), Some(SparseEntry::NonZero(&1)));
        assert_eq!(csr.get_entry(0, 1), Some(SparseEntry::Zero));
        assert_eq!(csr.get_entry(0, 2), Some(SparseEntry::NonZero(&3)));
        assert_eq!(csr.get_entry(1, 0), Some(SparseEntry::Zero));
        assert_eq!(csr.get_entry(1, 1), Some(SparseEntry::NonZero(&5)));
        assert_eq!(csr.get_entry(1, 2), Some(SparseEntry::NonZero(&6)));

        // Check some out of bounds with .get_entry
        assert_eq!(csr.get_entry(0, 3), None);
        assert_eq!(csr.get_entry(0, 4), None);
        assert_eq!(csr.get_entry(1, 3), None);
        assert_eq!(csr.get_entry(1, 4), None);
        assert_eq!(csr.get_entry(2, 0), None);
        assert_eq!(csr.get_entry(2, 1), None);
        assert_eq!(csr.get_entry(2, 2), None);
        assert_eq!(csr.get_entry(2, 3), None);
        assert_eq!(csr.get_entry(2, 4), None);
    }

    #[test]
    fn csc_iteration_through_columns() {
        const NROWS: usize = 4;
        const NCOLS: usize = 3;

        #[rustfmt::skip]
        let dense = SMatrix::<usize, NROWS, NCOLS>::from_row_slice(&[
            0, 3, 0,
            1, 0, 4,
            2, 0, 0,
            0, 0, 5,
        ]);

        let csc = CscMatrix::from(&dense);

        assert_eq!(NROWS, csc.nrows());
        assert_eq!(NCOLS, csc.ncols());

        let mut column_iter = csc.iter();

        assert_eq!(NCOLS, column_iter.len());

        let mut first_column = column_iter.next().unwrap();
        assert_eq!(first_column.len(), 2);
        assert_eq!((1, &1), first_column.next().unwrap());
        assert_eq!((2, &2), first_column.next().unwrap());
        assert!(first_column.next().is_none());

        let mut second_column = column_iter.next().unwrap();
        assert_eq!(second_column.len(), 1);
        assert_eq!((0, &3), second_column.next().unwrap());
        assert!(second_column.next().is_none());

        let mut third_column = column_iter.next().unwrap();
        assert_eq!(third_column.len(), 2);
        assert_eq!((1, &4), third_column.next().unwrap());
        assert_eq!((3, &5), third_column.next().unwrap());
        assert!(third_column.next().is_none());

        assert!(column_iter.next().is_none());
    }

    #[test]
    fn csc_iteration_through_rows() {
        const NROWS: usize = 4;
        const NCOLS: usize = 3;

        #[rustfmt::skip]
        let dense = SMatrix::<usize, NROWS, NCOLS>::from_row_slice(&[
            0, 3, 0,
            1, 0, 4,
            2, 0, 0,
            0, 0, 5,
        ]);

        let csc = CscMatrix::from(&dense);

        assert_eq!(NROWS, csc.nrows());
        assert_eq!(NCOLS, csc.ncols());

        let mut row_iter = csc.minor_lane_iter();

        assert_eq!(NROWS, row_iter.len());

        let mut first_row = row_iter.next().unwrap();
        assert_eq!((1, &3), first_row.next().unwrap());
        assert!(first_row.next().is_none());

        let mut second_row = row_iter.next().unwrap();
        assert_eq!((0, &1), second_row.next().unwrap());
        assert_eq!((2, &4), second_row.next().unwrap());
        assert!(second_row.next().is_none());

        let mut third_row = row_iter.next().unwrap();
        assert_eq!((0, &2), third_row.next().unwrap());
        assert!(third_row.next().is_none());

        let mut fourth_row = row_iter.next().unwrap();
        assert_eq!((2, &5), fourth_row.next().unwrap());
        assert!(fourth_row.next().is_none());

        assert!(row_iter.next().is_none());
    }

    #[test]
    fn csr_iteration_through_columns() {
        const NROWS: usize = 4;
        const NCOLS: usize = 3;

        #[rustfmt::skip]
        let dense = SMatrix::<usize, NROWS, NCOLS>::from_row_slice(&[
            0, 3, 0,
            1, 0, 4,
            2, 0, 0,
            0, 0, 5,
        ]);

        let csr = CsrMatrix::from(&dense);

        assert_eq!(NROWS, csr.nrows());
        assert_eq!(NCOLS, csr.ncols());

        let mut column_iter = csr.minor_lane_iter();

        assert_eq!(NCOLS, column_iter.len());

        let mut first_column = column_iter.next().unwrap();
        assert_eq!((1, &1), first_column.next().unwrap());
        assert_eq!((2, &2), first_column.next().unwrap());
        assert!(first_column.next().is_none());

        let mut second_column = column_iter.next().unwrap();
        assert_eq!((0, &3), second_column.next().unwrap());
        assert!(second_column.next().is_none());

        let mut third_column = column_iter.next().unwrap();
        assert_eq!((1, &4), third_column.next().unwrap());
        assert_eq!((3, &5), third_column.next().unwrap());
        assert!(third_column.next().is_none());

        assert!(column_iter.next().is_none());
    }

    #[test]
    fn csr_iteration_through_rows() {
        const NROWS: usize = 4;
        const NCOLS: usize = 3;

        #[rustfmt::skip]
        let dense = SMatrix::<usize, NROWS, NCOLS>::from_row_slice(&[
            0, 3, 0,
            1, 0, 4,
            2, 0, 0,
            0, 0, 5,
        ]);

        let csr = CsrMatrix::from(&dense);

        assert_eq!(NROWS, csr.nrows());
        assert_eq!(NCOLS, csr.ncols());

        let mut row_iter = csr.iter();

        assert_eq!(NROWS, row_iter.len());

        let mut first_row = row_iter.next().unwrap();
        assert_eq!(1, first_row.len());
        assert_eq!((1, &3), first_row.next().unwrap());
        assert!(first_row.next().is_none());

        let mut second_row = row_iter.next().unwrap();
        assert_eq!(2, second_row.len());
        assert_eq!((0, &1), second_row.next().unwrap());
        assert_eq!((2, &4), second_row.next().unwrap());
        assert!(second_row.next().is_none());

        let mut third_row = row_iter.next().unwrap();
        assert_eq!(1, third_row.len());
        assert_eq!((0, &2), third_row.next().unwrap());
        assert!(third_row.next().is_none());

        let mut fourth_row = row_iter.next().unwrap();
        assert_eq!(1, fourth_row.len());
        assert_eq!((2, &5), fourth_row.next().unwrap());
        assert!(fourth_row.next().is_none());

        assert!(row_iter.next().is_none());
    }

    proptest! {
        #[test]
        fn csc_double_transpose_is_identity(csc in csc_strategy()) {
            let csc_t = csc.transpose();
            let csc_t_t = csc_t.transpose();

            let (offsets, indices, data) = csc.cs_data();
            let (t_offsets, t_indices, t_data) = csc_t_t.cs_data();

            prop_assert_eq!(csc.shape(), csc_t_t.shape());

            prop_assert!(offsets.iter().zip(t_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(t_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(t_data).all(|(a, b)| a == b));
        }

        #[test]
        fn csr_double_transpose_is_identity(csr in csr_strategy()) {
            let csr_t = csr.transpose();
            let csr_t_t = csr_t.transpose();

            let (offsets, indices, data) = csr.cs_data();
            let (t_offsets, t_indices, t_data) = csr_t_t.cs_data();

            prop_assert_eq!(csr.shape(), csr_t_t.shape());

            prop_assert!(offsets.iter().zip(t_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(t_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(t_data).all(|(a, b)| a == b));
        }

        #[test]
        fn csc_transpose_agrees_with_dense(csc in csc_strategy()) {
            let dense_transpose = DMatrix::from(&csc).transpose();
            let csc_transpose = csc.transpose();

            prop_assert_eq!(dense_transpose, DMatrix::from(&csc_transpose));
            prop_assert_eq!(csc.nnz(), csc_transpose.nnz());
        }

        #[test]
        fn csr_transpose_agrees_with_dense(csr in csr_strategy()) {
            let dense_transpose = DMatrix::from(&csr).transpose();
            let csr_transpose = csr.transpose();

            prop_assert_eq!(dense_transpose, DMatrix::from(&csr_transpose));
            prop_assert_eq!(csr.nnz(), csr_transpose.nnz());
        }

        #[test]
        fn zero_matrix_valid_data(nrows in 0..500usize, ncols in 0..500usize) {
            let mat = CsrMatrix::<f32>::zeros(nrows, ncols);

            prop_assert_eq!(nrows, mat.nrows());
            prop_assert_eq!(ncols, mat.ncols());

            prop_assert_eq!((nrows, ncols), mat.shape());

            prop_assert_eq!(nrows, mat.nmajor());
            prop_assert_eq!(ncols, mat.nminor());

            prop_assert_eq!(0, mat.nnz());

            let (offsets, indices, data) = mat.cs_data();

            prop_assert!(offsets.iter().all(|&o| o == 0usize));
            prop_assert_eq!(offsets.len(), nrows);
            prop_assert_eq!(indices, &[]);
            prop_assert_eq!(data, &[]);

            prop_assert!(mat.triplet_iter().next().is_none());

            prop_assert_eq!(nrows, mat.iter().len());

            for mut lane in mat.iter() {
                prop_assert_eq!(0, lane.len());
                prop_assert!(lane.next().is_none());
            }

            prop_assert_eq!(ncols, mat.minor_lane_iter().len());

            for mut lane in mat.minor_lane_iter() {
                prop_assert!(lane.next().is_none());
            }

            prop_assert_eq!(nrows * ncols, mat.all_entries().len());

            for (major, minor, entry) in mat.all_entries() {
                prop_assert!(major < nrows);
                prop_assert!(minor < ncols);

                match entry {
                    SparseEntry::Zero => (),
                    _ => unreachable!(),
                }
            }
        }

        #[test]
        fn identity_matrix_valid_data(n in 0..500usize) {
            let mat = CsrMatrix::<f32>::identity(n);

            prop_assert_eq!(n, mat.nrows());
            prop_assert_eq!(n, mat.ncols());

            prop_assert_eq!((n, n), mat.shape());

            prop_assert_eq!(n, mat.nmajor());
            prop_assert_eq!(n, mat.nminor());

            prop_assert_eq!(n, mat.nnz());

            for (major, minor, &val) in mat.triplet_iter() {
                prop_assert_eq!(val, f32::one());
                prop_assert_eq!(major, minor);
            }

            prop_assert_eq!(n, mat.iter().len());

            for mut lane in mat.iter() {
                prop_assert_eq!(1, lane.len());

                prop_assert!(lane.next().is_some());
                prop_assert!(lane.next().is_none());
            }

            prop_assert_eq!(n, mat.minor_lane_iter().len());

            for mut lane in mat.minor_lane_iter() {
                prop_assert!(lane.next().is_some());
                prop_assert!(lane.next().is_none());
            }

            prop_assert_eq!(n * n, mat.all_entries().len());

            for (major, minor, entry) in mat.all_entries() {
                prop_assert!(major < n);
                prop_assert!(minor < n);

                match entry {
                    SparseEntry::NonZero(&val) => {
                        prop_assert_eq!(val, f32::one());
                        prop_assert_eq!(major, minor);
                    }
                    SparseEntry::Zero => {
                        prop_assert_ne!(major, minor);
                    }
                }
            }
        }
    }
}
