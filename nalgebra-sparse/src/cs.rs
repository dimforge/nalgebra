//! A type for representing compressed sparse (row-major / column-major) matrices.

use super::{
    error::{SparseFormatError, SparsityPatternFormatError},
    SparseEntry,
};
use num_traits::{One, Unsigned};
use std::{borrow::Borrow, cmp::Ord, cmp::Ordering, marker::PhantomData, ops::Add};

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
/// - `Offset`: An unsigned numeric type for describing how many elements in `indices` and `data`
/// to traverse to get to the start of a given lane.
/// - `Index`: An unsigned numeric type for describing the positions of elements in each minor axis
/// lane.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrix<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index = Offset>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
    MajorOffsets: Borrow<[Offset]>,
    MinorIndices: Borrow<[Index]>,
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
    /// `MajorOffsets` is a type that can be borrowed as `&[Offset]`, which should represent an
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
    /// in `data`. `MinorIndices` is a type that can be borrowed as `&[Index]`, which should
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
    _phantom: PhantomData<(T, Offset, Index, CompressionKind)>,
}

/// An alias for producing an owned, row-major compressed sparse matrix.
pub type CsrMatrix<T, O, I = O> = CsMatrix<T, O, Vec<O>, Vec<I>, Vec<T>, CompressedRowStorage, I>;

/// An alias for producing an owned, column-major compressed sparse matrix.
pub type CscMatrix<T, O, I = O> =
    CsMatrix<T, O, Vec<O>, Vec<I>, Vec<T>, CompressedColumnStorage, I>;

impl<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index>
    CsMatrix<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
    MajorOffsets: Borrow<[Offset]>,
    MinorIndices: Borrow<[Index]>,
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

    unsafe fn from_parts_unchecked(
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
            if first != &Offset::zero() {
                // First entry exists and is not zero
                return Err(SparsityPatternFormatError::InvalidFirstOffset.into());
            }
        }

        if borrowed_indices.len() != data.borrow().len() {
            // size mismatch
            return Err(SparsityPatternFormatError::DataAndIndicesSizeMismatch.into());
        }

        if borrowed_indices.iter().any(|&index| index.into() >= nminor) {
            // Index out-of-bounds
            return Err(SparsityPatternFormatError::MinorIndexOutOfBounds.into());
        }

        for major_index in 0..nmajor {
            let lower = borrowed_offsets[major_index].into();

            let lane_indices = if major_index + 1 < nmajor {
                let upper = borrowed_offsets[major_index + 1].into();

                if lower > upper {
                    // Offsets do not monotonically increase
                    return Err(SparsityPatternFormatError::NonmonotonicOffsets.into());
                }

                &borrowed_indices[lower..upper]
            } else {
                &borrowed_indices[lower..]
            };

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

        Ok(unsafe { Self::from_parts_unchecked(nrows, ncols, offsets, indices, data) })
    }

    /// Consumes self and returns the underlying major offsets, minor indices, and data contained
    /// within the sparse matrix format.
    pub fn disassemble(self) -> (MajorOffsets, MinorIndices, Data) {
        (self.offsets, self.indices, self.data)
    }

    /// Borrows self and returns three slices to the major offsets, minor indices, and data
    /// contained within the sparse matrix format.
    pub fn cs_data(&self) -> (&[Offset], &[Index], &[T]) {
        (
            self.offsets.borrow(),
            self.indices.borrow(),
            self.data.borrow(),
        )
    }

    /// Produces an immutable view of the transpose of the data by borrowing the underlying lanes
    /// and sparsity pattern data.
    pub fn transpose(
        &self,
    ) -> CsMatrix<T, Offset, &[Offset], &[Index], &[T], CompressionKind::Transpose, Index> {
        let (nrows, ncols) = self.shape;
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
    pub fn get_entry(&self, major_index: usize, minor_index: usize) -> Option<SparseEntry<'_, T>> {
        let nmajor = self.nmajor();
        let nminor = self.nminor();

        let major_index: usize = major_index.into();
        let minor_index: usize = minor_index.into();

        if major_index >= nmajor || minor_index >= nminor {
            return None;
        }

        let offset = self.offsets.borrow()[major_index].into();

        let (indices, data) = if major_index + 1 < self.nmajor() {
            let offset_upper = self.offsets.borrow()[major_index + 1].into();

            let indices = &self.indices.borrow()[offset..offset_upper];
            let data = &self.data.borrow()[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices.borrow()[offset..];
            let data = &self.data.borrow()[offset..];

            (indices, data)
        };

        let entry =
            if let Ok(local_index) = indices.binary_search_by(|&x| x.into().cmp(&minor_index)) {
                SparseEntry::NonZero(&data[local_index])
            } else {
                SparseEntry::Zero
            };

        Some(entry)
    }

    /// An iterator that iterates through every implicit and explicit entry in the matrix.
    pub fn all_entries(&self) -> AllElementsIter<'_, T, Offset, Index> {
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
            lane.map(move |(minor_index, value)| (major_index, minor_index.into(), value))
        })
    }

    /// Gets a major-axis lane of the data given a major index.
    ///
    /// Returns `None` iff the major index does not correspond to a lane in the `CsMatrix` (e.g. if
    /// you ask for lane 100 on a 3x3 sparse matrix).
    pub fn get_lane(&self, major_index: usize) -> Option<CsLaneIter<'_, T, Index>> {
        if major_index >= self.nmajor() {
            return None;
        }

        let offset = self.offsets.borrow()[major_index].into();

        let (indices, data) = if major_index + 1 < self.nmajor() {
            let offset_upper = self.offsets.borrow()[major_index + 1].into();

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

    /// An iterator that iterates across every lane of the `CsMatrix`, in order.
    pub fn iter(&self) -> CsMatrixIter<'_, T, Offset, Index> {
        let (offsets, indices, data) = self.cs_data();

        CsMatrixIter {
            current_major_index: 0,
            offsets,
            indices,
            data,
        }
    }
}

impl<T, O, I, C> CsMatrix<T, O, Vec<O>, Vec<I>, Vec<T>, C, I>
where
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord + Clone,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    C: Compression,
{
    /// Returns an owned `CsMatrix` of shape `(nrows, ncols)` entirely comprised of implicit zeros,
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let nmajor = C::nmajor(nrows, ncols);

        Self {
            shape: (nrows, ncols),
            offsets: vec![O::zero(); nmajor + 1],
            indices: Vec::new(),
            data: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl<T> CscMatrix<T, usize, usize>
where
    T: One + Clone,
{
    /// Produces an owned identity matrix of shape `(n, n)` in CSC format.
    #[inline]
    pub fn identity(n: usize) -> Self {
        let offsets = (0..=n).collect();
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
pub struct AllElementsIter<'a, T, Offset, Index>
where
    Offset: Into<usize> + Unsigned + Ord,
    Index: Into<usize> + Unsigned + Ord,
{
    current_major_index: usize,
    current_minor_index: usize,
    minor_length: usize,
    offsets: &'a [Offset],
    indices: &'a [Index],
    data: &'a [T],
}

impl<'a, T, Offset, Index> Iterator for AllElementsIter<'a, T, Offset, Index>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    type Item = (usize, usize, SparseEntry<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_major_index >= self.offsets.len() {
            return None;
        }

        let major_index = self.current_major_index;
        let minor_index = self.current_minor_index;

        let offset = self.offsets[major_index].into();

        let (indices, data) = if major_index + 1 < self.offsets.len() {
            let offset_upper = self.offsets[major_index + 1].into();

            let indices = &self.indices[offset..offset_upper];
            let data = &self.data[offset..offset_upper];

            (indices, data)
        } else {
            let indices = &self.indices[offset..];
            let data = &self.data[offset..];

            (indices, data)
        };

        let entry =
            if let Ok(local_index) = indices.binary_search_by(|&x| x.into().cmp(&minor_index)) {
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

impl<'a, T, Offset, Index> ExactSizeIterator for AllElementsIter<'a, T, Offset, Index>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
{
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
/// This yields `CsLaneIter<'_, T, Index>` for every lane. If you want the major index of each lane
/// alongside it, we suggest that users use `.enumerate()` on the resulting iterator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrixIter<'a, T, Offset, Index>
where
    Offset: Unsigned + Ord,
    Index: Unsigned + Ord,
{
    current_major_index: usize,
    offsets: &'a [Offset],
    indices: &'a [Index],
    data: &'a [T],
}

impl<'a, T, Offset, Index> Iterator for CsMatrixIter<'a, T, Offset, Index>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    type Item = CsLaneIter<'a, T, Index>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_major_index >= self.offsets.len() {
            return None;
        }

        let offset = self.offsets[self.current_major_index].into();

        let (indices, data) = if self.current_major_index + 1 < self.offsets.len() {
            let offset_upper = self.offsets[self.current_major_index + 1].into();

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

impl<'a, T, Offset, Index> ExactSizeIterator for CsMatrixIter<'a, T, Offset, Index>
where
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn len(&self) -> usize {
        let nlanes = self.offsets.len();

        if nlanes > self.current_major_index {
            nlanes - self.current_major_index
        } else {
            0
        }
    }
}

/// An iterator representing a single lane in a `CsMatrix`.
///
/// As an iterator yields `(Index, &T)` pairs for every element in the lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsLaneIter<'a, T, Index>
where
    Index: Unsigned + Ord,
{
    current_local_index: usize,
    indices: &'a [Index],
    data: &'a [T],
}

impl<'a, T, Index> Iterator for CsLaneIter<'a, T, Index>
where
    Index: Clone + Unsigned + Ord,
{
    type Item = (Index, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_local_index > self.indices.len() {
            return None;
        }

        let local_index = self.current_local_index;

        let index = &self.indices[local_index];
        let value = &self.data[local_index];

        self.current_local_index += 1;

        Some((index.clone(), value))
    }
}

impl<'a, T, Index> ExactSizeIterator for CsLaneIter<'a, T, Index>
where
    Index: Clone + Unsigned + Ord,
{
    fn len(&self) -> usize {
        let nnz = self.indices.len();

        if nnz > self.current_local_index {
            nnz - self.current_local_index
        } else {
            0
        }
    }
}
