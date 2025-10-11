//! Iterator types for traversing matrix elements, rows, and columns.
//!
//! This module provides several iterator types for working with matrices:
//!
//! - [`MatrixIter`] and [`MatrixIterMut`]: Iterate over individual matrix elements
//! - [`RowIter`] and [`RowIterMut`]: Iterate over matrix rows
//! - [`ColumnIter`] and [`ColumnIterMut`]: Iterate over matrix columns
//!
//! All iterators in this module implement standard iterator traits including
//! [`Iterator`], [`ExactSizeIterator`], and most also implement [`DoubleEndedIterator`],
//! allowing both forward and backward iteration.

use core::fmt::Debug;
use core::ops::Range;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem;

use crate::base::dimension::{Dim, U1};
use crate::base::storage::{RawStorage, RawStorageMut};
use crate::base::{Matrix, MatrixView, MatrixViewMut, Scalar, ViewStorage, ViewStorageMut};

#[derive(Clone, Debug)]
struct RawIter<Ptr, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> {
    ptr: Ptr,
    inner_ptr: Ptr,
    inner_end: Ptr,
    size: usize,
    strides: (RStride, CStride),
    _phantoms: PhantomData<(fn() -> T, R, C)>,
}

macro_rules! iterator {
    (struct $Name:ident for $Storage:ident.$ptr: ident -> $Ptr:ty, $Ref:ty, $SRef: ty, $($derives:ident),* $(,)?) => {
        // TODO: we need to specialize for the case where the matrix storage is owned (in which
        // case the iterator is trivial because it does not have any stride).
        impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
            RawIter<$Ptr, T, R, C, RStride, CStride>
        {
            /// Creates a new iterator for the given matrix storage.
            fn new<'a, S: $Storage<T, R, C, RStride = RStride, CStride = CStride>>(
                storage: $SRef,
            ) -> Self {
                let shape = storage.shape();
                let strides = storage.strides();
                let inner_offset = shape.0.value() * strides.0.value();
                let size = shape.0.value() * shape.1.value();
                let ptr = storage.$ptr();

                // If we have a size of 0, 'ptr' must be
                // dangling. However, 'inner_offset' might
                // not be zero if only one dimension is zero, so
                // we don't want to call 'offset'.
                // This pointer will never actually get used
                // if our size is '0', so it's fine to use
                // 'ptr' for both the start and end.
                let inner_end = if size == 0 {
                    ptr
                } else {
                    // Safety:
                    // If 'size' is non-zero, we know that 'ptr'
                    // is not dangling, and 'inner_offset' must lie
                    // within the allocation
                    unsafe { ptr.add(inner_offset) }
                };

                RawIter {
                    ptr,
                    inner_ptr: ptr,
                    inner_end,
                    size: shape.0.value() * shape.1.value(),
                    strides,
                    _phantoms: PhantomData,
                }
            }
        }

        impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Iterator
            for RawIter<$Ptr, T, R, C, RStride, CStride>
        {
            type Item = $Ptr;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.size == 0 {
                        None
                    } else {
                        self.size -= 1;

                        // Jump to the next outer dimension if needed.
                        if self.ptr == self.inner_end {
                            let stride = self.strides.1.value() as isize;
                            // This might go past the end of the allocation,
                            // depending on the value of 'size'. We use
                            // `wrapping_offset` to avoid UB
                            self.inner_end = self.ptr.wrapping_offset(stride);
                            // This will always be in bounds, since
                            // we're going to dereference it
                            self.ptr = self.inner_ptr.offset(stride);
                            self.inner_ptr = self.ptr;
                        }

                        // Go to the next element.
                        let old = self.ptr;

                        // Don't offset `self.ptr` for the last element,
                        // as this will be out of bounds. Iteration is done
                        // at this point (the next call to `next` will return `None`)
                        // so this is not observable.
                        if self.size != 0 {
                            let stride = self.strides.0.value();
                            self.ptr = self.ptr.add(stride);
                        }

                        Some(old)
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.size, Some(self.size))
            }

            #[inline]
            fn count(self) -> usize {
                self.size_hint().0
            }
        }

        impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> DoubleEndedIterator
            for RawIter<$Ptr, T, R, C, RStride, CStride>
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.size == 0 {
                        None
                    } else {
                        // Pre-decrement `size` such that it now counts to the
                        // element we want to return.
                        self.size -= 1;

                        // Fetch strides
                        let inner_stride = self.strides.0.value();
                        let outer_stride = self.strides.1.value();

                        // Compute number of rows
                        // Division should be exact
                        let inner_raw_size = self.inner_end.offset_from(self.inner_ptr) as usize;
                        let inner_size = inner_raw_size / inner_stride;

                        // Compute rows and cols remaining
                        let outer_remaining = self.size / inner_size;
                        let inner_remaining = self.size % inner_size;

                        // Compute pointer to last element
                        let last = self
                            .ptr
                            .add((outer_remaining * outer_stride + inner_remaining * inner_stride));

                        Some(last)
                    }
                }
            }
        }

        impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ExactSizeIterator
            for RawIter<$Ptr, T, R, C, RStride, CStride>
        {
            #[inline]
            fn len(&self) -> usize {
                self.size
            }
        }

        impl<T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> FusedIterator
            for RawIter<$Ptr, T, R, C, RStride, CStride>
        {
        }

        /// An iterator through the elements of a dense matrix with arbitrary strides.
        ///
        /// This iterator traverses matrix elements in column-major order (down columns, then across rows).
        /// It works with matrices that have custom memory layouts defined by row and column strides.
        ///
        /// # Iteration Order
        ///
        /// Elements are visited in column-major order:
        /// - First, all elements of the first column (top to bottom)
        /// - Then, all elements of the second column (top to bottom)
        /// - And so on, until the last column
        ///
        /// # Iterator Traits
        ///
        /// This iterator implements:
        /// - [`Iterator`]: Basic forward iteration
        /// - [`DoubleEndedIterator`]: Allows iterating from both ends
        /// - [`ExactSizeIterator`]: Provides exact count of remaining elements
        /// - [`FusedIterator`]: Guarantees `None` is returned forever after the first `None`
        ///
        /// # Examples
        ///
        /// Basic iteration over matrix elements:
        ///
        /// ```
        /// use nalgebra::Matrix2x3;
        ///
        /// let matrix = Matrix2x3::from_row_slice(&[
        ///     1, 2, 3,
        ///     4, 5, 6,
        /// ]);
        ///
        /// // Iterate in column-major order
        /// let mut iter = matrix.iter();
        /// assert_eq!(*iter.next().unwrap(), 1);  // column 0, row 0
        /// assert_eq!(*iter.next().unwrap(), 4);  // column 0, row 1
        /// assert_eq!(*iter.next().unwrap(), 2);  // column 1, row 0
        /// assert_eq!(*iter.next().unwrap(), 5);  // column 1, row 1
        /// assert_eq!(*iter.next().unwrap(), 3);  // column 2, row 0
        /// assert_eq!(*iter.next().unwrap(), 6);  // column 2, row 1
        /// assert_eq!(iter.next(), None);
        /// ```
        ///
        /// Collecting elements into a vector:
        ///
        /// ```
        /// use nalgebra::Matrix2x3;
        ///
        /// let matrix = Matrix2x3::from_row_slice(&[
        ///     1, 2, 3,
        ///     4, 5, 6,
        /// ]);
        ///
        /// let elements: Vec<i32> = matrix.iter().copied().collect();
        /// assert_eq!(elements, vec![1, 4, 2, 5, 3, 6]);  // column-major order
        /// ```
        ///
        /// Using iterator methods for computation:
        ///
        /// ```
        /// use nalgebra::Matrix3;
        ///
        /// let matrix = Matrix3::new(
        ///     1, 2, 3,
        ///     4, 5, 6,
        ///     7, 8, 9,
        /// );
        ///
        /// // Sum all elements
        /// let sum: i32 = matrix.iter().sum();
        /// assert_eq!(sum, 45);
        ///
        /// // Find maximum element
        /// let max = matrix.iter().max();
        /// assert_eq!(max, Some(&9));
        ///
        /// // Count elements greater than 5
        /// let count = matrix.iter().filter(|&&x| x > 5).count();
        /// assert_eq!(count, 4);
        /// ```
        ///
        /// Iterating from both ends:
        ///
        /// ```
        /// use nalgebra::Matrix2x3;
        ///
        /// let matrix = Matrix2x3::from_row_slice(&[
        ///     1, 2, 3,
        ///     4, 5, 6,
        /// ]);
        ///
        /// let mut iter = matrix.iter();
        /// assert_eq!(*iter.next().unwrap(), 1);      // first element
        /// assert_eq!(*iter.next_back().unwrap(), 6); // last element
        /// assert_eq!(*iter.next().unwrap(), 4);      // second element
        /// assert_eq!(*iter.next_back().unwrap(), 3); // second-to-last
        /// ```
        ///
        /// # See Also
        ///
        /// - [`Matrix::iter`](crate::Matrix::iter): Creates this iterator
        /// - [`MatrixIterMut`]: Mutable version of this iterator
        /// - [`RowIter`]: Iterator over matrix rows
        /// - [`ColumnIter`]: Iterator over matrix columns
        #[derive($($derives),*)]
        pub struct $Name<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> {
            inner: RawIter<$Ptr, T, R, C, S::RStride, S::CStride>,
            _marker: PhantomData<$Ref>,
        }

        impl<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> $Name<'a, T, R, C, S> {
            /// Creates a new iterator for the given matrix storage.
            ///
            /// This method is typically not called directly. Instead, use
            /// [`Matrix::iter`](crate::Matrix::iter) or [`Matrix::iter_mut`](crate::Matrix::iter_mut)
            /// to create iterators for your matrices.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2x3;
            ///
            /// let matrix = Matrix2x3::from_row_slice(&[
            ///     1, 2, 3,
            ///     4, 5, 6,
            /// ]);
            ///
            /// // Use the convenient iter() method instead of calling new() directly
            /// let sum: i32 = matrix.iter().sum();
            /// assert_eq!(sum, 21);
            /// ```
            ///
            /// # See Also
            ///
            /// - [`Matrix::iter`](crate::Matrix::iter): Convenient method to create this iterator
            /// - [`Matrix::iter_mut`](crate::Matrix::iter_mut): Creates a mutable iterator
            pub fn new(storage: $SRef) -> Self {
                Self {
                    inner: RawIter::<$Ptr, T, R, C, S::RStride, S::CStride>::new(storage),
                    _marker: PhantomData,
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> Iterator for $Name<'a, T, R, C, S> {
            type Item = $Ref;

            /// Advances the iterator and returns the next element.
            ///
            /// Returns `None` when iteration is complete.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2;
            ///
            /// let matrix = Matrix2::new(
            ///     1, 2,
            ///     3, 4,
            /// );
            ///
            /// let mut iter = matrix.iter();
            /// assert_eq!(*iter.next().unwrap(), 1);
            /// assert_eq!(*iter.next().unwrap(), 3);
            /// assert_eq!(*iter.next().unwrap(), 2);
            /// assert_eq!(*iter.next().unwrap(), 4);
            /// assert_eq!(iter.next(), None);
            /// ```
            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                // We want either `& *last` or `&mut *last` here, depending
                // on the mutability of `$Ref`.
                #[allow(clippy::transmute_ptr_to_ref)]
                self.inner.next().map(|ptr| unsafe { mem::transmute(ptr) })
            }

            /// Returns the bounds on the remaining length of the iterator.
            ///
            /// Returns a tuple where the first element is the lower bound and the second
            /// element is the upper bound. For this iterator, both bounds are always exact.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2x3;
            ///
            /// let matrix = Matrix2x3::from_row_slice(&[
            ///     1, 2, 3,
            ///     4, 5, 6,
            /// ]);
            ///
            /// let mut iter = matrix.iter();
            /// assert_eq!(iter.size_hint(), (6, Some(6)));
            ///
            /// iter.next();
            /// assert_eq!(iter.size_hint(), (5, Some(5)));
            /// ```
            #[inline(always)]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }

            /// Consumes the iterator and returns the number of remaining elements.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2x3;
            ///
            /// let matrix = Matrix2x3::from_row_slice(&[
            ///     1, 2, 3,
            ///     4, 5, 6,
            /// ]);
            ///
            /// let mut iter = matrix.iter();
            /// iter.next();
            /// iter.next();
            /// assert_eq!(iter.count(), 4);
            /// ```
            #[inline(always)]
            fn count(self) -> usize {
                self.inner.count()
            }
        }

        impl<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> DoubleEndedIterator
            for $Name<'a, T, R, C, S>
        {
            /// Removes and returns an element from the end of the iterator.
            ///
            /// Returns `None` when there are no more elements.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2x3;
            ///
            /// let matrix = Matrix2x3::from_row_slice(&[
            ///     1, 2, 3,
            ///     4, 5, 6,
            /// ]);
            ///
            /// let mut iter = matrix.iter();
            /// assert_eq!(*iter.next_back().unwrap(), 6);  // last element
            /// assert_eq!(*iter.next_back().unwrap(), 3);  // second-to-last
            /// assert_eq!(*iter.next().unwrap(), 1);       // first element
            /// ```
            #[inline(always)]
            fn next_back(&mut self) -> Option<Self::Item> {
                // We want either `& *last` or `&mut *last` here, depending
                // on the mutability of `$Ref`.
                #[allow(clippy::transmute_ptr_to_ref)]
                self.inner
                    .next_back()
                    .map(|ptr| unsafe { mem::transmute(ptr) })
            }
        }

        impl<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> ExactSizeIterator
            for $Name<'a, T, R, C, S>
        {
            /// Returns the exact number of elements remaining in the iterator.
            ///
            /// # Examples
            ///
            /// ```
            /// use nalgebra::Matrix2x3;
            ///
            /// let matrix = Matrix2x3::from_row_slice(&[
            ///     1, 2, 3,
            ///     4, 5, 6,
            /// ]);
            ///
            /// let mut iter = matrix.iter();
            /// assert_eq!(iter.len(), 6);
            ///
            /// iter.next();
            /// assert_eq!(iter.len(), 5);
            ///
            /// iter.next();
            /// iter.next();
            /// assert_eq!(iter.len(), 3);
            /// ```
            #[inline(always)]
            fn len(&self) -> usize {
                self.inner.len()
            }
        }

        impl<'a, T, R: Dim, C: Dim, S: 'a + $Storage<T, R, C>> FusedIterator
            for $Name<'a, T, R, C, S>
        {
        }
    };
}

iterator!(struct MatrixIter for RawStorage.ptr -> *const T, &'a T, &'a S, Clone, Debug);
iterator!(struct MatrixIterMut for RawStorageMut.ptr_mut -> *mut T, &'a mut T, &'a mut S, Debug);

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixIter<'a, T, R, C, ViewStorage<'a, T, R, C, RStride, CStride>>
{
    /// Creates a new iterator from an owned matrix storage view.
    ///
    /// This is a specialized constructor for creating iterators from owned
    /// [`ViewStorage`] instances. Typically, you should use [`Matrix::iter`](crate::Matrix::iter)
    /// instead of calling this method directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// // Use the convenient iter() method instead
    /// let elements: Vec<i32> = matrix.iter().copied().collect();
    /// assert_eq!(elements, vec![1, 4, 2, 5, 3, 6]);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Matrix::iter`](crate::Matrix::iter): Convenient method to create iterators
    /// - [`MatrixIter::new`]: General constructor for this iterator
    pub fn new_owned(storage: ViewStorage<'a, T, R, C, RStride, CStride>) -> Self {
        Self {
            inner: RawIter::<*const T, T, R, C, RStride, CStride>::new(&storage),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixIterMut<'a, T, R, C, ViewStorageMut<'a, T, R, C, RStride, CStride>>
{
    /// Creates a new mutable iterator from an owned matrix storage view.
    ///
    /// This is a specialized constructor for creating mutable iterators from owned
    /// [`ViewStorageMut`] instances. Typically, you should use [`Matrix::iter_mut`](crate::Matrix::iter_mut)
    /// instead of calling this method directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// // Use the convenient iter_mut() method to modify elements
    /// for elem in matrix.iter_mut() {
    ///     *elem *= 2;
    /// }
    ///
    /// assert_eq!(matrix, Matrix2x3::from_row_slice(&[
    ///     2, 4, 6,
    ///     8, 10, 12,
    /// ]));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Matrix::iter_mut`](crate::Matrix::iter_mut): Convenient method to create mutable iterators
    /// - [`MatrixIterMut::new`]: General constructor for this iterator
    pub fn new_owned_mut(mut storage: ViewStorageMut<'a, T, R, C, RStride, CStride>) -> Self {
        Self {
            inner: RawIter::<*mut T, T, R, C, RStride, CStride>::new(&mut storage),
            _marker: PhantomData,
        }
    }
}

/*
 *
 * Row iterators.
 *
 */
#[derive(Clone, Debug)]
/// An iterator through the rows of a matrix.
///
/// This iterator yields row views (1 x C matrices) from a matrix. Each element returned
/// is a [`MatrixView`] representing one complete row of the source matrix.
///
/// # Iteration Order
///
/// Rows are yielded from top to bottom (row 0, then row 1, etc.).
///
/// # Iterator Traits
///
/// This iterator implements:
/// - [`Iterator`]: Basic forward iteration
/// - [`ExactSizeIterator`]: Provides exact count of remaining rows
///
/// # Examples
///
/// Basic iteration over rows:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// let mut row_iter = matrix.row_iter();
///
/// // First row
/// let row0 = row_iter.next().unwrap();
/// assert_eq!(row0, Matrix3x2::from_row_slice(&[1, 2]).row(0));
/// assert_eq!(row0[(0, 0)], 1);
/// assert_eq!(row0[(0, 1)], 2);
///
/// // Second row
/// let row1 = row_iter.next().unwrap();
/// assert_eq!(row1[(0, 0)], 3);
/// assert_eq!(row1[(0, 1)], 4);
///
/// // Third row
/// let row2 = row_iter.next().unwrap();
/// assert_eq!(row2[(0, 0)], 5);
/// assert_eq!(row2[(0, 1)], 6);
///
/// assert!(row_iter.next().is_none());
/// ```
///
/// Computing row sums:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// let row_sums: Vec<i32> = matrix.row_iter()
///     .map(|row| row.iter().sum())
///     .collect();
///
/// assert_eq!(row_sums, vec![3, 7, 11]);
/// ```
///
/// Finding rows that meet a condition:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// // Count rows where all elements are greater than 2
/// let count = matrix.row_iter()
///     .filter(|row| row.iter().all(|&x| x > 2))
///     .count();
///
/// assert_eq!(count, 2);  // rows [3, 4] and [5, 6]
/// ```
///
/// Collecting rows into a vector:
///
/// ```
/// use nalgebra::{Matrix3x2, MatrixView, U1, U2};
///
/// let matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// let rows: Vec<_> = matrix.row_iter().collect();
/// assert_eq!(rows.len(), 3);
/// assert_eq!(rows[0][(0, 0)], 1);
/// assert_eq!(rows[1][(0, 0)], 3);
/// assert_eq!(rows[2][(0, 0)], 5);
/// ```
///
/// # See Also
///
/// - [`Matrix::row_iter`](crate::Matrix::row_iter): Creates this iterator
/// - [`RowIterMut`]: Mutable version of this iterator
/// - [`ColumnIter`]: Iterator over columns instead of rows
/// - [`Matrix::row`](crate::Matrix::row): Access a single row
pub struct RowIter<'a, T, R: Dim, C: Dim, S: RawStorage<T, R, C>> {
    mat: &'a Matrix<T, R, C, S>,
    curr: usize,
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> RowIter<'a, T, R, C, S> {
    pub(crate) const fn new(mat: &'a Matrix<T, R, C, S>) -> Self {
        RowIter { mat, curr: 0 }
    }
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> Iterator for RowIter<'a, T, R, C, S> {
    type Item = MatrixView<'a, T, U1, C, S::RStride, S::CStride>;

    /// Advances the iterator and returns the next row.
    ///
    /// Returns `None` when all rows have been iterated.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let matrix = Matrix3x2::from_row_slice(&[
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.row_iter();
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 1);
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 3);
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 5);
    /// assert!(iter.next().is_none());
    /// ```
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.mat.nrows() {
            let res = self.mat.row(self.curr);
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Returns a tuple where both elements are equal to the number of remaining rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter();
    ///
    /// assert_eq!(iter.size_hint(), (3, Some(3)));
    /// iter.next();
    /// assert_eq!(iter.size_hint(), (2, Some(2)));
    /// ```
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.mat.nrows() - self.curr,
            Some(self.mat.nrows() - self.curr),
        )
    }

    /// Consumes the iterator and returns the number of remaining rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter();
    ///
    /// iter.next();
    /// assert_eq!(iter.count(), 2);
    /// ```
    #[inline]
    fn count(self) -> usize {
        self.mat.nrows() - self.curr
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> ExactSizeIterator
    for RowIter<'a, T, R, C, S>
{
    /// Returns the exact number of remaining rows in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter();
    ///
    /// assert_eq!(iter.len(), 3);
    /// iter.next();
    /// assert_eq!(iter.len(), 2);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.mat.nrows() - self.curr
    }
}

/// An iterator through the mutable rows of a matrix.
///
/// This iterator yields mutable row views (1 x C matrices) from a matrix. Each element
/// returned is a [`MatrixViewMut`] representing one complete row of the source matrix,
/// allowing you to modify the row's elements.
///
/// # Iteration Order
///
/// Rows are yielded from top to bottom (row 0, then row 1, etc.).
///
/// # Iterator Traits
///
/// This iterator implements:
/// - [`Iterator`]: Basic forward iteration
/// - [`ExactSizeIterator`]: Provides exact count of remaining rows
///
/// # Examples
///
/// Basic iteration and modification:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let mut matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// // Double each element in every row
/// for mut row in matrix.row_iter_mut() {
///     for elem in row.iter_mut() {
///         *elem *= 2;
///     }
/// }
///
/// assert_eq!(matrix, Matrix3x2::from_row_slice(&[
///     2, 4,
///     6, 8,
///     10, 12,
/// ]));
/// ```
///
/// Modifying specific rows:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let mut matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// // Set all elements in each row to the row index
/// for (i, mut row) in matrix.row_iter_mut().enumerate() {
///     for elem in row.iter_mut() {
///         *elem = i as i32;
///     }
/// }
///
/// assert_eq!(matrix, Matrix3x2::from_row_slice(&[
///     0, 0,
///     1, 1,
///     2, 2,
/// ]));
/// ```
///
/// Normalizing rows (making each row sum to a target value):
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let mut matrix = Matrix2x3::from_row_slice(&[
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
/// ]);
///
/// // Normalize each row so its sum equals 1.0
/// for mut row in matrix.row_iter_mut() {
///     let sum: f64 = row.iter().sum();
///     for elem in row.iter_mut() {
///         *elem /= sum;
///     }
/// }
///
/// // Each row now sums to approximately 1.0
/// for row in matrix.row_iter() {
///     let sum: f64 = row.iter().sum();
///     assert!((sum - 1.0).abs() < 1e-10);
/// }
/// ```
///
/// Applying different operations to different rows:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let mut matrix = Matrix3x2::from_row_slice(&[
///     1, 2,
///     3, 4,
///     5, 6,
/// ]);
///
/// let mut iter = matrix.row_iter_mut();
///
/// // Add 10 to first row
/// if let Some(mut row) = iter.next() {
///     for elem in row.iter_mut() {
///         *elem += 10;
///     }
/// }
///
/// // Multiply second row by 2
/// if let Some(mut row) = iter.next() {
///     for elem in row.iter_mut() {
///         *elem *= 2;
///     }
/// }
///
/// assert_eq!(matrix, Matrix3x2::from_row_slice(&[
///     11, 12,
///     6, 8,
///     5, 6,
/// ]));
/// ```
///
/// # See Also
///
/// - [`Matrix::row_iter_mut`](crate::Matrix::row_iter_mut): Creates this iterator
/// - [`RowIter`]: Immutable version of this iterator
/// - [`ColumnIterMut`]: Mutable iterator over columns
/// - [`Matrix::row_mut`](crate::Matrix::row_mut): Access a single mutable row
#[derive(Debug)]
pub struct RowIterMut<'a, T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> {
    mat: *mut Matrix<T, R, C, S>,
    curr: usize,
    phantom: PhantomData<&'a mut Matrix<T, R, C, S>>,
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> RowIterMut<'a, T, R, C, S> {
    pub(crate) const fn new(mat: &'a mut Matrix<T, R, C, S>) -> Self {
        RowIterMut {
            mat,
            curr: 0,
            phantom: PhantomData,
        }
    }

    fn nrows(&self) -> usize {
        unsafe { (*self.mat).nrows() }
    }
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> Iterator
    for RowIterMut<'a, T, R, C, S>
{
    type Item = MatrixViewMut<'a, T, U1, C, S::RStride, S::CStride>;

    /// Advances the iterator and returns the next mutable row.
    ///
    /// Returns `None` when all rows have been iterated.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let mut matrix = Matrix3x2::from_row_slice(&[
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.row_iter_mut();
    ///
    /// // Modify first row
    /// if let Some(mut row) = iter.next() {
    ///     row[(0, 0)] = 10;
    /// }
    ///
    /// assert_eq!(matrix[(0, 0)], 10);
    /// ```
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.nrows() {
            let res = unsafe { (*self.mat).row_mut(self.curr) };
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Returns a tuple where both elements are equal to the number of remaining rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let mut matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter_mut();
    ///
    /// assert_eq!(iter.size_hint(), (3, Some(3)));
    /// iter.next();
    /// assert_eq!(iter.size_hint(), (2, Some(2)));
    /// ```
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nrows() - self.curr, Some(self.nrows() - self.curr))
    }

    /// Consumes the iterator and returns the number of remaining rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let mut matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter_mut();
    ///
    /// iter.next();
    /// assert_eq!(iter.count(), 2);
    /// ```
    #[inline]
    fn count(self) -> usize {
        self.nrows() - self.curr
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> ExactSizeIterator
    for RowIterMut<'a, T, R, C, S>
{
    /// Returns the exact number of remaining rows in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let mut matrix = Matrix3x2::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.row_iter_mut();
    ///
    /// assert_eq!(iter.len(), 3);
    /// iter.next();
    /// assert_eq!(iter.len(), 2);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.nrows() - self.curr
    }
}

/*
 * Column iterators.
 *
 */
#[derive(Clone, Debug)]
/// An iterator through the columns of a matrix.
///
/// This iterator yields column views (R x 1 matrices) from a matrix. Each element
/// returned is a [`MatrixView`] representing one complete column of the source matrix.
///
/// # Iteration Order
///
/// Columns are yielded from left to right (column 0, then column 1, etc.).
///
/// # Iterator Traits
///
/// This iterator implements:
/// - [`Iterator`]: Basic forward iteration
/// - [`DoubleEndedIterator`]: Allows iterating from both ends
/// - [`ExactSizeIterator`]: Provides exact count of remaining columns
///
/// # Examples
///
/// Basic iteration over columns:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// let mut col_iter = matrix.column_iter();
///
/// // First column
/// let col0 = col_iter.next().unwrap();
/// assert_eq!(col0[(0, 0)], 1);
/// assert_eq!(col0[(1, 0)], 4);
///
/// // Second column
/// let col1 = col_iter.next().unwrap();
/// assert_eq!(col1[(0, 0)], 2);
/// assert_eq!(col1[(1, 0)], 5);
///
/// // Third column
/// let col2 = col_iter.next().unwrap();
/// assert_eq!(col2[(0, 0)], 3);
/// assert_eq!(col2[(1, 0)], 6);
///
/// assert!(col_iter.next().is_none());
/// ```
///
/// Computing column sums:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// let col_sums: Vec<i32> = matrix.column_iter()
///     .map(|col| col.iter().sum())
///     .collect();
///
/// assert_eq!(col_sums, vec![5, 7, 9]);
/// ```
///
/// Finding the maximum value in each column:
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let matrix = Matrix3x2::from_row_slice(&[
///     1, 5,
///     2, 3,
///     4, 6,
/// ]);
///
/// let max_per_column: Vec<i32> = matrix.column_iter()
///     .map(|col| *col.iter().max().unwrap())
///     .collect();
///
/// assert_eq!(max_per_column, vec![4, 6]);
/// ```
///
/// Iterating from both ends:
///
/// ```
/// use nalgebra::Matrix2x4;
///
/// let matrix = Matrix2x4::from_row_slice(&[
///     1, 2, 3, 4,
///     5, 6, 7, 8,
/// ]);
///
/// let mut iter = matrix.column_iter();
///
/// // Get first and last columns
/// let first = iter.next().unwrap();
/// assert_eq!(first[(0, 0)], 1);
///
/// let last = iter.next_back().unwrap();
/// assert_eq!(last[(0, 0)], 4);
///
/// // Get remaining middle columns
/// let second = iter.next().unwrap();
/// assert_eq!(second[(0, 0)], 2);
///
/// let third = iter.next().unwrap();
/// assert_eq!(third[(0, 0)], 3);
///
/// assert!(iter.next().is_none());
/// ```
///
/// Collecting columns into a vector:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// let columns: Vec<_> = matrix.column_iter().collect();
/// assert_eq!(columns.len(), 3);
/// assert_eq!(columns[0][(0, 0)], 1);
/// assert_eq!(columns[1][(0, 0)], 2);
/// assert_eq!(columns[2][(0, 0)], 3);
/// ```
///
/// # See Also
///
/// - [`Matrix::column_iter`](crate::Matrix::column_iter): Creates this iterator
/// - [`ColumnIterMut`]: Mutable version of this iterator
/// - [`RowIter`]: Iterator over rows instead of columns
/// - [`Matrix::column`](crate::Matrix::column): Access a single column
pub struct ColumnIter<'a, T, R: Dim, C: Dim, S: RawStorage<T, R, C>> {
    mat: &'a Matrix<T, R, C, S>,
    range: Range<usize>,
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> ColumnIter<'a, T, R, C, S> {
    /// a new column iterator covering all columns of the matrix
    pub(crate) fn new(mat: &'a Matrix<T, R, C, S>) -> Self {
        ColumnIter {
            mat,
            range: 0..mat.ncols(),
        }
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
        // SAFETY: this makes sure the generated ranges are valid.
        let split_pos = (self.range.start + index).min(self.range.end);

        let left_iter = ColumnIter {
            mat: self.mat,
            range: self.range.start..split_pos,
        };

        let right_iter = ColumnIter {
            mat: self.mat,
            range: split_pos..self.range.end,
        };

        (left_iter, right_iter)
    }
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> Iterator for ColumnIter<'a, T, R, C, S> {
    type Item = MatrixView<'a, T, R, U1, S::RStride, S::CStride>;

    /// Advances the iterator and returns the next column.
    ///
    /// Returns `None` when all columns have been iterated.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.column_iter();
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 1);
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 2);
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 3);
    /// assert!(iter.next().is_none());
    /// ```
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.range.start <= self.range.end);
        if self.range.start < self.range.end {
            let res = self.mat.column(self.range.start);
            self.range.start += 1;
            Some(res)
        } else {
            None
        }
    }

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Returns a tuple where both elements are equal to the number of remaining columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter();
    ///
    /// assert_eq!(iter.size_hint(), (3, Some(3)));
    /// iter.next();
    /// assert_eq!(iter.size_hint(), (2, Some(2)));
    /// ```
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.range.len();
        (hint, Some(hint))
    }

    /// Consumes the iterator and returns the number of remaining columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter();
    ///
    /// iter.next();
    /// assert_eq!(iter.count(), 2);
    /// ```
    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> DoubleEndedIterator
    for ColumnIter<'a, T, R, C, S>
{
    /// Removes and returns a column from the end of the iterator.
    ///
    /// Returns `None` when there are no more columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.column_iter();
    /// assert_eq!(iter.next_back().unwrap()[(0, 0)], 3);  // last column
    /// assert_eq!(iter.next_back().unwrap()[(0, 0)], 2);  // second-to-last
    /// assert_eq!(iter.next().unwrap()[(0, 0)], 1);       // first column
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.range.start <= self.range.end);
        if !self.range.is_empty() {
            self.range.end -= 1;
            debug_assert!(self.range.end < self.mat.ncols());
            debug_assert!(self.range.end >= self.range.start);
            Some(self.mat.column(self.range.end))
        } else {
            None
        }
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorage<T, R, C>> ExactSizeIterator
    for ColumnIter<'a, T, R, C, S>
{
    /// Returns the exact number of remaining columns in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter();
    ///
    /// assert_eq!(iter.len(), 3);
    /// iter.next();
    /// assert_eq!(iter.len(), 2);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.range.end - self.range.start
    }
}

/// An iterator through the mutable columns of a matrix.
///
/// This iterator yields mutable column views (R x 1 matrices) from a matrix. Each element
/// returned is a [`MatrixViewMut`] representing one complete column of the source matrix,
/// allowing you to modify the column's elements.
///
/// # Iteration Order
///
/// Columns are yielded from left to right (column 0, then column 1, etc.).
///
/// # Iterator Traits
///
/// This iterator implements:
/// - [`Iterator`]: Basic forward iteration
/// - [`DoubleEndedIterator`]: Allows iterating from both ends
/// - [`ExactSizeIterator`]: Provides exact count of remaining columns
///
/// # Examples
///
/// Basic iteration and modification:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let mut matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// // Double each element in every column
/// for mut col in matrix.column_iter_mut() {
///     for elem in col.iter_mut() {
///         *elem *= 2;
///     }
/// }
///
/// assert_eq!(matrix, Matrix2x3::from_row_slice(&[
///     2, 4, 6,
///     8, 10, 12,
/// ]));
/// ```
///
/// Setting columns to specific values:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let mut matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// // Set all elements in each column to the column index
/// for (i, mut col) in matrix.column_iter_mut().enumerate() {
///     for elem in col.iter_mut() {
///         *elem = i as i32;
///     }
/// }
///
/// assert_eq!(matrix, Matrix2x3::from_row_slice(&[
///     0, 1, 2,
///     0, 1, 2,
/// ]));
/// ```
///
/// Normalizing columns (making each column sum to a target value):
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let mut matrix = Matrix3x2::from_row_slice(&[
///     1.0, 4.0,
///     2.0, 5.0,
///     3.0, 6.0,
/// ]);
///
/// // Normalize each column so its sum equals 1.0
/// for mut col in matrix.column_iter_mut() {
///     let sum: f64 = col.iter().sum();
///     for elem in col.iter_mut() {
///         *elem /= sum;
///     }
/// }
///
/// // Each column now sums to approximately 1.0
/// for col in matrix.column_iter() {
///     let sum: f64 = col.iter().sum();
///     assert!((sum - 1.0).abs() < 1e-10);
/// }
/// ```
///
/// Applying different operations to different columns:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let mut matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// let mut iter = matrix.column_iter_mut();
///
/// // Add 10 to first column
/// if let Some(mut col) = iter.next() {
///     for elem in col.iter_mut() {
///         *elem += 10;
///     }
/// }
///
/// // Multiply third column (from the end) by 2
/// if let Some(mut col) = iter.next_back() {
///     for elem in col.iter_mut() {
///         *elem *= 2;
///     }
/// }
///
/// assert_eq!(matrix, Matrix2x3::from_row_slice(&[
///     11, 2, 6,
///     14, 5, 12,
/// ]));
/// ```
///
/// Swapping values between columns:
///
/// ```
/// use nalgebra::Matrix2x3;
///
/// let mut matrix = Matrix2x3::from_row_slice(&[
///     1, 2, 3,
///     4, 5, 6,
/// ]);
///
/// // Collect columns into a vector for manipulation
/// let mut cols: Vec<_> = matrix.column_iter_mut().collect();
///
/// // Swap elements between first and last column
/// for i in 0..cols[0].len() {
///     let temp = cols[0][i];
///     cols[0][i] = cols[2][i];
///     cols[2][i] = temp;
/// }
///
/// assert_eq!(matrix, Matrix2x3::from_row_slice(&[
///     3, 2, 1,
///     6, 5, 4,
/// ]));
/// ```
///
/// # See Also
///
/// - [`Matrix::column_iter_mut`](crate::Matrix::column_iter_mut): Creates this iterator
/// - [`ColumnIter`]: Immutable version of this iterator
/// - [`RowIterMut`]: Mutable iterator over rows
/// - [`Matrix::column_mut`](crate::Matrix::column_mut): Access a single mutable column
#[derive(Debug)]
pub struct ColumnIterMut<'a, T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> {
    mat: *mut Matrix<T, R, C, S>,
    range: Range<usize>,
    phantom: PhantomData<&'a mut Matrix<T, R, C, S>>,
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> ColumnIterMut<'a, T, R, C, S> {
    pub(crate) fn new(mat: &'a mut Matrix<T, R, C, S>) -> Self {
        let range = 0..mat.ncols();
        ColumnIterMut {
            mat,
            range,
            phantom: Default::default(),
        }
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
        // SAFETY: this makes sure the generated ranges are valid.
        let split_pos = (self.range.start + index).min(self.range.end);

        let left_iter = ColumnIterMut {
            mat: self.mat,
            range: self.range.start..split_pos,
            phantom: Default::default(),
        };

        let right_iter = ColumnIterMut {
            mat: self.mat,
            range: split_pos..self.range.end,
            phantom: Default::default(),
        };

        (left_iter, right_iter)
    }

    fn ncols(&self) -> usize {
        unsafe { (*self.mat).ncols() }
    }
}

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> Iterator
    for ColumnIterMut<'a, T, R, C, S>
{
    type Item = MatrixViewMut<'a, T, R, U1, S::RStride, S::CStride>;

    /// Advances the iterator and returns the next mutable column.
    ///
    /// Returns `None` when all columns have been iterated.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.column_iter_mut();
    ///
    /// // Modify first column
    /// if let Some(mut col) = iter.next() {
    ///     col[(0, 0)] = 10;
    /// }
    ///
    /// assert_eq!(matrix[(0, 0)], 10);
    /// ```
    #[inline]
    fn next(&'_ mut self) -> Option<Self::Item> {
        debug_assert!(self.range.start <= self.range.end);
        if self.range.start < self.range.end {
            let res = unsafe { (*self.mat).column_mut(self.range.start) };
            self.range.start += 1;
            Some(res)
        } else {
            None
        }
    }

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Returns a tuple where both elements are equal to the number of remaining columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter_mut();
    ///
    /// assert_eq!(iter.size_hint(), (3, Some(3)));
    /// iter.next();
    /// assert_eq!(iter.size_hint(), (2, Some(2)));
    /// ```
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.range.len();
        (hint, Some(hint))
    }

    /// Consumes the iterator and returns the number of remaining columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter_mut();
    ///
    /// iter.next();
    /// assert_eq!(iter.count(), 2);
    /// ```
    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> ExactSizeIterator
    for ColumnIterMut<'a, T, R, C, S>
{
    /// Returns the exact number of remaining columns in the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut iter = matrix.column_iter_mut();
    ///
    /// assert_eq!(iter.len(), 3);
    /// iter.next();
    /// assert_eq!(iter.len(), 2);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.range.len()
    }
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> DoubleEndedIterator
    for ColumnIterMut<'a, T, R, C, S>
{
    /// Removes and returns a mutable column from the end of the iterator.
    ///
    /// Returns `None` when there are no more columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let mut matrix = Matrix2x3::from_row_slice(&[
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    ///
    /// let mut iter = matrix.column_iter_mut();
    ///
    /// // Modify last column
    /// if let Some(mut col) = iter.next_back() {
    ///     col[(0, 0)] = 30;
    /// }
    ///
    /// assert_eq!(matrix[(0, 2)], 30);
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.range.start <= self.range.end);
        if !self.range.is_empty() {
            self.range.end -= 1;
            debug_assert!(self.range.end < self.ncols());
            debug_assert!(self.range.end >= self.range.start);
            Some(unsafe { (*self.mat).column_mut(self.range.end) })
        } else {
            None
        }
    }
}
