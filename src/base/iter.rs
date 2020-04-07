//! Matrix iterators.

use std::marker::PhantomData;
use std::mem;

use crate::base::dimension::{Dim, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{Matrix, MatrixSlice, MatrixSliceMut, Scalar};

macro_rules! iterator {
    (struct $Name:ident for $Storage:ident.$ptr: ident -> $Ptr:ty, $Ref:ty, $SRef: ty) => {
        /// An iterator through a dense matrix with arbitrary strides matrix.
        pub struct $Name<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> {
            ptr: $Ptr,
            inner_ptr: $Ptr,
            inner_end: $Ptr,
            size: usize, // We can't use an end pointer here because a stride might be zero.
            strides: (S::RStride, S::CStride),
            _phantoms: PhantomData<($Ref, R, C, S)>,
        }

        // FIXME: we need to specialize for the case where the matrix storage is owned (in which
        // case the iterator is trivial because it does not have any stride).
        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> $Name<'a, N, R, C, S> {
            /// Creates a new iterator for the given matrix storage.
            pub fn new(storage: $SRef) -> $Name<'a, N, R, C, S> {
                let shape = storage.shape();
                let strides = storage.strides();
                let inner_offset = shape.0.value() * strides.0.value();
                let size = shape.0.value() * shape.1.value();
                let ptr = storage.$ptr();

                // If we have a size of 0, 'ptr' must be
                // dangling. Howver, 'inner_offset' might
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
                    unsafe { ptr.offset(inner_offset as isize) }
                };

                $Name {
                    ptr: ptr,
                    inner_ptr: ptr,
                    inner_end,
                    size: shape.0.value() * shape.1.value(),
                    strides: strides,
                    _phantoms: PhantomData,
                }
            }
        }

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> Iterator
            for $Name<'a, N, R, C, S>
        {
            type Item = $Ref;

            #[inline]
            fn next(&mut self) -> Option<$Ref> {
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

                        let stride = self.strides.0.value() as isize;
                        // Don't offset `self.ptr` for the last element,
                        // as this will be out of bounds. Iteration is done
                        // at this point (the next call to `next` will return `None`)
                        // so this is not observable.
                        if self.size != 0 {
                            self.ptr = self.ptr.offset(stride);
                        }
                        Some(mem::transmute(old))
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

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> ExactSizeIterator
            for $Name<'a, N, R, C, S>
        {
            #[inline]
            fn len(&self) -> usize {
                self.size
            }
        }
    };
}

iterator!(struct MatrixIter for Storage.ptr -> *const N, &'a N, &'a S);
iterator!(struct MatrixIterMut for StorageMut.ptr_mut -> *mut N, &'a mut N, &'a mut S);

/*
 *
 * Row iterators.
 *
 */
#[derive(Clone)]
/// An iterator through the rows of a matrix.
pub struct RowIter<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> {
    mat: &'a Matrix<N, R, C, S>,
    curr: usize,
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> RowIter<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a Matrix<N, R, C, S>) -> Self {
        RowIter { mat, curr: 0 }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> Iterator for RowIter<'a, N, R, C, S> {
    type Item = MatrixSlice<'a, N, U1, C, S::RStride, S::CStride>;

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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.mat.nrows() - self.curr,
            Some(self.mat.nrows() - self.curr),
        )
    }

    #[inline]
    fn count(self) -> usize {
        self.mat.nrows() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ExactSizeIterator
    for RowIter<'a, N, R, C, S>
{
    #[inline]
    fn len(&self) -> usize {
        self.mat.nrows() - self.curr
    }
}

/// An iterator through the mutable rows of a matrix.
pub struct RowIterMut<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> {
    mat: *mut Matrix<N, R, C, S>,
    curr: usize,
    phantom: PhantomData<&'a mut Matrix<N, R, C, S>>,
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> RowIterMut<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a mut Matrix<N, R, C, S>) -> Self {
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

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> Iterator
    for RowIterMut<'a, N, R, C, S>
{
    type Item = MatrixSliceMut<'a, N, U1, C, S::RStride, S::CStride>;

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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nrows() - self.curr, Some(self.nrows() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.nrows() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ExactSizeIterator
    for RowIterMut<'a, N, R, C, S>
{
    #[inline]
    fn len(&self) -> usize {
        self.nrows() - self.curr
    }
}

/*
 *
 * Column iterators.
 *
 */
#[derive(Clone)]
/// An iterator through the columns of a matrix.
pub struct ColumnIter<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> {
    mat: &'a Matrix<N, R, C, S>,
    curr: usize,
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ColumnIter<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a Matrix<N, R, C, S>) -> Self {
        ColumnIter { mat, curr: 0 }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> Iterator
    for ColumnIter<'a, N, R, C, S>
{
    type Item = MatrixSlice<'a, N, R, U1, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.mat.ncols() {
            let res = self.mat.column(self.curr);
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.mat.ncols() - self.curr,
            Some(self.mat.ncols() - self.curr),
        )
    }

    #[inline]
    fn count(self) -> usize {
        self.mat.ncols() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ExactSizeIterator
    for ColumnIter<'a, N, R, C, S>
{
    #[inline]
    fn len(&self) -> usize {
        self.mat.ncols() - self.curr
    }
}

/// An iterator through the mutable columns of a matrix.
pub struct ColumnIterMut<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> {
    mat: *mut Matrix<N, R, C, S>,
    curr: usize,
    phantom: PhantomData<&'a mut Matrix<N, R, C, S>>,
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ColumnIterMut<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a mut Matrix<N, R, C, S>) -> Self {
        ColumnIterMut {
            mat,
            curr: 0,
            phantom: PhantomData,
        }
    }

    fn ncols(&self) -> usize {
        unsafe { (*self.mat).ncols() }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> Iterator
    for ColumnIterMut<'a, N, R, C, S>
{
    type Item = MatrixSliceMut<'a, N, R, U1, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.ncols() {
            let res = unsafe { (*self.mat).column_mut(self.curr) };
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.ncols() - self.curr, Some(self.ncols() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.ncols() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ExactSizeIterator
    for ColumnIterMut<'a, N, R, C, S>
{
    #[inline]
    fn len(&self) -> usize {
        self.ncols() - self.curr
    }
}
