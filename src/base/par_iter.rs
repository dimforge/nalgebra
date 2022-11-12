//! Parallel iterators for matrices compatible with rayon.

use crate::{
    iter::{ColumnIter, ColumnIterMut},
    Dim, Matrix, MatrixSlice, MatrixSliceMut, RawStorage, RawStorageMut, Scalar, U1,
};
use rayon::{iter::plumbing::bridge, prelude::*};

/// A rayon parallel iterator over the colums of a matrix
pub struct ParColumnIter<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> {
    mat: &'a Matrix<T, R, Cols, S>,
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> ParColumnIter<'a, T, R, Cols, S> {
    /// create a new parallel iterator for the given matrix
    fn new(matrix: &'a Matrix<T, R, Cols, S>) -> Self {
        Self { mat: matrix }
    }
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> ParallelIterator
    for ParColumnIter<'a, T, R, Cols, S>
where
    T: Sync + Send + Scalar,
    S: Sync,
{
    type Item = MatrixSlice<'a, T, R, U1, S::RStride, S::CStride>;

    fn drive_unindexed<Consumer>(self, consumer: Consumer) -> Consumer::Result
    where
        Consumer: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.mat.ncols())
    }
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> IndexedParallelIterator
    for ParColumnIter<'a, T, R, Cols, S>
where
    T: Send + Sync + Scalar,
    S: Sync,
{
    fn len(&self) -> usize {
        self.mat.ncols()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        let producer = ColumnIter::new(self.mat);
        callback.callback(producer)
    }
}

impl<T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> Matrix<T, R, Cols, S>
where
    T: Send + Sync + Scalar,
    S: Sync,
{
    /// Iterate through the columns of the matrix in parallel using rayon.
    pub fn par_column_iter(&self) -> ParColumnIter<'_, T, R, Cols, S> {
        ParColumnIter::new(self)
    }
}

/// A rayon parallel iterator through the mutable columns of a matrix
pub struct ParColumnIterMut<
    'a,
    T,
    R: Dim,
    Cols: Dim,
    S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>,
> {
    mat: &'a mut Matrix<T, R, Cols, S>,
}

impl<'a, T, R, Cols, S> ParColumnIterMut<'a, T, R, Cols, S>
where
    R: Dim,
    Cols: Dim,
    S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>,
{
    /// create a new parallel iterator for the given matrix
    fn new(mat: &'a mut Matrix<T, R, Cols, S>) -> Self {
        Self { mat }
    }
}

impl<'a, T, R, Cols, S> ParallelIterator for ParColumnIterMut<'a, T, R, Cols, S>
where
    R: Dim,
    Cols: Dim,
    S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>,
    T: Send + Sync + Scalar,
    S: Send + Sync,
{
    type Item = MatrixSliceMut<'a, T, R, U1, S::RStride, S::CStride>;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.mat.ncols())
    }
}

impl<'a, T, R, Cols, S> IndexedParallelIterator for ParColumnIterMut<'a, T, R, Cols, S>
where
    R: Dim,
    Cols: Dim,
    S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>,
    T: Send + Sync + Scalar,
    S: Send + Sync,
{
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.mat.ncols()
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        let producer = ColumnIterMut::new(self.mat);
        callback.callback(producer)
    }
}

impl<T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>>
    Matrix<T, R, Cols, S>
where
    T: Send + Sync + Scalar,
    S: Sync,
{
    /// Mutably iterate through the columns of this matrix in parallel using rayon
    pub fn par_column_iter_mut(&mut self) -> ParColumnIterMut<'_, T, R, Cols, S> {
        ParColumnIterMut::new(self)
    }
}
