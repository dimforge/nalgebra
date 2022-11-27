//! Parallel iterators for matrices compatible with rayon.

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

use crate::{
    iter::{ColumnIter, ColumnIterMut},
    Dim, Matrix, MatrixSlice, MatrixSliceMut, RawStorage, RawStorageMut, Scalar, U1,
};
use rayon::{iter::plumbing::bridge, prelude::*};
use rayon::iter::plumbing::Producer;

/// A rayon parallel iterator over the colums of a matrix. It is created
/// using the [`par_column_iter`] method of [`Matrix`].
///
/// *only availabe if compiled with the feature `par-iter`*
/// [`par_column_iter`]: crate::Matrix::par_column_iter
/// [`Matrix`]: crate::Matrix
#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
pub struct ParColumnIter<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> {
    mat: &'a Matrix<T, R, Cols, S>,
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> ParColumnIter<'a, T, R, Cols, S> {
    /// create a new parallel iterator for the given matrix
    fn new(matrix: &'a Matrix<T, R, Cols, S>) -> Self {
        Self { mat: matrix }
    }
}

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// *only availabe if compiled with the feature `par-iter`*
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
        let producer = ColumnProducer(ColumnIter::new(self.mat));
        callback.callback(producer)
    }
}

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// A rayon parallel iterator through the mutable columns of a matrix
/// *only availabe if compiled with the feature `par-iter`*
pub struct ParColumnIterMut<
    'a,
    T,
    R: Dim,
    Cols: Dim,
    S: RawStorage<T, R, Cols> + RawStorageMut<T, R, Cols>,
> {
    mat: &'a mut Matrix<T, R, Cols, S>,
}

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// *only availabe if compiled with the feature `par-iter`*
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// *only availabe if compiled with the feature `par-iter`*
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// *only availabe if compiled with the feature `par-iter`*
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
        let producer = ColumnProducerMut(ColumnIterMut::new(self.mat));
        callback.callback(producer)
    }
}

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// # Parallel iterators using `rayon`
/// *Only availabe if compiled with the feature `par-iter`*
impl<T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> Matrix<T, R, Cols, S>
where
    T: Send + Sync + Scalar,
    S: Sync,
{
    /// Iterate through the columns of the matrix in parallel using rayon.
    /// This iterates over *immutable* references ot the columns of the matrix,
    /// if *mutable* access to the columns is required, use [`par_column_iter_mut`]
    /// instead.
    ///
    /// # Example
    /// Using parallel column iterators to calculate the sum of the maximum
    /// elements in each column:
    /// ```
    /// use nalgebra::{dmatrix,DMatrix};
    /// use rayon::prelude::*;
    ///
    /// let matrix : DMatrix<f64> =
    ///         nalgebra::dmatrix![1.,0.,5.;
    ///                            2.,4.,1.;
    ///                            3.,2.,2.;];
    /// let sum_of_max :f64 =
    ///         matrix
    ///         .par_column_iter()
    ///         .map(|col|col.max())
    ///         .sum();
    ///
    /// assert_eq!(sum_of_max,3.+4.+5.);
    ///                             
    /// ```
    ///
    /// [`par_column_iter_mut`]: crate::Matrix::par_column_iter_mut
    pub fn par_column_iter(&self) -> ParColumnIter<'_, T, R, Cols, S> {
        ParColumnIter::new(self)
    }
    /// Mutably iterate through the columns of this matrix in parallel using rayon.
    /// Allows mutable access to the columns in parallel using mutable references.
    /// If mutable access to the columns is not required rather use [`par_column_iter`]
    /// instead.
    ///
    /// # Example
    /// Normalize each column of a matrix with respect to its own maximum value.
    ///
    /// ```
    /// use nalgebra::{dmatrix,DMatrix};
    /// use rayon::prelude::*;
    ///
    /// let mut matrix : DMatrix<f64> =
    ///                     dmatrix![2.,4.,6.;
    ///                             1.,2.,3.];
    /// matrix.par_column_iter_mut().for_each(|mut col| col /= col.max());
    ///
    /// assert_eq!(matrix,
    ///             dmatrix![1. ,1. , 1.;
    ///                      0.5,0.5,0.5]);
    /// ```
    ///
    /// [`par_column_iter`]: crate::Matrix::par_column_iter
    pub fn par_column_iter_mut(&mut self) -> ParColumnIterMut<'_, T, R, Cols, S>
    where
        S: RawStorageMut<T, R, Cols>,
    {
        ParColumnIterMut::new(self)
    }
}

/// a private helper newtype that wraps the `ColumnIter` and implements
/// the rayon `Producer` trait. It's just here so we don't have to make the
/// rayon trait part of the public interface of the `ColumnIter`
struct ColumnProducer<'a,T,R:Dim,C:Dim,S:RawStorage<T,R,C>>(ColumnIter<'a,T,R,C,S>); 

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// *only available if compiled with the feature `par-iter`*
impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> Producer for ColumnProducer<'a, T, R, Cols, S>
where
T: Send + Sync + Scalar,
S: Sync,
{
    type Item = MatrixSlice<'a, T, R, U1, S::RStride, S::CStride>;
    type IntoIter = ColumnIter<'a, T, R, Cols, S>;
    
    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        // the index is relative to the size of this current iterator
        // it will always start at zero so it serves as an offset
        let left_iter = ColumnIter {
            mat: self.0.mat,
            range: self.0.range.start..(self.0.range.start + index),
        };

        let right_iter = ColumnIter {
            mat: self.0.mat,
            range: (self.0.range.start + index)..self.0.range.end,
        };
        (Self(left_iter), Self(right_iter))
    }

    #[inline]   
    fn into_iter(self) -> Self::IntoIter {
        self.0
    }
}

/// See `ColumnProducer`. A private wrapper newtype that keeps the Producer
/// implementation private
struct ColumnProducerMut<'a, T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>>(ColumnIterMut<'a,T,R,C,S>);

impl<'a, T, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> Producer
for ColumnProducerMut<'a, T, R, C, S>
where
T: Send + Sync + Scalar,
S: Send + Sync,
{
    type Item = MatrixSliceMut<'a, T, R, U1, S::RStride, S::CStride>;
    type IntoIter = ColumnIterMut<'a, T, R, C, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // the index is relative to the size of this current iterator
        // it will always start at zero so it serves as an offset

        let left_iter = ColumnIterMut {
            mat: self.0.mat,
            range: self.0.range.start..(self.0.range.start + index),
            phantom: Default::default(),
        };

        let right_iter = ColumnIterMut {
            mat: self.0.mat,
            range: (self.0.range.start + index)..self.0.range.end,
            phantom: Default::default(),
        };
        (Self(left_iter), Self(right_iter))
    }
}


/// this implementation is safe because we are enforcing exclusive access
/// to the columns through the active range of the iterator
unsafe impl<'a, T: Scalar, R: Dim, C: Dim, S: 'a + RawStorageMut<T, R, C>> Send
for ColumnIterMut<'a, T, R, C, S>
{
}
