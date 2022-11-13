//! Parallel iterators for matrices compatible with rayon.

//only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

use crate::{
    iter::{ColumnIter, ColumnIterMut},
    Dim, Matrix, MatrixSlice, MatrixSliceMut, RawStorage, RawStorageMut, Scalar, U1,
};
use rayon::{iter::plumbing::bridge, prelude::*};

/// A rayon parallel iterator over the colums of a matrix. It is created
/// using the [`par_column_iter`] method of [`Matrix`].
///
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
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

#[cfg_attr(doc_cfg, doc(cfg(feature = "par-iter")))]
/// # Parallel iterators using `rayon`
/// *Only availabe if compiled with the feature `par-iter`*
impl<T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>>
    Matrix<T, R, Cols, S>
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
    where S: RawStorageMut<T, R, Cols>{
        ParColumnIterMut::new(self)
    }
}
