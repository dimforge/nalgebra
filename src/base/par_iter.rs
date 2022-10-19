//! this module implements parallelators to make matrices work with
//! the rayon crate seamlessly

use core::{
    fmt::Debug,
    iter::{Skip, Take},
    marker::PhantomData,
    ops::Range,
};
use std::os::unix::prelude::AsRawFd;

use rayon::{
    iter::plumbing::{bridge, Producer},
    prelude::*,
};

use crate::{
    iter::{ColumnIter, ColumnIterMut}, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, MatrixSliceMut,
    RawStorage, RawStorageMut, U1, SliceStorageMut,
};

use super::conversion;

/// a rayon parallel iterator over the columns of a matrix
pub struct ParColumnIter<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> {
    mat: &'a Matrix<T, R, Cols, S>,
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> ParColumnIter<'a, T, R, Cols, S> {
    fn new(matrix: &'a Matrix<T, R, Cols, S>) -> Self {
        Self { mat: matrix }
    }
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> ParallelIterator
    for ParColumnIter<'a, T, R, Cols, S>
where
    T: Sync + Send + Clone + Debug + PartialEq + 'static,
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
    T: Send + Sync + Clone + Debug + PartialEq + 'static,
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

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols>> Matrix<T, R, Cols, S>
where
    T: Send + Sync + Clone + Debug + PartialEq + 'static,
    S: Sync,
{
    /// TODO
    pub fn par_column_iter(&self) -> ParColumnIter<'_, T, R, Cols, S> {
        ParColumnIter::new(self)
    }
}

/// TODO
pub struct ParColumnIterMut<'a,T,R:Dim ,Cols:Dim, S:RawStorage<T,R,Cols>+RawStorageMut<T,R,Cols>> {
    mat : &'a mut Matrix<T,R,Cols,S>,
}

impl<'a,T,R,Cols,S> ParColumnIterMut<'a,T,R,Cols,S> 
where R: Dim, Cols : Dim, S:RawStorage<T,R,Cols> + RawStorageMut<T,R,Cols> {
    /// TODO
    pub fn new(mat : &'a mut Matrix<T,R,Cols,S>) -> Self {
        Self {
            mat,
        }
    }
}

impl<'a,T,R,Cols,S> ParallelIterator for ParColumnIterMut<'a,T,R,Cols,S> 
where R: Dim, Cols : Dim, S:RawStorage<T,R,Cols> + RawStorageMut<T,R,Cols>, 
T : Send + Sync + Debug + PartialEq + Clone + 'static, 
S : Send + Sync {
    type Item = MatrixSliceMut<'a, T, R, U1, S::RStride, S::CStride>;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
                bridge(self,consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.mat.ncols())
    }
}


impl<'a,T,R,Cols,S> IndexedParallelIterator for ParColumnIterMut<'a,T,R,Cols,S> 
where R: Dim, Cols : Dim, S:RawStorage<T,R,Cols> + RawStorageMut<T,R,Cols>, 
T : Send + Sync + Debug + PartialEq + Clone + 'static, 
S : Send + Sync {
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self,consumer)
    }

    fn len(&self) -> usize {
        self.mat.ncols()
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = ColumnIterMut::new(self.mat);
        callback.callback(producer)
    }
}

impl<'a, T, R: Dim, Cols: Dim, S: RawStorage<T, R, Cols> + RawStorageMut<T,R,Cols>> Matrix<T, R, Cols, S>
where
    T: Send + Sync + Clone + Debug + PartialEq + 'static,
    S: Sync,
{
    /// TODO
    pub fn par_column_iter_mut(&mut self) -> ParColumnIterMut<'_, T, R, Cols, S> {
        ParColumnIterMut::new(self)
    }
}



#[test]
fn test_mut_parallel_iter() {
    let mut matrix = DMatrix::<f32>::zeros(4, 3);
    matrix.par_column_iter_mut().enumerate().for_each(|(idx,mut col)| col[idx]=1f32);
    let identity = DMatrix::<f32>::identity(4, 3);
    assert_eq!(matrix,identity);
}


fn try_some_stuff() {
    let mut mat = DMatrix::<f32>::zeros(3, 4);
    let _left = mat.columns_mut(0, 1);
    let _right = mat.columns_mut(1, 3);
}
