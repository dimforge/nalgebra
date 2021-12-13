//! Implements core traits for use with `matrixcompare`.
use crate::{
    coo::CooMatrix,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix},
};
use matrixcompare_core::{self, Access, SparseAccess};
use nalgebra::Scalar;
use std::borrow::Borrow;

impl<T, MajorOffsets, MinorIndices, Data> SparseAccess<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedRowStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn nnz(&self) -> usize {
        Self::nnz(self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplet_iter()
            .map(|(i, j, v)| (i, j, v.clone()))
            .collect()
    }
}

impl<T, MajorOffsets, MinorIndices, Data> SparseAccess<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedColumnStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn nnz(&self) -> usize {
        Self::nnz(self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplet_iter()
            .map(|(i, j, v)| (j, i, v.clone()))
            .collect()
    }
}

impl<T, MajorOffsets, MinorIndices, Data> matrixcompare_core::Matrix<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedRowStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> Access<'_, T> {
        Access::Sparse(self)
    }
}

impl<T, MajorOffsets, MinorIndices, Data> matrixcompare_core::Matrix<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedColumnStorage>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> Access<'_, T> {
        Access::Sparse(self)
    }
}

impl<T: Clone> SparseAccess<T> for CooMatrix<T> {
    fn nnz(&self) -> usize {
        CooMatrix::nnz(self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplet_iter()
            .map(|(i, j, v)| (i, j, v.clone()))
            .collect()
    }
}

impl<T: Clone> matrixcompare_core::Matrix<T> for CooMatrix<T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> Access<'_, T> {
        Access::Sparse(self)
    }
}
