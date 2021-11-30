//! Implements core traits for use with `matrixcompare`.
use crate::{
    coo::CooMatrix,
    cs::{Compression, CsMatrix},
};
use matrixcompare_core::{self, Access, SparseAccess};
use std::borrow::Borrow;

impl<T, MajorOffsets, MinorIndices, Data, CompressionKind> SparseAccess<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressionKind>
where
    T: Clone,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
    CompressionKind: Compression,
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

impl<T, MajorOffsets, MinorIndices, Data, CompressionKind> matrixcompare_core::Matrix<T>
    for CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressionKind>
where
    T: Clone,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
    CompressionKind: Compression,
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
