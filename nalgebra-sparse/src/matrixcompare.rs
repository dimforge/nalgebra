//! Implements core traits for use with `matrixcompare`.
use crate::{
    coo::CooMatrix,
    cs::{Compression, CsMatrix},
};
use matrixcompare_core::{self, Access, SparseAccess};
use num_traits::Unsigned;
use std::{borrow::Borrow, ops::Add};

impl<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index> SparseAccess<T>
    for CsMatrix<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index>
where
    T: Clone,
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
    MajorOffsets: Borrow<[Offset]>,
    MinorIndices: Borrow<[Index]>,
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

impl<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index>
    matrixcompare_core::Matrix<T>
    for CsMatrix<T, Offset, MajorOffsets, MinorIndices, Data, CompressionKind, Index>
where
    T: Clone,
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
    MajorOffsets: Borrow<[Offset]>,
    MinorIndices: Borrow<[Index]>,
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
