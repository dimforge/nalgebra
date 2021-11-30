use crate::{
    convert::serial::*,
    coo::CooMatrix,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
};
use nalgebra::{storage::RawStorage, ClosedAdd, DMatrix, Dim, Matrix, Scalar};
use num_traits::Zero;
use std::borrow::Borrow;

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_coo(matrix)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(coo: &'a CooMatrix<T>) -> Self {
        convert_coo_dense(coo)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csr(matrix)
    }
}

impl<'a, T> From<&'a CsrMatrix<T>> for CooMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        convert_csr_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CsrMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_csr(matrix)
    }
}

impl<'a, T> From<&'a CsrMatrix<T>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        convert_csr_dense(matrix)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for CscMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csc(matrix)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for CooMatrix<T>
where
    T: Scalar + Zero,
{
    fn from(matrix: &'a CscMatrix<T>) -> Self {
        convert_csc_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CscMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_csc(matrix)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CscMatrix<T>) -> Self {
        convert_csc_dense(matrix)
    }
}

impl<T, MajorOffsets, MinorIndices, Data>
    From<CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedColumnStorage>> for CsrMatrix<T>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn from(
        matrix: CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedColumnStorage>,
    ) -> Self {
        convert_csc_csr(&matrix)
    }
}

impl<T, MajorOffsets, MinorIndices, Data>
    From<CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedRowStorage>> for CscMatrix<T>
where
    T: Scalar,
    MajorOffsets: Borrow<[usize]>,
    MinorIndices: Borrow<[usize]>,
    Data: Borrow<[T]>,
{
    fn from(matrix: CsMatrix<T, MajorOffsets, MinorIndices, Data, CompressedRowStorage>) -> Self {
        convert_csr_csc(&matrix)
    }
}
