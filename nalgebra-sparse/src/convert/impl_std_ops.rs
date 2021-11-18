use crate::{
    convert::serial::*,
    coo::CooMatrix,
    cs::{CscMatrix, CsrMatrix},
};
use nalgebra::{storage::RawStorage, ClosedAdd, DMatrix, Dim, Matrix, Scalar};
use num_traits::{Unsigned, Zero};
use std::ops::Add;

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

impl<'a, T> From<&'a CooMatrix<T>> for CsrMatrix<T, usize>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csr(matrix)
    }
}

impl<'a, T, O, I> From<&'a CsrMatrix<T, O, I>> for CooMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CsrMatrix<T, O, I>) -> Self {
        convert_csr_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CsrMatrix<T, usize>
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

impl<'a, T, O, I> From<&'a CsrMatrix<T, O, I>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CsrMatrix<T, O, I>) -> Self {
        convert_csr_dense(matrix)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for CscMatrix<T, usize>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csc(matrix)
    }
}

impl<'a, T, O, I> From<&'a CscMatrix<T, O, I>> for CooMatrix<T>
where
    T: Scalar + Zero,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CscMatrix<T, O, I>) -> Self {
        convert_csc_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CscMatrix<T, usize>
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

impl<'a, T, O, I> From<&'a CscMatrix<T, O, I>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CscMatrix<T, O, I>) -> Self {
        convert_csc_dense(matrix)
    }
}

impl<'a, T, O, I> From<&'a CscMatrix<T, O, I>> for CsrMatrix<T, usize>
where
    T: Scalar,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CscMatrix<T, O, I>) -> Self {
        convert_csc_csr(matrix)
    }
}

impl<'a, T, O, I> From<&'a CsrMatrix<T, O, I>> for CscMatrix<T, usize>
where
    T: Scalar,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    fn from(matrix: &'a CsrMatrix<T, O, I>) -> Self {
        convert_csr_csc(matrix)
    }
}
