use crate::coo::CooMatrix;
use crate::convert::serial::{convert_dense_coo, convert_coo_dense};
use nalgebra::{Matrix, Scalar, Dim, ClosedAdd, DMatrix};
use nalgebra::storage::{Storage};
use num_traits::Zero;

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>
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