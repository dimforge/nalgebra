use crate::{
    convert::serial::*,
    coo::CooMatrix,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
};
use nalgebra::{storage::RawStorage, ClosedAdd, DMatrix, Dim, Matrix, Scalar};
use num_traits::Zero;
use std::{borrow::Borrow, ops::Add};

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

impl<T> From<CooMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + Add<Output = T>,
{
    fn from(matrix: CooMatrix<T>) -> Self {
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

impl<'a, T, MO, MI, D> From<&'a CsMatrix<T, MO, MI, D, CompressedRowStorage>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
{
    fn from(matrix: &'a CsMatrix<T, MO, MI, D, CompressedRowStorage>) -> Self {
        convert_csr_dense(matrix)
    }
}

impl<T> From<CooMatrix<T>> for CscMatrix<T>
where
    T: Scalar + Add<Output = T>,
{
    fn from(matrix: CooMatrix<T>) -> Self {
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

impl<'a, T, MO, MI, D> From<&'a CsMatrix<T, MO, MI, D, CompressedColumnStorage>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
{
    fn from(matrix: &'a CsMatrix<T, MO, MI, D, CompressedColumnStorage>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proptest::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn dense_from_coo_from_dense_is_reflective(dense in dense_strategy()) {
            let final_dense = DMatrix::from(&CooMatrix::from(&dense));
            prop_assert_eq!(dense, final_dense);
        }

        #[test]
        fn csr_from_coo_from_csr_is_reflective(csr in csr_strategy()) {
            let final_csr = CsrMatrix::from(CooMatrix::from(&csr));

            prop_assert_eq!(csr.shape(), final_csr.shape());

            let (offsets, indices, data) = csr.cs_data();
            let (expected_offsets, expected_indices, expected_data) = final_csr.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn csr_from_dense_from_csr_is_reflective(csr in non_zero_csr_strategy()) {
            let final_csr = CsrMatrix::from(&DMatrix::from(&csr));

            prop_assert_eq!(csr.shape(), final_csr.shape());

            let (offsets, indices, data) = csr.cs_data();
            let (expected_offsets, expected_indices, expected_data) = final_csr.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn dense_from_csr_from_dense_is_reflective(dense in dense_strategy()) {
            let final_dense = DMatrix::from(&CsrMatrix::from(&dense));
            prop_assert_eq!(dense, final_dense);
        }

        #[test]
        fn csc_from_coo_from_csc_is_reflective(csc in csc_strategy()) {
            let final_csc = CscMatrix::from(CooMatrix::from(&csc));

            prop_assert_eq!(csc.shape(), final_csc.shape());

            let (offsets, indices, data) = csc.cs_data();
            let (expected_offsets, expected_indices, expected_data) = final_csc.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn csc_from_dense_from_csc_is_reflective(csc in non_zero_csc_strategy()) {
            let final_csc = CscMatrix::from(&DMatrix::from(&csc));

            prop_assert_eq!(csc.shape(), final_csc.shape());

            let (offsets, indices, data) = csc.cs_data();
            let (expected_offsets, expected_indices, expected_data) = final_csc.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn dense_from_csc_from_dense_is_reflective(dense in dense_strategy()) {
            let final_dense = DMatrix::from(&CscMatrix::from(&dense));
            prop_assert_eq!(dense, final_dense);
        }
    }
}
