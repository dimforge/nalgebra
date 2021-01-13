//! Implements core traits for use with `matrixcompare`.
use crate::csr::CsrMatrix;
use crate::csc::CscMatrix;
use matrixcompare_core;
use matrixcompare_core::{Access, SparseAccess};
use crate::coo::CooMatrix;

macro_rules! impl_matrix_for_csr_csc {
    ($MatrixType:ident) => {
    impl<T: Clone> SparseAccess<T> for $MatrixType<T> {
        fn nnz(&self) -> usize {
            $MatrixType::nnz(self)
        }

        fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
            self.triplet_iter().map(|(i, j, v)| (i, j, v.clone())).collect()
        }
    }

    impl<T: Clone> matrixcompare_core::Matrix<T> for $MatrixType<T> {
        fn rows(&self) -> usize {
            self.nrows()
        }

        fn cols(&self) -> usize {
            self.ncols()
        }

        fn access(&self) -> Access<T> {
            Access::Sparse(self)
        }
    }
    }
}

impl_matrix_for_csr_csc!(CsrMatrix);
impl_matrix_for_csr_csc!(CscMatrix);

impl<T: Clone> SparseAccess<T> for CooMatrix<T> {
    fn nnz(&self) -> usize {
        CooMatrix::nnz(self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplet_iter().map(|(i, j, v)| (i, j, v.clone())).collect()
    }
}

impl<T: Clone> matrixcompare_core::Matrix<T> for CooMatrix<T> {
    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn access(&self) -> Access<T> {
        Access::Sparse(self)
    }
}