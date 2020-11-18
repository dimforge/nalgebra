//! TODO
use nalgebra::{DMatrix, Scalar, Matrix, Dim};
use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;
use num_traits::Zero;
use std::ops::{Add, AddAssign};
use nalgebra::storage::Storage;

/// TODO
pub fn convert_dense_coo<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>
{
    let mut coo = CooMatrix::new(dense.nrows(), dense.ncols());

    for (index, v) in dense.iter().enumerate() {
        if v != &T::zero() {
            // We use the fact that matrix iteration is guaranteed to be column-major
            let i = index % dense.nrows();
            let j = index / dense.nrows();
            coo.push(i, j, v.inlined_clone());
        }
    }

    coo
}

/// TODO
///
/// TODO: What should the actual trait bounds be?
pub fn convert_coo_dense<T>(coo: &CooMatrix<T>) -> DMatrix<T>
where
    T: Scalar + Zero + Add + AddAssign,
{
    let mut output = DMatrix::repeat(coo.nrows(), coo.ncols(), T::zero());
    for (i, j, v) in coo.triplet_iter() {
        output[(i, j)] += v.inlined_clone();
    }
    output
}

/// TODO
pub fn convert_coo_csr<T>(_coo: &CooMatrix<T>) -> CsrMatrix<T> {
    todo!()
}