//! Matrix operations involving sparse matrices.

use crate::coo::CooMatrix;
use nalgebra::base::storage::{Storage, StorageMut};
use nalgebra::{ClosedAdd, ClosedMul, Dim, Scalar, Vector};
use num_traits::{One, Zero};

/// Sparse matrix-vector multiplication `y = beta * y + alpha * A * x`.
///
/// Computes a matrix-vector product with the COO matrix "A" and the vector `x`, storing the
/// result in `y`.
///
/// If `beta == 0`, the elements in `y` are never read.
///
/// TODO: Rethink this function
///
/// Panics
/// ------
///
/// Panics if `y`, `a` and `x` do not have compatible dimensions.
pub fn spmv_coo<T, Y, X, YDim, XDim>(
    beta: T,
    y: &mut Vector<T, YDim, Y>,
    alpha: T,
    a: &CooMatrix<T>,
    x: &Vector<T, XDim, X>,
) where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
    YDim: Dim,
    XDim: Dim,
    Y: StorageMut<T, YDim>,
    X: Storage<T, XDim>,
{
    assert_eq!(
        y.len(),
        a.nrows(),
        "y and a must be dimensionally compatible"
    );
    assert_eq!(
        a.ncols(),
        x.len(),
        "a and x must be dimensionally compatible"
    );

    if beta == T::zero() {
        // If `y` is constructed through `new_uninitialized()`, we must make sure to not read
        // any of the elements in order to avoid UB, so we special case beta == 0
        // in order to ensure that we only write, not read, the elements in y.
        for y_i in y.iter_mut() {
            *y_i = T::zero();
        }
    } else if beta != T::one() {
        // Since the COO triplets have no particular structure, we cannot combine initialization
        // of y with the triplet loop below, and instead have to do it in a pre-pass.
        for y_i in y.iter_mut() {
            *y_i *= beta.inlined_clone();
        }
    }

    for (i, j, v) in a.triplet_iter() {
        // TODO: We could skip bounds checks with unsafe here, since COO ensures that all indices
        // are in bounds and we assert on dimensions up-front.
        // The compiler will not be able to elide the checks, since we're doing
        // random/unpredictable access to elements in `x` and `y`.
        let (alpha, v, x_j) = (
            alpha.inlined_clone(),
            v.inlined_clone(),
            x[j].inlined_clone(),
        );
        y[i] += alpha * v * x_j;
    }
}
