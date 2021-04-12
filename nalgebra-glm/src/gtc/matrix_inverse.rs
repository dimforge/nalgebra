use na::RealField;

use crate::aliases::TMat;

/// Fast matrix inverse for affine matrix.
pub fn affine_inverse<T: RealField, const D: usize>(m: TMat<T, D, D>) -> TMat<T, D, D> {
    // TODO: this should be optimized.
    m.try_inverse().unwrap_or_else(TMat::<_, D, D>::zeros)
}

/// Compute the transpose of the inverse of a matrix.
pub fn inverse_transpose<T: RealField, const D: usize>(m: TMat<T, D, D>) -> TMat<T, D, D> {
    m.try_inverse()
        .unwrap_or_else(TMat::<_, D, D>::zeros)
        .transpose()
}
