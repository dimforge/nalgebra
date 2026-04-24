use crate::proptest::*;
use na::{
    DMatrix, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimSub, Matrix, Matrix5x3,
    OMatrix, RawStorage, SVD, U1, allocator::Allocator,
};
use num_traits::Zero;
use proptest::prelude::*;

pub fn is_upper_triangular<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> bool
where
    T: Zero + PartialEq,
    C: Dim,
    R: Dim,
    S: RawStorage<T, R, C>,
{
    let ncols = mat.ncols();
    let nrows = mat.nrows();

    let zero = T::zero();
    for c in 0..ncols {
        for r in c + 1..nrows {
            if mat[(r, c)] != zero {
                return false;
            }
        }
    }
    return true;
}

pub fn square_or_overdetermined_dmatrix() -> impl Strategy<Value = DMatrix<f64>> {
    PROPTEST_MATRIX_DIM.prop_flat_map(|rows| {
        (1..=rows).prop_flat_map(move |cols| matrix(PROPTEST_F64, rows..=rows, cols..=cols))
    })
}

/// give us A and B matrices of the system AX = B
pub fn linear_system_dynamic() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    square_or_overdetermined_dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.nrows(), PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

/// give us A and B matrices of the system AX = B, where A has full rank
pub fn full_rank_linear_system_dynamic() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    square_or_overdetermined_dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.nrows(), PROPTEST_MATRIX_DIM);
        (Just(make_full_rank(a)), b)
    })
}

pub fn full_rank_matrix5x3() -> impl Strategy<Value = Matrix5x3<f64>> {
    matrix5x3().prop_flat_map(|a| Just(make_full_rank(a)))
}

fn make_full_rank<R, C>(a: OMatrix<f64, R, C>) -> OMatrix<f64, R, C>
where
    R: Dim,
    C: Dim + DimSub<na::Const<1>>,
    R: DimMin<C, Output = C>,
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<DimMinimum<R, C>, C>
        + Allocator<R, DimMinimum<R, C>>
        + Allocator<DimMinimum<R, C>>,
{
    // this is very brute force, but it should work:
    // we make a matrix full rank by moving the eigenvalues away from zero
    // in the direction that they are already pointing.
    let (nrows, ncols) = (a.nrows(), a.ncols());
    let svd = SVD::new(a, true, true);
    let u = svd.u.unwrap();
    let mut v_t = svd.v_t.unwrap();
    let mut singular_values = svd.singular_values;
    // see also the rank determination algorithm of the octave library.
    // We either set the lower bound for the singular values as a fraction
    // of the largest value, but no less than sqrt(EPS).
    let lambda_lower_bound_abs =
        (f64::EPSILON * singular_values.max() * (nrows as f64).max(ncols as f64))
            .abs()
            .max(f64::EPSILON.sqrt());
    // now we set the magnitude of the singular value at least
    // to the lower bound, but we keep the sign
    singular_values.iter_mut().for_each(|lambda| {
        // sign should be +1 or -1, not zero
        let sign = lambda.signum();
        let lambda_abs = lambda.abs();
        // the duplicate ma
        *lambda = sign * lambda_abs.max(lambda_lower_bound_abs);
    });

    // this is singular values * V^T
    v_t.row_iter_mut()
        .zip(singular_values.iter())
        .for_each(|(mut v_row, sigma)| {
            v_row *= *sigma;
        });

    let a_new = u * v_t;

    debug_assert_eq!(
        a_new.rank(f64::EPSILON.sqrt()),
        a_new.ncols().min(a_new.nrows())
    );
    a_new
}
