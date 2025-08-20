use crate::num::FromPrimitive;
use na::{ComplexField, Dim, IsContiguous, Matrix, OMatrix, RawStorage, Scalar, Storage};

/// describes different algorithms of calculating a matrix rank from the upper
/// diagonal R matrix of a column-pivoted QR decomposition.
pub enum RankEstimationAlgo<T: ComplexField> {
    /// this simplest (and cheapest) strategy is to count the diagonal entries
    /// >= epsilon and count those as the rank.
    FixedEps(T::RealField),
    /// like fixed eps, but the epsilon is scaled with max(M,N) of the matrix
    /// where M: number of rows, N:number of columns
    ScaledEps(T::RealField),
}

// WARNING: qr must be a qr decomposition of a matrix where the upper triangular
// part stores the matrix R
pub(crate) fn calculate_rank<T, R, C, S>(
    qr: &Matrix<T, R, C, S>,
    method: RankEstimationAlgo<T>,
) -> usize
where
    T: Scalar + ComplexField + Copy,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + RawStorage<T, R, C> + IsContiguous,
{
    match method {
        RankEstimationAlgo::FixedEps(eps) => calculate_rank_with_fixed_eps(qr, eps),
        RankEstimationAlgo::ScaledEps(eps) => calculate_rank_with_fixed_eps(
            qr,
            eps * T::RealField::from_usize(qr.nrows().max(qr.ncols()))
                .expect("matrix dimensions out of floating point bounds"),
        ),
    }
}

fn calculate_rank_with_fixed_eps<T, R, C, S>(qr: &Matrix<T, R, C, S>, eps: T::RealField) -> usize
where
    T: Scalar + ComplexField + Copy,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + RawStorage<T, R, C> + IsContiguous,
{
    let eps = eps.abs();
    let dim = qr.nrows().min(qr.ncols());
    let mut rank = 0;
    for j in 0..dim {
        if qr[(j, j)].abs() >= eps {
            rank += 1;
        }
    }
    rank
}
