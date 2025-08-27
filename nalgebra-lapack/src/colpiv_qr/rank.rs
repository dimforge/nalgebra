use crate::num::FromPrimitive;
use na::ComplexField;
use na::{Dim, IsContiguous, Matrix, RawStorage, RealField, Scalar, Storage};
use num::{ConstOne, ToPrimitive};
use num::{Float, float::TotalOrder};

#[cfg(test)]
mod test;

/// describes different algorithms of calculating a matrix rank from the upper
/// diagonal R matrix of a column-pivoted QR decomposition.
pub enum RankEstimationAlgo<T: RealField> {
    /// this simplest (and cheapest) strategy is to estimate the rank of the
    /// matrix $$AP = QR$$ as the number of diagonal elements $$R_{ii}$$ greater
    /// than this fixed lower bound.
    ///
    /// **Note**: this is the fastest method, but
    /// should be used with extreme care, because it can easily produce bad
    /// results even in trivial cases.
    FixedLowerBound(T::RealField),
    //@todo
    ScaledEps1,
    //@todo
    ScaledEps2,
}

impl<T> Default for RankEstimationAlgo<T>
where
    T: RealField + Float,
{
    fn default() -> Self {
        Self::ScaledEps2
    }
}

// WARNING: qr must be a qr decomposition of a matrix where the upper triangular
// part stores the matrix R
pub(crate) fn calculate_rank<T, R, C, S>(
    qr: &Matrix<T, R, C, S>,
    method: RankEstimationAlgo<T>,
) -> usize
where
    T: Scalar + RealField + Copy + Float + TotalOrder,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + RawStorage<T, R, C> + IsContiguous,
{
    match method {
        RankEstimationAlgo::FixedLowerBound(eps) => calculate_rank_with_fixed_minimum(qr, eps),
        RankEstimationAlgo::ScaledEps1 => {
            let r_max = calculate_max_abs_diag(qr);

            let tol = r_max
                * T::epsilon()
                * T::RealField::from_usize(qr.nrows().max(qr.ncols()))
                    .expect("matrix dimensions out of floating point bounds");
            calculate_rank_with_fixed_minimum(qr, tol)
        }
        RankEstimationAlgo::ScaledEps2 => {
            let r_max = calculate_max_abs_diag(qr);

            let tol = eps(r_max)
                * T::RealField::from_usize(qr.nrows().max(qr.ncols()))
                    .expect("matrix dimensions out of floating point bounds");
            println!("rmax = {r_max} tol = {tol}");
            calculate_rank_with_fixed_minimum(qr, tol)
        }
    }
}

fn calculate_rank_with_fixed_minimum<T, R, C, S>(qr: &Matrix<T, R, C, S>, eps: T) -> usize
where
    T: Scalar + RealField + Copy,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + RawStorage<T, R, C> + IsContiguous,
{
    let eps = eps.abs();
    let dim = qr.nrows().min(qr.ncols());
    let mut rank = 0;
    for j in 0..dim {
        if qr[(j, j)].abs() > eps {
            rank += 1;
        }
    }
    rank
}

/// helper function to calculate the maximum diagonal element of a matrix
fn calculate_max_abs_diag<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> T
where
    T: Scalar + RealField + Copy + TotalOrder,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + RawStorage<T, R, C> + IsContiguous,
{
    let dim = mat.nrows().min(mat.ncols());

    (0..dim)
        .flat_map(|idx| {
            let val = mat[(idx, idx)];
            val.is_finite().then_some(val.abs())
        })
        .max_by(T::total_cmp)
        .unwrap_or(T::zero())
}

//@todo document
fn eps<T>(x: T) -> T
where
    T: ToPrimitive,
    T: Float,
{
    let x = x.abs();
    if x < T::min_positive_value() {
        return T::min_positive_value();
    }
    let Some(exponent) = T::log2(x).floor().to_i32() else {
        return T::min_positive_value();
    };

    let two = T::one() + T::one();
    two.powi(exponent) * T::epsilon()
}
