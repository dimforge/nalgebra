// Matrix properties checks.
use num::{Zero, One};
use approx::ApproxEq;

use alga::general::{ClosedAdd, ClosedMul, ClosedSub, Field};

use core::{Scalar, Matrix, SquareMatrix};
use core::dimension::Dim;
use core::storage::Storage;


impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Indicates if this is a square matrix.
    #[inline]
    pub fn is_square(&self) -> bool {
        let shape = self.shape();
        shape.0 == shape.1
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
    // FIXME: ApproxEq prevents us from using those methods on integer matrices…
    where N: ApproxEq,
          N::Epsilon: Copy {
    /// Indicated if this is the identity matrix within a relative error of `eps`.
    ///
    /// If the matrix is diagonal, this checks that diagonal elements (i.e. at coordinates `(i, i)`
    /// for i from `0` to `min(R, C)`) are equal one; and that all other elements are zero.
    #[inline]
    pub fn is_identity(&self, eps: N::Epsilon) -> bool
        where N: Zero + One {
        let (nrows, ncols) = self.shape();
        let d;

        if nrows > ncols {
            d = ncols;

            for i in d .. nrows {
                for j in 0 .. ncols {
                    if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps) {
                        return false;
                    }
                }
            }
        }
        else { // nrows <= ncols
            d = nrows;

            for i in 0 .. nrows {
                for j in d .. ncols {
                    if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps) {
                        return false;
                    }
                }
            }
        }

        // Off-diagonal elements of the sub-square matrix.
        for i in 1 .. d {
            for j in 0 .. i {
                // FIXME: use unsafe indexing.
                if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps) ||
                   !relative_eq!(self[(j, i)], N::zero(), epsilon = eps) {
                    return false;
                }
            }
        }

        // Diagonal elements of the sub-square matrix.
        for i in 0 .. d {
            if !relative_eq!(self[(i, i)], N::one(), epsilon = eps) {
                return false;
            }
        }

        true
    }
}


impl<N: Scalar + ApproxEq, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S>
    where N: Zero + One + ClosedAdd + ClosedMul,
          N::Epsilon: Copy {
    /// Checks that this matrix is orthogonal, i.e., that it is square and `M × Mᵀ = Id`.
    ///
    /// In this definition `Id` is approximately equal to the identity matrix with a relative error
    /// equal to `eps`.
    #[inline]
    pub fn is_orthogonal(&self, eps: N::Epsilon) -> bool {
        self.is_square() && (self.tr_mul(self)).is_identity(eps)
    }

    /// Checks that this matrix is orthogonal and has a determinant equal to 1.
    #[inline]
    pub fn is_special_orthogonal(&self, eps: N::Epsilon) -> bool
        where N: ClosedSub + PartialOrd {
            self.is_orthogonal(eps) && self.determinant() > N::zero()
    }

    /// Returns `true` if this matrix is invertible.
    #[inline]
    pub fn is_invertible(&self) -> bool
        where N: Field {
        // FIXME: improve this?
        self.clone_owned().try_inverse().is_some()
    }
}
