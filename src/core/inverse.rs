use approx::ApproxEq;

use alga::general::Field;

use core::{Scalar, Matrix, SquareMatrix, OwnedSquareMatrix};
use core::dimension::Dim;
use core::storage::{Storage, StorageMut};


impl<N, D: Dim, S> SquareMatrix<N, D, S>
    where N: Scalar + Field + ApproxEq,
          S: Storage<N, D, D> {
    /// Attempts to invert this matrix.
    #[inline]
    pub fn try_inverse(self) -> Option<OwnedSquareMatrix<N, D, S::Alloc>> {
        let mut res = self.into_owned();

        if res.shape().0 <= 3 {
            if res.try_inverse_mut() {
                Some(res)
            }
            else {
                None
            }
        }
        else {
            gauss_jordan_inverse(res)
        }
    }
}


impl<N, D: Dim, S> SquareMatrix<N, D, S>
    where N: Scalar + Field + ApproxEq,
          S: StorageMut<N, D, D> {
    /// Attempts to invert this matrix in-place. Returns `false` and leaves `self` untouched if
    /// inversion fails.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        assert!(self.is_square(), "Unable to invert a non-square matrix.");

        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => true,
                1 => {
                    if relative_eq!(self.get_unchecked(0, 0), &N::zero()) {
                        false
                    }
                    else {
                        *self.get_unchecked_mut(0, 0) = N::one() / self.determinant();
                        true
                    }
                },
                2 => {
                    let determinant = self.determinant();

                    if relative_eq!(&determinant, &N::zero()) {
                        false
                    }
                    else {
                        let m11 = *self.get_unchecked(0, 0); let m12 = *self.get_unchecked(0, 1);
                        let m21 = *self.get_unchecked(1, 0); let m22 = *self.get_unchecked(1, 1);

                        *self.get_unchecked_mut(0, 0) =  m22 / determinant;
                        *self.get_unchecked_mut(0, 1) = -m12 / determinant;

                        *self.get_unchecked_mut(1, 0) = -m21 / determinant;
                        *self.get_unchecked_mut(1, 1) =  m11 / determinant;

                        true
                    }
                },
                3 => {
                    let m11 = *self.get_unchecked(0, 0);
                    let m12 = *self.get_unchecked(0, 1);
                    let m13 = *self.get_unchecked(0, 2);

                    let m21 = *self.get_unchecked(1, 0);
                    let m22 = *self.get_unchecked(1, 1);
                    let m23 = *self.get_unchecked(1, 2);

                    let m31 = *self.get_unchecked(2, 0);
                    let m32 = *self.get_unchecked(2, 1);
                    let m33 = *self.get_unchecked(2, 2);


                    let minor_m12_m23 = m22 * m33 - m32 * m23;
                    let minor_m11_m23 = m21 * m33 - m31 * m23;
                    let minor_m11_m22 = m21 * m32 - m31 * m22;

                    let determinant = m11 * minor_m12_m23 -
                                      m12 * minor_m11_m23 +
                                      m13 * minor_m11_m22;

                    if relative_eq!(&determinant, &N::zero()) {
                        false
                    }
                    else {
                        *self.get_unchecked_mut(0, 0) = minor_m12_m23 / determinant;
                        *self.get_unchecked_mut(0, 1) = (m13 * m32 - m33 * m12) / determinant;
                        *self.get_unchecked_mut(0, 2) = (m12 * m23 - m22 * m13) / determinant;

                        *self.get_unchecked_mut(1, 0) = -minor_m11_m23 / determinant;
                        *self.get_unchecked_mut(1, 1) = (m11 * m33 - m31 * m13) / determinant;
                        *self.get_unchecked_mut(1, 2) = (m13 * m21 - m23 * m11) / determinant;

                        *self.get_unchecked_mut(2, 0) = minor_m11_m22  / determinant;
                        *self.get_unchecked_mut(2, 1) = (m12 * m31 - m32 * m11) / determinant;
                        *self.get_unchecked_mut(2, 2) = (m11 * m22 - m21 * m12) / determinant;

                        true
                    }
                },
                _ => {
                    let oself = self.clone_owned();
                    if let Some(res) = gauss_jordan_inverse(oself) {
                        self.copy_from(&res);
                        true
                    }
                    else {
                        false
                    }
                }
            }
        }
    }
}


/// Inverts the given matrix using Gauss-Jordan Ellimitation.
fn gauss_jordan_inverse<N, D, S>(mut matrix: SquareMatrix<N, D, S>) -> Option<OwnedSquareMatrix<N, D, S::Alloc>>
    where D: Dim,
          N: Scalar + Field + ApproxEq,
          S: StorageMut<N, D, D> {

    assert!(matrix.is_square(), "Unable to invert a non-square matrix.");
    let dim = matrix.data.shape().0;
    let mut res: OwnedSquareMatrix<N, D, S::Alloc> = Matrix::identity_generic(dim, dim);
    let dim = dim.value();

    unsafe {
        for k in 0 .. dim {
            // Search a non-zero value on the k-th column.
            // FIXME: would it be worth it to spend some more time searching for the
            // max instead?

            let mut n0 = k; // index of a non-zero entry.

            while n0 != dim {
                if !matrix.get_unchecked(n0, k).is_zero() {
                    break;
                }

                n0 += 1;
            }

            if n0 == dim {
                return None
            }

            // Swap pivot line.
            if n0 != k {
                for j in 0 .. dim {
                    matrix.swap_unchecked((n0, j), (k, j));
                    res.swap_unchecked((n0, j), (k, j));
                }
            }

            let pivot = *matrix.get_unchecked(k, k);

            for j in k .. dim {
                let selfval = *matrix.get_unchecked(k, j) / pivot;
                *matrix.get_unchecked_mut(k, j) = selfval;
            }

            for j in 0 .. dim {
                let resval = *res.get_unchecked(k, j) / pivot;
                *res.get_unchecked_mut(k, j) = resval;
            }

            for l in 0 .. dim {
                if l != k {
                    let normalizer = *matrix.get_unchecked(l, k);

                    for j in k .. dim {
                        let selfval = *matrix.get_unchecked(l, j) - *matrix.get_unchecked(k, j) * normalizer;
                        *matrix.get_unchecked_mut(l, j) = selfval;
                    }

                    for j in 0 .. dim {
                        let resval  = *res.get_unchecked(l, j) - *res.get_unchecked(k, j) * normalizer;
                        *res.get_unchecked_mut(l, j) = resval;
                    }
                }
            }
        }

        Some(res)
    }
}
