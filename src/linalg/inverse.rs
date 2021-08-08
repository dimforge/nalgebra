use simba::scalar::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, OMatrix, SquareMatrix};

use crate::linalg::lu;

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this matrix.
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<T, D, D>,
    {
        let mut me = self.into_owned();
        if me.try_inverse_mut() {
            Some(me)
        } else {
            None
        }
    }
}

impl<T: ComplexField, D: Dim, S: StorageMut<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this matrix in-place. Returns `false` and leaves `self` untouched if
    /// inversion fails.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where
        DefaultAllocator: Allocator<T, D, D>,
    {
        assert!(self.is_square(), "Unable to invert a non-square matrix.");

        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => true,
                1 => {
                    let determinant = self.get_unchecked((0, 0)).clone();
                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = T::one() / determinant;
                        true
                    }
                }
                2 => {
                    let m11 = self.get_unchecked((0, 0)).clone();
                    let m12 = self.get_unchecked((0, 1)).clone();
                    let m21 = self.get_unchecked((1, 0)).clone();
                    let m22 = self.get_unchecked((1, 1)).clone();

                    let determinant = m11.clone() * m22.clone() - m21.clone() * m12.clone();

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = m22 / determinant.clone();
                        *self.get_unchecked_mut((0, 1)) = -m12 / determinant.clone();

                        *self.get_unchecked_mut((1, 0)) = -m21 / determinant.clone();
                        *self.get_unchecked_mut((1, 1)) = m11 / determinant;

                        true
                    }
                }
                3 => {
                    let m11 = self.get_unchecked((0, 0)).clone();
                    let m12 = self.get_unchecked((0, 1)).clone();
                    let m13 = self.get_unchecked((0, 2)).clone();

                    let m21 = self.get_unchecked((1, 0)).clone();
                    let m22 = self.get_unchecked((1, 1)).clone();
                    let m23 = self.get_unchecked((1, 2)).clone();

                    let m31 = self.get_unchecked((2, 0)).clone();
                    let m32 = self.get_unchecked((2, 1)).clone();
                    let m33 = self.get_unchecked((2, 2)).clone();

                    let minor_m12_m23 = m22.clone() * m33.clone() - m32.clone() * m23.clone();
                    let minor_m11_m23 = m21.clone() * m33.clone() - m31.clone() * m23.clone();
                    let minor_m11_m22 = m21.clone() * m32.clone() - m31.clone() * m22.clone();

                    let determinant = m11.clone() * minor_m12_m23.clone()
                        - m12.clone() * minor_m11_m23.clone()
                        + m13.clone() * minor_m11_m22.clone();

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = minor_m12_m23 / determinant.clone();
                        *self.get_unchecked_mut((0, 1)) = (m13.clone() * m32.clone()
                            - m33.clone() * m12.clone())
                            / determinant.clone();
                        *self.get_unchecked_mut((0, 2)) = (m12.clone() * m23.clone()
                            - m22.clone() * m13.clone())
                            / determinant.clone();

                        *self.get_unchecked_mut((1, 0)) = -minor_m11_m23 / determinant.clone();
                        *self.get_unchecked_mut((1, 1)) =
                            (m11.clone() * m33 - m31.clone() * m13.clone()) / determinant.clone();
                        *self.get_unchecked_mut((1, 2)) =
                            (m13 * m21.clone() - m23 * m11.clone()) / determinant.clone();

                        *self.get_unchecked_mut((2, 0)) = minor_m11_m22 / determinant.clone();
                        *self.get_unchecked_mut((2, 1)) =
                            (m12.clone() * m31 - m32 * m11.clone()) / determinant.clone();
                        *self.get_unchecked_mut((2, 2)) = (m11 * m22 - m21 * m12) / determinant;

                        true
                    }
                }
                4 => {
                    let oself = self.clone_owned();
                    do_inverse4(&oself, self)
                }
                _ => {
                    let oself = self.clone_owned();
                    lu::try_invert_to(oself, self)
                }
            }
        }
    }
}

// NOTE: this is an extremely efficient, loop-unrolled matrix inverse from MESA (MIT licensed).
fn do_inverse4<T: ComplexField, D: Dim, S: StorageMut<T, D, D>>(
    m: &OMatrix<T, D, D>,
    out: &mut SquareMatrix<T, D, S>,
) -> bool
where
    DefaultAllocator: Allocator<T, D, D>,
{
    let m = m.as_slice();

    out[(0, 0)] = m[5].clone() * m[10].clone() * m[15].clone()
        - m[5].clone() * m[11].clone() * m[14].clone()
        - m[9].clone() * m[6].clone() * m[15].clone()
        + m[9].clone() * m[7].clone() * m[14].clone()
        + m[13].clone() * m[6].clone() * m[11].clone()
        - m[13].clone() * m[7].clone() * m[10].clone();

    out[(1, 0)] = -m[1].clone() * m[10].clone() * m[15].clone()
        + m[1].clone() * m[11].clone() * m[14].clone()
        + m[9].clone() * m[2].clone() * m[15].clone()
        - m[9].clone() * m[3].clone() * m[14].clone()
        - m[13].clone() * m[2].clone() * m[11].clone()
        + m[13].clone() * m[3].clone() * m[10].clone();

    out[(2, 0)] = m[1].clone() * m[6].clone() * m[15].clone()
        - m[1].clone() * m[7].clone() * m[14].clone()
        - m[5].clone() * m[2].clone() * m[15].clone()
        + m[5].clone() * m[3].clone() * m[14].clone()
        + m[13].clone() * m[2].clone() * m[7].clone()
        - m[13].clone() * m[3].clone() * m[6].clone();

    out[(3, 0)] = -m[1].clone() * m[6].clone() * m[11].clone()
        + m[1].clone() * m[7].clone() * m[10].clone()
        + m[5].clone() * m[2].clone() * m[11].clone()
        - m[5].clone() * m[3].clone() * m[10].clone()
        - m[9].clone() * m[2].clone() * m[7].clone()
        + m[9].clone() * m[3].clone() * m[6].clone();

    out[(0, 1)] = -m[4].clone() * m[10].clone() * m[15].clone()
        + m[4].clone() * m[11].clone() * m[14].clone()
        + m[8].clone() * m[6].clone() * m[15].clone()
        - m[8].clone() * m[7].clone() * m[14].clone()
        - m[12].clone() * m[6].clone() * m[11].clone()
        + m[12].clone() * m[7].clone() * m[10].clone();

    out[(1, 1)] = m[0].clone() * m[10].clone() * m[15].clone()
        - m[0].clone() * m[11].clone() * m[14].clone()
        - m[8].clone() * m[2].clone() * m[15].clone()
        + m[8].clone() * m[3].clone() * m[14].clone()
        + m[12].clone() * m[2].clone() * m[11].clone()
        - m[12].clone() * m[3].clone() * m[10].clone();

    out[(2, 1)] = -m[0].clone() * m[6].clone() * m[15].clone()
        + m[0].clone() * m[7].clone() * m[14].clone()
        + m[4].clone() * m[2].clone() * m[15].clone()
        - m[4].clone() * m[3].clone() * m[14].clone()
        - m[12].clone() * m[2].clone() * m[7].clone()
        + m[12].clone() * m[3].clone() * m[6].clone();

    out[(3, 1)] = m[0].clone() * m[6].clone() * m[11].clone()
        - m[0].clone() * m[7].clone() * m[10].clone()
        - m[4].clone() * m[2].clone() * m[11].clone()
        + m[4].clone() * m[3].clone() * m[10].clone()
        + m[8].clone() * m[2].clone() * m[7].clone()
        - m[8].clone() * m[3].clone() * m[6].clone();

    out[(0, 2)] = m[4].clone() * m[9].clone() * m[15].clone()
        - m[4].clone() * m[11].clone() * m[13].clone()
        - m[8].clone() * m[5].clone() * m[15].clone()
        + m[8].clone() * m[7].clone() * m[13].clone()
        + m[12].clone() * m[5].clone() * m[11].clone()
        - m[12].clone() * m[7].clone() * m[9].clone();

    out[(1, 2)] = -m[0].clone() * m[9].clone() * m[15].clone()
        + m[0].clone() * m[11].clone() * m[13].clone()
        + m[8].clone() * m[1].clone() * m[15].clone()
        - m[8].clone() * m[3].clone() * m[13].clone()
        - m[12].clone() * m[1].clone() * m[11].clone()
        + m[12].clone() * m[3].clone() * m[9].clone();

    out[(2, 2)] = m[0].clone() * m[5].clone() * m[15].clone()
        - m[0].clone() * m[7].clone() * m[13].clone()
        - m[4].clone() * m[1].clone() * m[15].clone()
        + m[4].clone() * m[3].clone() * m[13].clone()
        + m[12].clone() * m[1].clone() * m[7].clone()
        - m[12].clone() * m[3].clone() * m[5].clone();

    out[(0, 3)] = -m[4].clone() * m[9].clone() * m[14].clone()
        + m[4].clone() * m[10].clone() * m[13].clone()
        + m[8].clone() * m[5].clone() * m[14].clone()
        - m[8].clone() * m[6].clone() * m[13].clone()
        - m[12].clone() * m[5].clone() * m[10].clone()
        + m[12].clone() * m[6].clone() * m[9].clone();

    out[(3, 2)] = -m[0].clone() * m[5].clone() * m[11].clone()
        + m[0].clone() * m[7].clone() * m[9].clone()
        + m[4].clone() * m[1].clone() * m[11].clone()
        - m[4].clone() * m[3].clone() * m[9].clone()
        - m[8].clone() * m[1].clone() * m[7].clone()
        + m[8].clone() * m[3].clone() * m[5].clone();

    out[(1, 3)] = m[0].clone() * m[9].clone() * m[14].clone()
        - m[0].clone() * m[10].clone() * m[13].clone()
        - m[8].clone() * m[1].clone() * m[14].clone()
        + m[8].clone() * m[2].clone() * m[13].clone()
        + m[12].clone() * m[1].clone() * m[10].clone()
        - m[12].clone() * m[2].clone() * m[9].clone();

    out[(2, 3)] = -m[0].clone() * m[5].clone() * m[14].clone()
        + m[0].clone() * m[6].clone() * m[13].clone()
        + m[4].clone() * m[1].clone() * m[14].clone()
        - m[4].clone() * m[2].clone() * m[13].clone()
        - m[12].clone() * m[1].clone() * m[6].clone()
        + m[12].clone() * m[2].clone() * m[5].clone();

    out[(3, 3)] = m[0].clone() * m[5].clone() * m[10].clone()
        - m[0].clone() * m[6].clone() * m[9].clone()
        - m[4].clone() * m[1].clone() * m[10].clone()
        + m[4].clone() * m[2].clone() * m[9].clone()
        + m[8].clone() * m[1].clone() * m[6].clone()
        - m[8].clone() * m[2].clone() * m[5].clone();

    let det = m[0].clone() * out[(0, 0)].clone()
        + m[1].clone() * out[(0, 1)].clone()
        + m[2].clone() * out[(0, 2)].clone()
        + m[3].clone() * out[(0, 3)].clone();

    if !det.is_zero() {
        let inv_det = T::one() / det;

        for j in 0..4 {
            for i in 0..4 {
                out[(i, j)] *= inv_det.clone();
            }
        }
        true
    } else {
        false
    }
}
