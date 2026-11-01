use simba::scalar::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, OMatrix, SquareMatrix};

use crate::linalg::lu;

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this square matrix.
    ///
    /// # Panics
    ///
    /// Panics if `self` isn’t a square matrix.
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        let mut me = self.into_owned();
        if me.try_inverse_mut() { Some(me) } else { None }
    }
}

impl<T: ComplexField, D: Dim, S: StorageMut<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this square matrix in-place.
    ///
    /// Returns `true` if the inversion succeeded, `false` otherwise.
    ///
    /// # Behavior
    ///
    /// - For small dimensions (`n < 5`), the matrix is left unchanged if the inversion fails.
    /// - For dimensions `n >= 5`, the matrix may be **partially modified** even if the inversion fails,
    ///   because LU decomposition is used and it modifies the matrix in-place.
    ///
    /// If you need to preserve the original matrix regardless of success or failure,
    /// consider using [`Self::try_inverse`] instead.
    ///
    /// # Panics
    ///
    /// Panics if `self` isn’t a square matrix.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where
        DefaultAllocator: Allocator<D, D>,
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

// NOTE: this is an efficient, loop-unrolled matrix inverse from MESA (MIT licensed) and further factorized.
fn do_inverse4<T: ComplexField, D: Dim, S: StorageMut<T, D, D>>(
    mat: &OMatrix<T, D, D>,
    out: &mut SquareMatrix<T, D, S>,
) -> bool
where
    DefaultAllocator: Allocator<D, D>,
{
    let a = unsafe { mat.get_unchecked((0, 0)) };
    let b = unsafe { mat.get_unchecked((0, 1)) };
    let c = unsafe { mat.get_unchecked((0, 2)) };
    let d = unsafe { mat.get_unchecked((0, 3)) };
    let e = unsafe { mat.get_unchecked((1, 0)) };
    let f = unsafe { mat.get_unchecked((1, 1)) };
    let g = unsafe { mat.get_unchecked((1, 2)) };
    let h = unsafe { mat.get_unchecked((1, 3)) };
    let i = unsafe { mat.get_unchecked((2, 0)) };
    let j = unsafe { mat.get_unchecked((2, 1)) };
    let k = unsafe { mat.get_unchecked((2, 2)) };
    let l = unsafe { mat.get_unchecked((2, 3)) };
    let m = unsafe { mat.get_unchecked((3, 0)) };
    let n = unsafe { mat.get_unchecked((3, 1)) };
    let o = unsafe { mat.get_unchecked((3, 2)) };
    let p = unsafe { mat.get_unchecked((3, 3)) };

    let cofactor00 = f.clone() * (k.clone() * p.clone() - l.clone() * o.clone())
        + j.clone() * (h.clone() * o.clone() - g.clone() * p.clone())
        + n.clone() * (g.clone() * l.clone() - h.clone() * k.clone());
    let cofactor01 = e.clone() * (l.clone() * o.clone() - k.clone() * p.clone())
        + i.clone() * (g.clone() * p.clone() - h.clone() * o.clone())
        + m.clone() * (h.clone() * k.clone() - g.clone() * l.clone());
    let cofactor02 = e.clone() * (j.clone() * p.clone() - l.clone() * n.clone())
        + i.clone() * (h.clone() * n.clone() - f.clone() * p.clone())
        + m.clone() * (f.clone() * l.clone() - h.clone() * j.clone());
    let cofactor03 = e.clone() * (k.clone() * n.clone() - j.clone() * o.clone())
        + i.clone() * (f.clone() * o.clone() - g.clone() * n.clone())
        + m.clone() * (g.clone() * j.clone() - f.clone() * k.clone());

    let det = a.clone() * cofactor00.clone()
        + b.clone() * cofactor01.clone()
        + c.clone() * cofactor02.clone()
        + d.clone() * cofactor03.clone();

    if det.is_zero() {
        return false;
    }

    out[(0, 0)] = cofactor00;
    out[(1, 0)] = b.clone() * (l.clone() * o.clone() - k.clone() * p.clone())
        + j.clone() * (c.clone() * p.clone() - d.clone() * o.clone())
        + n.clone() * (d.clone() * k.clone() - c.clone() * l.clone());
    out[(2, 0)] = b.clone() * (g.clone() * p.clone() - h.clone() * o.clone())
        + f.clone() * (d.clone() * o.clone() - c.clone() * p.clone())
        + n.clone() * (c.clone() * h.clone() - d.clone() * g.clone());
    out[(3, 0)] = b.clone() * (h.clone() * k.clone() - g.clone() * l.clone())
        + f.clone() * (c.clone() * l.clone() - d.clone() * k.clone())
        + j.clone() * (d.clone() * g.clone() - c.clone() * h.clone());

    out[(0, 1)] = cofactor01;
    out[(1, 1)] = a.clone() * (k.clone() * p.clone() - l.clone() * o.clone())
        + i.clone() * (d.clone() * o.clone() - c.clone() * p.clone())
        + m.clone() * (c.clone() * l.clone() - d.clone() * k.clone());
    out[(2, 1)] = a.clone() * (h.clone() * o.clone() - g.clone() * p.clone())
        + e.clone() * (c.clone() * p.clone() - d.clone() * o.clone())
        + m.clone() * (d.clone() * g.clone() - c.clone() * h.clone());
    out[(3, 1)] = a.clone() * (g.clone() * l.clone() - h.clone() * k.clone())
        + e.clone() * (d.clone() * k.clone() - c.clone() * l.clone())
        + i.clone() * (c.clone() * h.clone() - d.clone() * g.clone());

    out[(0, 2)] = cofactor02;
    out[(1, 2)] = a.clone() * (l.clone() * n.clone() - j.clone() * p.clone())
        + i.clone() * (b.clone() * p.clone() - d.clone() * n.clone())
        + m.clone() * (d.clone() * j.clone() - b.clone() * l.clone());
    out[(2, 2)] = a.clone() * (f.clone() * p.clone() - h.clone() * n.clone())
        + e.clone() * (d.clone() * n.clone() - b.clone() * p.clone())
        + m.clone() * (b.clone() * h.clone() - d.clone() * f.clone());
    out[(3, 2)] = a.clone() * (h.clone() * j.clone() - f.clone() * l.clone())
        + e.clone() * (b.clone() * l.clone() - d.clone() * j.clone())
        + i.clone() * (d.clone() * f.clone() - b.clone() * h.clone());

    out[(0, 3)] = cofactor03;
    out[(1, 3)] = a.clone() * (j.clone() * o.clone() - k.clone() * n.clone())
        + i.clone() * (c.clone() * n.clone() - b.clone() * o.clone())
        + m.clone() * (b.clone() * k.clone() - c.clone() * j.clone());
    out[(2, 3)] = a.clone() * (g.clone() * n.clone() - f.clone() * o.clone())
        + e.clone() * (b.clone() * o.clone() - c.clone() * n.clone())
        + m.clone() * (c.clone() * f.clone() - b.clone() * g.clone());
    out[(3, 3)] = a.clone() * (f.clone() * k.clone() - g.clone() * j.clone())
        + e.clone() * (c.clone() * j.clone() - b.clone() * k.clone())
        + i.clone() * (b.clone() * g.clone() - c.clone() * f.clone());

    let inv_det = T::one() / det;

    for j in 0..4 {
        for i in 0..4 {
            out[(i, j)] *= inv_det.clone();
        }
    }
    true
}
