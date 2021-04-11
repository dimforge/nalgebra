use simba::scalar::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::DimMin;
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, SquareMatrix};

use crate::linalg::LU;

impl<T: ComplexField, D: DimMin<D, Output = D>, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Computes the matrix determinant.
    ///
    /// If the matrix has a dimension larger than 3, an LU decomposition is used.
    #[inline]
    pub fn determinant(&self) -> T
    where
        DefaultAllocator: Allocator<T, D, D> + Allocator<(usize, usize), D>,
    {
        assert!(
            self.is_square(),
            "Unable to compute the determinant of a non-square matrix."
        );
        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => T::one(),
                1 => *self.get_unchecked((0, 0)),
                2 => {
                    let m11 = *self.get_unchecked((0, 0));
                    let m12 = *self.get_unchecked((0, 1));
                    let m21 = *self.get_unchecked((1, 0));
                    let m22 = *self.get_unchecked((1, 1));

                    m11 * m22 - m21 * m12
                }
                3 => {
                    let m11 = *self.get_unchecked((0, 0));
                    let m12 = *self.get_unchecked((0, 1));
                    let m13 = *self.get_unchecked((0, 2));

                    let m21 = *self.get_unchecked((1, 0));
                    let m22 = *self.get_unchecked((1, 1));
                    let m23 = *self.get_unchecked((1, 2));

                    let m31 = *self.get_unchecked((2, 0));
                    let m32 = *self.get_unchecked((2, 1));
                    let m33 = *self.get_unchecked((2, 2));

                    let minor_m12_m23 = m22 * m33 - m32 * m23;
                    let minor_m11_m23 = m21 * m33 - m31 * m23;
                    let minor_m11_m22 = m21 * m32 - m31 * m22;

                    m11 * minor_m12_m23 - m12 * minor_m11_m23 + m13 * minor_m11_m22
                }
                _ => LU::new(self.clone_owned()).determinant(),
            }
        }
    }
}
