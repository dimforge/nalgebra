use num::{One, Zero};

use simba::scalar::{ClosedAdd, ClosedMul, SupersetOf};

use crate::base::{SMatrix, Scalar};

use crate::geometry::Rotation;

/// # Identity
impl<T, const D: usize> Rotation<T, D>
where
    T: Scalar + Zero + One,
{
    /// Creates a new square identity rotation of the given `dimension`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let rot1 = Quaternion::identity();
    /// let rot2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// assert_eq!(rot1 * rot2, rot2);
    /// assert_eq!(rot2 * rot1, rot2);
    /// ```
    #[inline]
    pub fn identity() -> Rotation<T, D> {
        Self::from_matrix_unchecked(SMatrix::<T, D, D>::identity())
    }
}

impl<T: Scalar, const D: usize> Rotation<T, D> {
    /// Cast the components of `self` to another type.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::<f64>::identity();
    /// let rot2 = rot.cast::<f32>();
    /// assert_eq!(rot2, Rotation2::<f32>::identity());
    /// ```
    pub fn cast<To: Scalar>(self) -> Rotation<To, D>
    where
        Rotation<To, D>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T, const D: usize> One for Rotation<T, D>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
