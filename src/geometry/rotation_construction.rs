use num::{One, Zero};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign, RealField, SupersetOf};

use crate::base::{storage::Storage, Const, SMatrix, Scalar, Unit, Vector};

use crate::geometry::Rotation;

impl<T, const D: usize> Default for Rotation<T, D>
where
    T: Scalar + Zero + One,
{
    fn default() -> Self {
        Self::identity()
    }
}

/// # Identity
impl<T, const D: usize> Rotation<T, D>
where
    T: Scalar + Zero + One,
{
    /// Creates a new square identity rotation of the given `dimension`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3};
    /// # use nalgebra::Vector3;
    /// let rot1 = Rotation2::identity();
    /// let rot2 = Rotation2::new(std::f32::consts::FRAC_PI_2);
    ///
    /// assert_eq!(rot1 * rot2, rot2);
    /// assert_eq!(rot2 * rot1, rot2);
    ///
    /// let rot1 = Rotation3::identity();
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2);
    ///
    /// assert_eq!(rot1 * rot2, rot2);
    /// assert_eq!(rot2 * rot1, rot2);
    /// ```
    #[inline]
    pub fn identity() -> Rotation<T, D> {
        Self::from_matrix_unchecked(SMatrix::<T, D, D>::identity())
    }
}

/// # Construction in any dimension
impl<T: RealField, const D: usize> Rotation<T, D> {
    /// The n-dimensional rotation matrix described by an oriented minor arc.
    ///
    /// This is the rotation `rot` aligning `from` with `to` over their minor angle such that
    /// `(rot * from).angle(to) == 0` and `(rot * from).dot(to).is_positive()`.
    ///
    /// Returns `None` with `from` and `to` being anti-parallel. In contrast to
    /// [`Self::from_arc_angle`], this method is robust for approximately parallel vectors
    /// continuously approaching identity.
    ///
    /// See also [`OMatrix::from_arc`](`crate::OMatrix::from_arc`) for owned matrices generic over
    /// storage.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation, Unit, Vector6};
    /// let from = Unit::new_normalize(Vector6::new(-4.0, -2.4, 0.0, -3.3, -1.0, -9.0));
    /// let to = Unit::new_normalize(Vector6::new(3.0, 1.0, 2.0, 2.0, 9.0, 6.0));
    ///
    /// // Aligns `from` with `to`.
    /// let rot = Rotation::from_arc(&from, &to).unwrap();
    /// assert_relative_eq!(rot * from, to, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot.inverse() * to, from, epsilon = 1.0e-6);
    ///
    /// // Returns identity with `from` and `to` being parallel.
    /// let rot = Rotation::from_arc(&from, &from).unwrap();
    /// assert_relative_eq!(rot, Rotation::identity(), epsilon = 1.0e-6);
    ///
    /// // Returns `None` with `from` and `to` being anti-parallel.
    /// assert!(Rotation::from_arc(&from, &-from).is_none());
    /// ```
    #[must_use]
    #[inline]
    pub fn from_arc<SB, SC>(
        from: &Unit<Vector<T, Const<D>, SB>>,
        to: &Unit<Vector<T, Const<D>, SC>>,
    ) -> Option<Self>
    where
        T: RealField,
        SB: Storage<T, Const<D>> + Clone,
        SC: Storage<T, Const<D>> + Clone,
    {
        SMatrix::from_arc(from, to).map(Self::from_matrix_unchecked)
    }
    /// The n-dimensional rotation matrix described by an oriented minor arc and a signed angle.
    ///
    /// Returns `None` with `from` and `to` being collinear. This method is more robust, the less
    /// `from` and `to` are collinear, regardless of `angle`.
    ///
    /// See also [`Self::from_arc`] aligning `from` with `to` over their minor angle.
    ///
    /// See also [`OMatrix::from_arc_angle`](`crate::OMatrix::from_arc_angle`) for owned matrices
    /// generic over storage.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation, Unit, Vector6};
    /// let from = Unit::new_normalize(Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    /// let to = Unit::new_normalize(Vector6::new(3.0, 1.0, 2.0, 5.0, 9.0, 4.0));
    ///
    /// // Rotates by signed angle where `from` and `to` define its orientation.
    /// let angle = 70f64.to_radians();
    /// let rot = Rotation::from_arc_angle(&from, &to, angle).unwrap();
    /// assert_relative_eq!((rot * from).angle(&from), angle, epsilon = 1.0e-6);
    /// assert_relative_eq!((rot.inverse() * to).angle(&to), angle, epsilon = 1.0e-6);
    /// let inv = Rotation::from_arc_angle(&from, &to, -angle).unwrap();
    /// assert_relative_eq!(rot.inverse(), inv, epsilon = 1.0e-6);
    ///
    /// // Returns `None` with `from` and `to` being collinear.
    /// assert!(Rotation::from_arc_angle(&from, &from, angle).is_none());
    /// assert!(Rotation::from_arc_angle(&from, &-from, angle).is_none());
    /// ```
    #[must_use]
    #[inline]
    pub fn from_arc_angle<SB, SC>(
        from: &Unit<Vector<T, Const<D>, SB>>,
        to: &Unit<Vector<T, Const<D>, SC>>,
        angle: T,
    ) -> Option<Self>
    where
        T: RealField,
        SB: Storage<T, Const<D>> + Clone,
        SC: Storage<T, Const<D>> + Clone,
    {
        SMatrix::from_arc_angle(from, to, angle).map(Self::from_matrix_unchecked)
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
    T: Scalar + Zero + One + ClosedAddAssign + ClosedMulAssign,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
