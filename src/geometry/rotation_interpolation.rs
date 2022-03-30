use crate::{Allocator, ArrayStorage, Const, DefaultAllocator, DimDiff, DimSub, Storage, U1};
use crate::{
    RealField, Rotation, Rotation2, Rotation3, SimdRealField, UnitComplex, UnitQuaternion,
};

/// # Interpolation
impl<T: SimdRealField> Rotation2<T> {
    /// Spherical linear interpolation between two rotation matrices.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::geometry::Rotation2;
    ///
    /// let rot1 = Rotation2::new(std::f32::consts::FRAC_PI_4);
    /// let rot2 = Rotation2::new(-std::f32::consts::PI);
    ///
    /// let rot = rot1.slerp(&rot2, 1.0 / 3.0);
    ///
    /// assert_relative_eq!(rot.angle(), std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    #[must_use]
    fn slerp_2d(&self, other: &Self, t: T) -> Self
    where
        T::Element: SimdRealField,
    {
        let c1 = UnitComplex::from(self.clone());
        let c2 = UnitComplex::from(other.clone());
        c1.slerp(&c2, t).into()
    }
}

impl<T: SimdRealField> Rotation3<T> {
    /// Spherical linear interpolation between two rotation matrices.
    ///
    /// Panics if the angle between both rotations is 180 degrees (in which case the interpolation
    /// is not well-defined). Use `.try_slerp` instead to avoid the panic.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::geometry::Rotation3;
    ///
    /// let q1 = Rotation3::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0);
    /// let q2 = Rotation3::from_euler_angles(-std::f32::consts::PI, 0.0, 0.0);
    ///
    /// let q = q1.slerp(&q2, 1.0 / 3.0);
    ///
    /// assert_eq!(q.euler_angles(), (std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    /// ```
    #[inline]
    #[must_use]
    fn slerp_3d(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.slerp(&q2, t).into()
    }

    /// Computes the spherical linear interpolation between two rotation matrices or returns `None`
    /// if both rotations are approximately 180 degrees apart (in which case the interpolation is
    /// not well-defined).
    ///
    /// # Arguments
    /// * `self`: the first rotation to interpolate from.
    /// * `other`: the second rotation to interpolate toward.
    /// * `t`: the interpolation parameter. Should be between 0 and 1.
    /// * `epsilon`: the value below which the sinus of the angle separating both rotations
    /// must be to return `None`.
    #[inline]
    #[must_use]
    pub fn try_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.try_slerp(&q2, t, epsilon).map(|q| q.into())
    }
}

impl<T: RealField, const D: usize> Rotation<T, D>
where
    Const<D>: DimSub<U1>,
    ArrayStorage<T, D, D>: Storage<T, Const<D>, Const<D>>,
    DefaultAllocator: Allocator<T, Const<D>, Const<D>, Buffer = ArrayStorage<T, D, D>>
        + Allocator<T, Const<D>>
        + Allocator<T, Const<D>, DimDiff<Const<D>, U1>>
        + Allocator<T, DimDiff<Const<D>, U1>>,
{
    ///
    /// Computes the spherical linear interpolation between two general rotations.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::geometry::Rotation3;
    ///
    /// let q1 = Rotation3::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0);
    /// let q2 = Rotation3::from_euler_angles(-std::f32::consts::PI, 0.0, 0.0);
    ///
    /// let q = q1.slerp(&q2, 1.0 / 3.0);
    ///
    /// assert_eq!(q.euler_angles(), (std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    /// ```
    ///
    //FIXME: merging slerp for Rotation2 and Rotation3 into this raises the trait bounds
    //from SimdRealField to RealField
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        use std::mem::transmute;

        //The best option here would be to use #[feature(specialization)], but until
        //that's stabilized, this is the best we can do. Theoretically, the compiler should
        //pretty thoroughly optimize away all the excess checks and conversions
        match D {
            0 => self.clone(),

            //FIXME: this doesn't really work in 1D since we can't interp between -1 and 1
            1 => self.clone(),

            //NOTE: Not pretty, but without refactoring the API, this is the best we can do
            //NOTE: This is safe because we directly check the dimension first
            2 => unsafe {
                let (self2d, other2d) = (
                    transmute::<&Self, &Rotation2<T>>(self),
                    transmute::<&Self, &Rotation2<T>>(other),
                );
                transmute::<&Rotation2<T>, &Self>(&self2d.slerp_2d(other2d, t)).clone()
            },
            3 => unsafe {
                let (self3d, other3d) = (
                    transmute::<&Self, &Rotation3<T>>(self),
                    transmute::<&Self, &Rotation3<T>>(other),
                );
                transmute::<&Rotation3<T>, &Self>(&self3d.slerp_3d(other3d, t)).clone()
            },

            //the multiplication order matters here
            _ => (other / self).powf(t) * self,
        }
    }
}
