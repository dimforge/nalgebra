use crate::{
    DualQuaternion, Isometry3, Quaternion, Scalar, SimdRealField, Translation3, UnitDualQuaternion,
    UnitQuaternion,
};
use num::{One, Zero};
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
use simba::scalar::SupersetOf;

impl<N: Scalar> DualQuaternion<N> {
    /// Creates a dual quaternion from its rotation and translation components.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let rot = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let trans = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    ///
    /// let dq = DualQuaternion::from_real_and_dual(rot, trans);
    /// assert_eq!(dq.real.w, 1.0);
    /// ```
    #[inline]
    pub fn from_real_and_dual(real: Quaternion<N>, dual: Quaternion<N>) -> Self {
        Self { real, dual }
    }

    /// The dual quaternion multiplicative identity.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    ///
    /// let dq1 = DualQuaternion::identity();
    /// let dq2 = DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.,2.,3.,4.),
    ///     Quaternion::new(5.,6.,7.,8.)
    /// );
    ///
    /// assert_eq!(dq1 * dq2, dq2);
    /// assert_eq!(dq2 * dq1, dq2);
    /// ```
    #[inline]
    pub fn identity() -> Self
    where
        N: SimdRealField,
    {
        Self::from_real_and_dual(
            Quaternion::from_real(N::one()),
            Quaternion::from_real(N::zero()),
        )
    }

    /// Cast the components of `self` to another type.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, DualQuaternion};
    /// let q = DualQuaternion::from_real(Quaternion::new(1.0f64, 2.0, 3.0, 4.0));
    /// let q2 = q.cast::<f32>();
    /// assert_eq!(q2, DualQuaternion::from_real(Quaternion::new(1.0f32, 2.0, 3.0, 4.0)));
    /// ```
    pub fn cast<To: Scalar>(self) -> DualQuaternion<To>
    where
        DualQuaternion<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<N: SimdRealField> DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    /// Creates a dual quaternion from only its real part, with no translation
    /// component.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let rot = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// let dq = DualQuaternion::from_real(rot);
    /// assert_eq!(dq.real.w, 1.0);
    /// assert_eq!(dq.dual.w, 0.0);
    /// ```
    #[inline]
    pub fn from_real(real: Quaternion<N>) -> Self {
        Self {
            real,
            dual: Quaternion::zero(),
        }
    }
}

impl<N: SimdRealField> One for DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: SimdRealField> Zero for DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn zero() -> Self {
        DualQuaternion::from_real_and_dual(Quaternion::zero(), Quaternion::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

#[cfg(feature = "arbitrary")]
impl<N> Arbitrary for DualQuaternion<N>
where
    N: SimdRealField + Arbitrary + Send,
    N::Element: SimdRealField,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        Self::from_real_and_dual(Arbitrary::arbitrary(rng), Arbitrary::arbitrary(rng))
    }
}

impl<N: SimdRealField> UnitDualQuaternion<N> {
    /// The unit dual quaternion multiplicative identity, which also represents
    /// the identity transformation as an isometry.
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let ident = UnitDualQuaternion::identity();
    /// let point = Point3::new(1.0, -4.3, 3.33);
    ///
    /// assert_eq!(ident * point, point);
    /// assert_eq!(ident, ident.inverse());
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(DualQuaternion::identity())
    }

    /// Cast the components of `self` to another type.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitDualQuaternion;
    /// let q = UnitDualQuaternion::<f64>::identity();
    /// let q2 = q.cast::<f32>();
    /// assert_eq!(q2, UnitDualQuaternion::<f32>::identity());
    /// ```
    pub fn cast<To: Scalar>(self) -> UnitDualQuaternion<To>
    where
        UnitDualQuaternion<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<N: SimdRealField> UnitDualQuaternion<N>
where
    N::Element: SimdRealField,
{
    /// Return a dual quaternion representing the translation and orientation
    /// given by the provided rotation quaternion and translation vector.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// assert_relative_eq!(dq * point, Point3::new(1.0, 0.0, 2.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_parts(translation: Translation3<N>, rotation: UnitQuaternion<N>) -> Self {
        let half: N = crate::convert(0.5f64);
        UnitDualQuaternion::new_unchecked(DualQuaternion {
            real: rotation.clone().into_inner(),
            dual: Quaternion::from_parts(N::zero(), translation.vector)
                * rotation.clone().into_inner()
                * half,
        })
    }

    /// Return a unit dual quaternion representing the translation and orientation
    /// given by the provided isometry.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let iso = Isometry3::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let dq = UnitDualQuaternion::from_isometry(&iso);
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// assert_relative_eq!(dq * point, iso * point, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_isometry(isometry: &Isometry3<N>) -> Self {
        UnitDualQuaternion::from_parts(isometry.translation, isometry.rotation)
    }

    /// Creates a dual quaternion from a unit quaternion rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, UnitDualQuaternion, Quaternion};
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let rot = UnitQuaternion::new_normalize(q);
    ///
    /// let dq = UnitDualQuaternion::from_rotation(rot);
    /// assert_relative_eq!(dq.as_ref().real.norm(), 1.0, epsilon = 1.0e-6);
    /// assert_eq!(dq.as_ref().dual.norm(), 0.0);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: UnitQuaternion<N>) -> Self {
        Self::new_unchecked(DualQuaternion::from_real(rotation.into_inner()))
    }
}

impl<N: SimdRealField> One for UnitDualQuaternion<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "arbitrary")]
impl<N> Arbitrary for UnitDualQuaternion<N>
where
    N: SimdRealField + Arbitrary + Send,
    N::Element: SimdRealField,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        Self::new_normalize(Arbitrary::arbitrary(rng))
    }
}
