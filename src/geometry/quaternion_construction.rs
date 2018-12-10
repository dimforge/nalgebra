#[cfg(feature = "arbitrary")]
use base::dimension::U4;
#[cfg(feature = "arbitrary")]
use base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{One, Zero};
use rand::distributions::{Distribution, OpenClosed01, Standard};
use rand::Rng;

use alga::general::Real;

use base::dimension::U3;
use base::storage::Storage;
#[cfg(feature = "arbitrary")]
use base::Vector3;
use base::{Unit, Vector, Vector4};

use geometry::{Quaternion, Rotation, UnitQuaternion};

impl<N: Real> Quaternion<N> {
    /// Creates a quaternion from a 4D vector. The quaternion scalar part corresponds to the `w`
    /// vector component.
    #[inline]
    #[deprecated(note = "Use `::from` instead.")]
    pub fn from_vector(vector: Vector4<N>) -> Self {
        Quaternion { coords: vector }
    }

    /// Creates a new quaternion from its individual components. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is `[ i, j, k, w ]` while the arguments for this functions are in the
    /// order `(w, i, j, k)`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector4};
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub fn new(w: N, i: N, j: N, k: N) -> Self {
        let v = Vector4::<N>::new(i, j, k, w);
        Self::from(v)
    }

    /// Creates a new quaternion from its scalar and vector parts. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is [ vector, scalar ].
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector3, Vector4};
    /// let w = 1.0;
    /// let ijk = Vector3::new(2.0, 3.0, 4.0);
    /// let q = Quaternion::from_parts(w, ijk);
    /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    // FIXME: take a reference to `vector`?
    pub fn from_parts<SB>(scalar: N, vector: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        Self::new(scalar, vector[0], vector[1], vector[2])
    }

    /// Creates a new quaternion from its polar decomposition.
    ///
    /// Note that `axis` is assumed to be a unit vector.
    // FIXME: take a reference to `axis`?
    pub fn from_polar_decomposition<SB>(scale: N, theta: N, axis: Unit<Vector<N, U3, SB>>) -> Self
    where SB: Storage<N, U3> {
        let rot = UnitQuaternion::<N>::from_axis_angle(&axis, theta * ::convert(2.0f64));

        rot.into_inner() * scale
    }

    /// The quaternion multiplicative identity.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::identity();
    /// let q2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// assert_eq!(q * q2, q2);
    /// assert_eq!(q2 * q, q2);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new(N::one(), N::zero(), N::zero(), N::zero())
    }
}

impl<N: Real> One for Quaternion<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real> Zero for Quaternion<N> {
    #[inline]
    fn zero() -> Self {
        Self::new(N::zero(), N::zero(), N::zero(), N::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coords.is_zero()
    }
}

impl<N: Real> Distribution<Quaternion<N>> for Standard
where Standard: Distribution<N>
{
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> Quaternion<N> {
        Quaternion::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for Quaternion<N>
where Owned<N, U4>: Send
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Quaternion::new(
            N::arbitrary(g),
            N::arbitrary(g),
            N::arbitrary(g),
            N::arbitrary(g),
        )
    }
}

impl<N: Real> UnitQuaternion<N> {
    /// The rotation identity.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Point3};
    /// let q = UnitQuaternion::identity();
    /// let q2 = UnitQuaternion::new(Vector3::new(1.0, 2.0, 3.0));
    /// let v = Vector3::new_random();
    /// let p = Point3::from(v);
    ///
    /// assert_eq!(q * q2, q2);
    /// assert_eq!(q2 * q, q2);
    /// assert_eq!(q * v, v);
    /// assert_eq!(q * p, p);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Quaternion::identity())
    }

    /// Creates a new quaternion from a unit vector (the rotation axis) and an angle
    /// (the rotation angle).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axis = Vector3::y_axis();
    /// let angle = f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_axis_angle(&axis, angle);
    ///
    /// assert_eq!(q.axis().unwrap(), axis);
    /// assert_eq!(q.angle(), angle);
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_axis_angle<SB>(axis: &Unit<Vector<N, U3, SB>>, angle: N) -> Self
    where SB: Storage<N, U3> {
        let (sang, cang) = (angle / ::convert(2.0f64)).sin_cos();

        let q = Quaternion::from_parts(cang, axis.as_ref() * sang);
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion from a quaternion.
    ///
    /// The input quaternion will be normalized.
    #[inline]
    pub fn from_quaternion(q: Quaternion<N>) -> Self {
        Self::new_normalize(q)
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::UnitQuaternion;
    /// let rot = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let euler = rot.euler_angles();
    /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self {
        let (sr, cr) = (roll * ::convert(0.5f64)).sin_cos();
        let (sp, cp) = (pitch * ::convert(0.5f64)).sin_cos();
        let (sy, cy) = (yaw * ::convert(0.5f64)).sin_cos();

        let q = Quaternion::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        );

        Self::new_unchecked(q)
    }

    /// Builds an unit quaternion from a rotation matrix.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Rotation3, UnitQuaternion, Vector3};
    /// let axis = Vector3::y_axis();
    /// let angle = 0.1;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// let q = UnitQuaternion::from_rotation_matrix(&rot);
    /// assert_relative_eq!(q.to_rotation_matrix(), rot, epsilon = 1.0e-6);
    /// assert_relative_eq!(q.axis().unwrap(), rot.axis().unwrap(), epsilon = 1.0e-6);
    /// assert_relative_eq!(q.angle(), rot.angle(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_rotation_matrix(rotmat: &Rotation<N, U3>) -> Self {
        // Robust matrix to quaternion transformation.
        // See http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion
        let tr = rotmat[(0, 0)] + rotmat[(1, 1)] + rotmat[(2, 2)];
        let res;

        let _0_25: N = ::convert(0.25);

        if tr > N::zero() {
            let denom = (tr + N::one()).sqrt() * ::convert(2.0);
            res = Quaternion::new(
                _0_25 * denom,
                (rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                (rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                (rotmat[(1, 0)] - rotmat[(0, 1)]) / denom,
            );
        } else if rotmat[(0, 0)] > rotmat[(1, 1)] && rotmat[(0, 0)] > rotmat[(2, 2)] {
            let denom = (N::one() + rotmat[(0, 0)] - rotmat[(1, 1)] - rotmat[(2, 2)]).sqrt()
                * ::convert(2.0);
            res = Quaternion::new(
                (rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                _0_25 * denom,
                (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom,
            );
        } else if rotmat[(1, 1)] > rotmat[(2, 2)] {
            let denom = (N::one() + rotmat[(1, 1)] - rotmat[(0, 0)] - rotmat[(2, 2)]).sqrt()
                * ::convert(2.0);
            res = Quaternion::new(
                (rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                _0_25 * denom,
                (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom,
            );
        } else {
            let denom = (N::one() + rotmat[(2, 2)] - rotmat[(0, 0)] - rotmat[(1, 1)]).sqrt()
                * ::convert(2.0);
            res = Quaternion::new(
                (rotmat[(1, 0)] - rotmat[(0, 1)]) / denom,
                (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom,
                (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom,
                _0_25 * denom,
            );
        }

        Self::new_unchecked(res)
    }

    /// The unit quaternion needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Vector3, UnitQuaternion};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let q = UnitQuaternion::rotation_between(&a, &b).unwrap();
    /// assert_relative_eq!(q * a, b);
    /// assert_relative_eq!(q.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U3, SB>, b: &Vector<N, U3, SC>) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Vector3, UnitQuaternion};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let q2 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// let q5 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// assert_relative_eq!(q2 * q2 * q2 * q2 * q2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(q5 * q5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<N, U3, SB>,
        b: &Vector<N, U3, SC>,
        s: N,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        // FIXME: code duplication with Rotation.
        if let (Some(na), Some(nb)) = (
            Unit::try_new(a.clone_owned(), N::zero()),
            Unit::try_new(b.clone_owned(), N::zero()),
        ) {
            Self::scaled_rotation_between_axis(&na, &nb, s)
        } else {
            Some(Self::identity())
        }
    }

    /// The unit quaternion needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Unit, Vector3, UnitQuaternion};
    /// let a = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let b = Unit::new_normalize(Vector3::new(3.0, 1.0, 2.0));
    /// let q = UnitQuaternion::rotation_between(&a, &b).unwrap();
    /// assert_relative_eq!(q * a, b);
    /// assert_relative_eq!(q.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between_axis<SB, SC>(
        a: &Unit<Vector<N, U3, SB>>,
        b: &Unit<Vector<N, U3, SC>>,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::scaled_rotation_between_axis(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Unit, Vector3, UnitQuaternion};
    /// let a = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let b = Unit::new_normalize(Vector3::new(3.0, 1.0, 2.0));
    /// let q2 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// let q5 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// assert_relative_eq!(q2 * q2 * q2 * q2 * q2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(q5 * q5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between_axis<SB, SC>(
        na: &Unit<Vector<N, U3, SB>>,
        nb: &Unit<Vector<N, U3, SC>>,
        s: N,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        // FIXME: code duplication with Rotation.
        let c = na.cross(&nb);

        if let Some(axis) = Unit::try_new(c, N::default_epsilon()) {
            let cos = na.dot(&nb);

            // The cosinus may be out of [-1, 1] because of inaccuracies.
            if cos <= -N::one() {
                return None;
            } else if cos >= N::one() {
                return Some(Self::identity());
            } else {
                return Some(Self::from_axis_angle(&axis, cos.acos() * s));
            }
        } else if na.dot(&nb) < N::zero() {
            // PI
            //
            // The rotation axis is undefined but the angle not zero. This is not a
            // simple rotation.
            return None;
        } else {
            // Zero
            Some(Self::identity())
        }
    }

    /// Creates an unit quaternion that corresponds to the local frame of an observer standing at the
    /// origin and looking toward `dir`.
    ///
    /// It maps the `z` axis to the direction `dir`.
    ///
    /// # Arguments
    ///   * dir - The look direction. It does not need to be normalized.
    ///   * up - The vertical direction. It does not need to be normalized.
    ///   The only requirement of this parameter is to not be collinear to `dir`. Non-collinearity
    ///   is not checked.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::new_observer_frame(&dir, &up);
    /// assert_relative_eq!(q * Vector3::z(), dir.normalize());
    /// ```
    #[inline]
    pub fn new_observer_frame<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::from_rotation_matrix(&Rotation::<N, U3>::new_observer_frame(dir, up))
    }

    /// Builds a right-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **negative** `z` axis.
    /// This conforms to the common notion of right handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir − The view direction. It does not need to be normalized.
    ///   * up - A vector approximately aligned with required the vertical axis. It does not need
    ///   to be normalized. The only requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::look_at_rh(&dir, &up);
    /// assert_relative_eq!(q * dir.normalize(), -Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_rh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::new_observer_frame(&-dir, up).inverse()
    }

    /// Builds a left-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **positive** `z` axis.
    /// This conforms to the common notion of left handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir − The view direction. It does not need to be normalized.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::look_at_lh(&dir, &up);
    /// assert_relative_eq!(q * dir.normalize(), Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_lh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::new_observer_frame(dir, up).inverse()
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `N::default_epsilon()`, this returns the identity rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::new(axisangle);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::new(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn new<SB>(axisangle: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        let two: N = ::convert(2.0f64);
        let q = Quaternion::<N>::from_parts(N::zero(), axisangle / two).exp();
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `eps`, this returns the identity rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::new_eps(axisangle, 1.0e-6);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // An almost zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::new_eps(Vector3::new(1.0e-8, 1.0e-9, 1.0e-7), 1.0e-6), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn new_eps<SB>(axisangle: Vector<N, U3, SB>, eps: N) -> Self
    where SB: Storage<N, U3> {
        let two: N = ::convert(2.0f64);
        let q = Quaternion::<N>::from_parts(N::zero(), axisangle / two).exp_eps(eps);
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `N::default_epsilon()`, this returns the identity rotation.
    /// Same as `Self::new(axisangle)`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_scaled_axis(axisangle);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_scaled_axis<SB>(axisangle: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        Self::new(axisangle)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `eps`, this returns the identity rotation.
    /// Same as `Self::new_eps(axisangle, eps)`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_scaled_axis_eps(axisangle, 1.0e-6);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // An almost zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis_eps(Vector3::new(1.0e-8, 1.0e-9, 1.0e-7), 1.0e-6), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_scaled_axis_eps<SB>(axisangle: Vector<N, U3, SB>, eps: N) -> Self
    where SB: Storage<N, U3> {
        Self::new_eps(axisangle, eps)
    }
}

impl<N: Real> One for UnitQuaternion<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real> Distribution<UnitQuaternion<N>> for Standard
where OpenClosed01: Distribution<N>
{
    /// Generate a uniformly distributed random rotation quaternion.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> UnitQuaternion<N> {
        // Ken Shoemake's Subgroup Algorithm
        // Uniform random rotations.
        // In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
        let x0 = rng.sample(OpenClosed01);
        let x1 = rng.sample(OpenClosed01);
        let x2 = rng.sample(OpenClosed01);
        let theta1 = N::two_pi() * x1;
        let theta2 = N::two_pi() * x2;
        let s1 = theta1.sin();
        let c1 = theta1.cos();
        let s2 = theta2.sin();
        let c2 = theta2.cos();
        let r1 = (N::one() - x0).sqrt();
        let r2 = x0.sqrt();
        Unit::new_unchecked(Quaternion::new(s1 * r1, c1 * r1, s2 * r2, c2 * r2))
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for UnitQuaternion<N>
where
    Owned<N, U4>: Send,
    Owned<N, U3>: Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let axisangle = Vector3::arbitrary(g);
        UnitQuaternion::from_scaled_axis(axisangle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{self, SeedableRng};

    #[test]
    fn random_unit_quats_are_unit() {
        let mut rng = rand::prng::XorShiftRng::from_seed([0xAB; 16]);
        for _ in 0..1000 {
            let x = rng.gen::<UnitQuaternion<f32>>();
            assert!(relative_eq!(x.into_inner().norm(), 1.0))
        }
    }
}
