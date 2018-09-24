use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::Zero;
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use base::storage::Owned;
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::Real;

use base::dimension::{U1, U3, U4};
use base::storage::{CStride, RStride};
use base::{Matrix3, MatrixN, MatrixSlice, MatrixSliceMut, Unit, Vector3, Vector4};

use geometry::Rotation;

/// A quaternion. See the type alias `UnitQuaternion = Unit<Quaternion>` for a quaternion
/// that may be used as a rotation.
#[repr(C)]
#[derive(Debug)]
pub struct Quaternion<N: Real> {
    /// This quaternion as a 4D vector of coordinates in the `[ x, y, z, w ]` storage order.
    pub coords: Vector4<N>,
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Real> Abomonation for Quaternion<N>
where
    Vector4<N>: Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.coords.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.coords.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.coords.exhume(bytes)
    }
}

impl<N: Real + Eq> Eq for Quaternion<N> {}

impl<N: Real> PartialEq for Quaternion<N> {
    fn eq(&self, rhs: &Self) -> bool {
        self.coords == rhs.coords ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(rhs.as_vector().iter()).all(|(a, b)| *a == -*b)
    }
}

impl<N: Real + hash::Hash> hash::Hash for Quaternion<N> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<N: Real> Copy for Quaternion<N> {}

impl<N: Real> Clone for Quaternion<N> {
    #[inline]
    fn clone(&self) -> Self {
        Quaternion::from_vector(self.coords.clone())
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Real> Serialize for Quaternion<N>
where
    Owned<N, U4>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Real> Deserialize<'a> for Quaternion<N>
where
    Owned<N, U4>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = Vector4::<N>::deserialize(deserializer)?;

        Ok(Quaternion::from_vector(coords))
    }
}

impl<N: Real> Quaternion<N> {
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn into_owned(self) -> Quaternion<N> {
        self
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn clone_owned(&self) -> Quaternion<N> {
        Quaternion::from_vector(self.coords.clone_owned())
    }

    /// Normalizes this quaternion.
    #[inline]
    pub fn normalize(&self) -> Quaternion<N> {
        Quaternion::from_vector(self.coords.normalize())
    }

    /// Compute the conjugate of this quaternion.
    #[inline]
    pub fn conjugate(&self) -> Quaternion<N> {
        let v = Vector4::new(
            -self.coords[0],
            -self.coords[1],
            -self.coords[2],
            self.coords[3],
        );
        Quaternion::from_vector(v)
    }

    /// Inverts this quaternion if it is not zero.
    #[inline]
    pub fn try_inverse(&self) -> Option<Quaternion<N>> {
        let mut res = Quaternion::from_vector(self.coords.clone_owned());

        if res.try_inverse_mut() {
            Some(res)
        } else {
            None
        }
    }

    /// Linear interpolation between two quaternion.
    #[inline]
    pub fn lerp(&self, other: &Quaternion<N>, t: N) -> Quaternion<N> {
        self * (N::one() - t) + other * t
    }

    /// The vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector(&self) -> MatrixSlice<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
        self.coords.fixed_rows::<U3>(0)
    }

    /// The scalar part `w` of this quaternion.
    #[inline]
    pub fn scalar(&self) -> N {
        self.coords[3]
    }

    /// Reinterprets this quaternion as a 4D vector.
    #[inline]
    pub fn as_vector(&self) -> &Vector4<N> {
        &self.coords
    }

    /// The norm of this quaternion.
    #[inline]
    pub fn norm(&self) -> N {
        self.coords.norm()
    }

    /// A synonym for the norm of this quaternion.
    ///
    /// Aka the length.
    ///
    /// This function is simply implemented as a call to `norm()`
    #[inline]
    pub fn magnitude(&self) -> N {
        self.norm()
    }

    /// A synonym for the squared norm of this quaternion.
    ///
    /// Aka the squared length.
    ///
    /// This function is simply implemented as a call to `norm_squared()`
    #[inline]
    pub fn magnitude_squared(&self) -> N {
        self.norm_squared()
    }

    /// The squared norm of this quaternion.
    #[inline]
    pub fn norm_squared(&self) -> N {
        self.coords.norm_squared()
    }

    /// The dot product of two quaternions.
    #[inline]
    pub fn dot(&self, rhs: &Self) -> N {
        self.coords.dot(&rhs.coords)
    }

    /// The polar decomposition of this quaternion.
    ///
    /// Returns, from left to right: the quaternion norm, the half rotation angle, the rotation
    /// axis. If the rotation angle is zero, the rotation axis is set to `None`.
    pub fn polar_decomposition(&self) -> (N, N, Option<Unit<Vector3<N>>>) {
        if let Some((q, n)) = Unit::try_new_and_get(*self, N::zero()) {
            if let Some(axis) = Unit::try_new(self.vector().clone_owned(), N::zero()) {
                let angle = q.angle() / ::convert(2.0f64);

                (n, angle, Some(axis))
            } else {
                (n, N::zero(), None)
            }
        } else {
            (N::zero(), N::zero(), None)
        }
    }

    /// Compute the exponential of a quaternion.
    #[inline]
    pub fn exp(&self) -> Quaternion<N> {
        self.exp_eps(N::default_epsilon())
    }

    /// Compute the exponential of a quaternion.
    #[inline]
    pub fn exp_eps(&self, eps: N) -> Quaternion<N> {
        let v = self.vector();
        let nn = v.norm_squared();

        if nn <= eps * eps {
            Quaternion::identity()
        } else {
            let w_exp = self.scalar().exp();
            let n = nn.sqrt();
            let nv = v * (w_exp * n.sin() / n);

            Quaternion::from_parts(n.cos(), nv)
        }
    }

    /// Compute the natural logarithm of a quaternion.
    #[inline]
    pub fn ln(&self) -> Quaternion<N> {
        let n = self.norm();
        let v = self.vector();
        let s = self.scalar();

        Quaternion::from_parts(n.ln(), v.normalize() * (s / n).acos())
    }

    /// Raise the quaternion to a given floating power.
    #[inline]
    pub fn powf(&self, n: N) -> Quaternion<N> {
        (self.ln() * n).exp()
    }

    /// Transforms this quaternion into its 4D vector form (Vector part, Scalar part).
    #[inline]
    pub fn as_vector_mut(&mut self) -> &mut Vector4<N> {
        &mut self.coords
    }

    /// The mutable vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector_mut(
        &mut self,
    ) -> MatrixSliceMut<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
        self.coords.fixed_rows_mut::<U3>(0)
    }

    /// Replaces this quaternion by its conjugate.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.coords[0] = -self.coords[0];
        self.coords[1] = -self.coords[1];
        self.coords[2] = -self.coords[2];
    }

    /// Inverts this quaternion in-place if it is not zero.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        let norm_squared = self.norm_squared();

        if relative_eq!(&norm_squared, &N::zero()) {
            false
        } else {
            self.conjugate_mut();
            self.coords /= norm_squared;

            true
        }
    }

    /// Normalizes this quaternion.
    #[inline]
    pub fn normalize_mut(&mut self) -> N {
        self.coords.normalize_mut()
    }
}

impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for Quaternion<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_vector().abs_diff_eq(other.as_vector(), epsilon) ||
        // Account for the double-covering of S², i.e. q = -q
       self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.abs_diff_eq(&-*b, epsilon))
    }
}

impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for Quaternion<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_vector().relative_eq(other.as_vector(), epsilon, max_relative) ||
        // Account for the double-covering of S², i.e. q = -q
       self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.relative_eq(&-*b, epsilon, max_relative))
    }
}

impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for Quaternion<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_vector().ulps_eq(other.as_vector(), epsilon, max_ulps) ||
        // Account for the double-covering of S², i.e. q = -q.
       self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.ulps_eq(&-*b, epsilon, max_ulps))
    }
}

impl<N: Real + fmt::Display> fmt::Display for Quaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quaternion {} − ({}, {}, {})",
            self[3], self[0], self[1], self[2]
        )
    }
}

/// A unit quaternions. May be used to represent a rotation.
pub type UnitQuaternion<N> = Unit<Quaternion<N>>;

impl<N: Real> UnitQuaternion<N> {
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn into_owned(self) -> UnitQuaternion<N> {
        self
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn clone_owned(&self) -> UnitQuaternion<N> {
        *self
    }

    /// The rotation angle in [0; pi] of this unit quaternion.
    #[inline]
    pub fn angle(&self) -> N {
        let w = self.quaternion().scalar().abs();

        // Handle inaccuracies that make break `.acos`.
        if w >= N::one() {
            N::zero()
        } else {
            w.acos() * ::convert(2.0f64)
        }
    }

    /// The underlying quaternion.
    ///
    /// Same as `self.as_ref()`.
    #[inline]
    pub fn quaternion(&self) -> &Quaternion<N> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit quaternion.
    #[inline]
    pub fn conjugate(&self) -> UnitQuaternion<N> {
        UnitQuaternion::new_unchecked(self.as_ref().conjugate())
    }

    /// Inverts this quaternion if it is not zero.
    #[inline]
    pub fn inverse(&self) -> UnitQuaternion<N> {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(&self, other: &UnitQuaternion<N>) -> N {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit quaternion needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to(&self, other: &UnitQuaternion<N>) -> UnitQuaternion<N> {
        other / self
    }

    /// Linear interpolation between two unit quaternions.
    ///
    /// The result is not normalized.
    #[inline]
    pub fn lerp(&self, other: &UnitQuaternion<N>, t: N) -> Quaternion<N> {
        self.as_ref().lerp(other.as_ref(), t)
    }

    /// Normalized linear interpolation between two unit quaternions.
    #[inline]
    pub fn nlerp(&self, other: &UnitQuaternion<N>, t: N) -> UnitQuaternion<N> {
        let mut res = self.lerp(other, t);
        let _ = res.normalize_mut();

        UnitQuaternion::new_unchecked(res)
    }

    /// Spherical linear interpolation between two unit quaternions.
    ///
    /// Panics if the angle between both quaternion is 180 degrees (in which case the interpolation
    /// is not well-defined).
    #[inline]
    pub fn slerp(&self, other: &UnitQuaternion<N>, t: N) -> UnitQuaternion<N> {
        Unit::new_unchecked(
            Quaternion::from_vector(Unit::new_unchecked(self.coords).slerp(&Unit::new_unchecked(other.coords), t).unwrap())
        )
    }

    /// Computes the spherical linear interpolation between two unit quaternions or returns `None`
    /// if both quaternions are approximately 180 degrees apart (in which case the interpolation is
    /// not well-defined).
    ///
    /// # Arguments
    /// * `self`: the first quaternion to interpolate from.
    /// * `other`: the second quaternion to interpolate toward.
    /// * `t`: the interpolation parameter. Should be between 0 and 1.
    /// * `epsilon`: the value below which the sinus of the angle separating both quaternion
    /// must be to return `None`.
    #[inline]
    pub fn try_slerp(
        &self,
        other: &UnitQuaternion<N>,
        t: N,
        epsilon: N,
    ) -> Option<UnitQuaternion<N>> {
        Unit::new_unchecked(self.coords).try_slerp(&Unit::new_unchecked(other.coords), t, epsilon)
            .map(|q| Unit::new_unchecked(Quaternion::from_vector(q.unwrap())))
    }

    /// Compute the conjugate of this unit quaternion in-place.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.as_mut_unchecked().conjugate_mut()
    }

    /// Inverts this quaternion if it is not zero.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.as_mut_unchecked().conjugate_mut()
    }

    /// The rotation axis of this unit quaternion or `None` if the rotation is zero.
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<N>>> {
        let v = if self.quaternion().scalar() >= N::zero() {
            self.as_ref().vector().clone_owned()
        } else {
            -self.as_ref().vector()
        };

        Unit::try_new(v, N::zero())
    }

    /// The rotation axis of this unit quaternion multiplied by the rotation angle.
    #[inline]
    pub fn scaled_axis(&self) -> Vector3<N> {
        if let Some(axis) = self.axis() {
            axis.unwrap() * self.angle()
        } else {
            Vector3::zero()
        }
    }

    /// The rotation axis and angle in ]0, pi] of this unit quaternion.
    ///
    /// Returns `None` if the angle is zero.
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<N>>, N)> {
        if let Some(axis) = self.axis() {
            Some((axis, self.angle()))
        } else {
            None
        }
    }

    /// Compute the exponential of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it looses the unit property.
    #[inline]
    pub fn exp(&self) -> Quaternion<N> {
        self.as_ref().exp()
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it looses the unit property.
    /// The vector part of the return value corresponds to the axis-angle representation (divided
    /// by 2.0) of this unit quaternion.
    #[inline]
    pub fn ln(&self) -> Quaternion<N> {
        if let Some(v) = self.axis() {
            Quaternion::from_parts(N::zero(), v.unwrap() * self.angle())
        } else {
            Quaternion::zero()
        }
    }

    /// Raise the quaternion to a given floating power.
    ///
    /// This returns the unit quaternion that identifies a rotation with axis `self.axis()` and
    /// angle `self.angle() × n`.
    #[inline]
    pub fn powf(&self, n: N) -> UnitQuaternion<N> {
        if let Some(v) = self.axis() {
            UnitQuaternion::from_axis_angle(&v, self.angle() * n)
        } else {
            UnitQuaternion::identity()
        }
    }

    /// Builds a rotation matrix from this unit quaternion.
    #[inline]
    pub fn to_rotation_matrix(&self) -> Rotation<N, U3> {
        let i = self.as_ref()[0];
        let j = self.as_ref()[1];
        let k = self.as_ref()[2];
        let w = self.as_ref()[3];

        let ww = w * w;
        let ii = i * i;
        let jj = j * j;
        let kk = k * k;
        let ij = i * j * ::convert(2.0f64);
        let wk = w * k * ::convert(2.0f64);
        let wj = w * j * ::convert(2.0f64);
        let ik = i * k * ::convert(2.0f64);
        let jk = j * k * ::convert(2.0f64);
        let wi = w * i * ::convert(2.0f64);

        Rotation::from_matrix_unchecked(Matrix3::new(
            ww + ii - jj - kk,
            ij - wk,
            wj + ik,
            wk + ij,
            ww - ii + jj - kk,
            jk - wi,
            ik - wj,
            wi + jk,
            ww - ii - jj + kk,
        ))
    }

    /// Converts this unit quaternion into its equivalent Euler angles.
    ///
    /// The angles are produced in the form (roll, yaw, pitch).
    #[inline]
    pub fn to_euler_angles(&self) -> (N, N, N) {
        self.to_rotation_matrix().to_euler_angles()
    }

    /// Converts this unit quaternion into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, U4> {
        self.to_rotation_matrix().to_homogeneous()
    }
}

impl<N: Real + fmt::Display> fmt::Display for UnitQuaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(axis) = self.axis() {
            let axis = axis.unwrap();
            write!(
                f,
                "UnitQuaternion angle: {} − axis: ({}, {}, {})",
                self.angle(),
                axis[0],
                axis[1],
                axis[2]
            )
        } else {
            write!(
                f,
                "UnitQuaternion angle: {} − axis: (undefined)",
                self.angle()
            )
        }
    }
}

impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for UnitQuaternion<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for UnitQuaternion<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for UnitQuaternion<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
