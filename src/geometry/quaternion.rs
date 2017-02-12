use std::fmt;
use num::Zero;
use approx::ApproxEq;

use alga::general::Real;

use core::{Unit, ColumnVector, OwnedColumnVector, MatrixSlice, MatrixSliceMut, SquareMatrix,
           OwnedSquareMatrix};
use core::storage::{Storage, StorageMut};
use core::allocator::Allocator;
use core::dimension::{U1, U3, U4};

use geometry::{RotationBase, OwnedRotation};

/// A quaternion with an owned storage allocated by `A`.
pub type OwnedQuaternionBase<N, A> = QuaternionBase<N, <A as Allocator<N, U4, U1>>::Buffer>;

/// A unit quaternion with an owned storage allocated by `A`.
pub type OwnedUnitQuaternionBase<N, A> = UnitQuaternionBase<N, <A as Allocator<N, U4, U1>>::Buffer>;

/// A quaternion. See the type alias `UnitQuaternionBase = Unit<QuaternionBase>` for a quaternion
/// that may be used as a rotation.
#[repr(C)]
#[derive(Hash, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct QuaternionBase<N: Real, S: Storage<N, U4, U1>> {
    /// This quaternion as a 4D vector of coordinates in the `[ x, y, z, w ]` storage order.
    pub coords: ColumnVector<N, U4, S>
}

impl<N, S> Eq for QuaternionBase<N, S>
    where N: Real + Eq,
          S: Storage<N, U4, U1> {
}

impl<N, S> PartialEq for QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    fn eq(&self, rhs: &Self) -> bool {
        self.coords == rhs.coords ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(rhs.as_vector().iter()).all(|(a, b)| *a == -*b)
    }
}

impl<N, S> QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    /// Moves this quaternion into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OwnedQuaternionBase<N, S::Alloc> {
        QuaternionBase::from_vector(self.coords.into_owned())
    }

    /// Clones this quaternion into one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        QuaternionBase::from_vector(self.coords.clone_owned())
    }

    /// The vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector(&self) -> MatrixSlice<N, U3, U1, S::RStride, S::CStride, S::Alloc> {
        self.coords.fixed_rows::<U3>(0)
    }

    /// The scalar part `w` of this quaternion.
    #[inline]
    pub fn scalar(&self) -> N {
        self.coords[3]
    }

    /// Reinterprets this quaternion as a 4D vector.
    #[inline]
    pub fn as_vector(&self) -> &ColumnVector<N, U4, S> {
        &self.coords
    }

    /// The norm of this quaternion.
    #[inline]
    pub fn norm(&self) -> N {
        self.coords.norm()
    }

    /// The squared norm of this quaternion.
    #[inline]
    pub fn norm_squared(&self) -> N {
        self.coords.norm_squared()
    }

    /// Normalizes this quaternion.
    #[inline]
    pub fn normalize(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        QuaternionBase::from_vector(self.coords.normalize())
    }

    /// Compute the conjugate of this quaternion.
    #[inline]
    pub fn conjugate(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        let v = OwnedColumnVector::<N, U4, S::Alloc>::new(-self.coords[0],
                                                          -self.coords[1],
                                                          -self.coords[2],
                                                          self.coords[3]);
        QuaternionBase::from_vector(v)
    }

    /// Inverts this quaternion if it is not zero.
    #[inline]
    pub fn try_inverse(&self) -> Option<OwnedQuaternionBase<N, S::Alloc>> {
        let mut res = QuaternionBase::from_vector(self.coords.clone_owned());

        if res.try_inverse_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    /// Linear interpolation between two quaternion.
    #[inline]
    pub fn lerp<S2>(&self, other: &QuaternionBase<N, S2>, t: N) -> OwnedQuaternionBase<N, S::Alloc>
        where S2: Storage<N, U4, U1> {
        self * (N::one() - t) + other * t
    }
}


impl<N, S> QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1>,
          S::Alloc: Allocator<N, U3, U1> {
    /// The polar decomposition of this quaternion.
    ///
    /// Returns, from left to right: the quaternion norm, the half rotation angle, the rotation
    /// axis. If the rotation angle is zero, the rotation axis is set to `None`.
    pub fn polar_decomposition(&self) -> (N, N, Option<Unit<OwnedColumnVector<N, U3, S::Alloc>>>) {
        if let Some((q, n)) = Unit::try_new_and_get(self.clone_owned(), N::zero()) {
            if let Some(axis) = Unit::try_new(self.vector().clone_owned(), N::zero()) {
                let angle = q.angle() / ::convert(2.0f64);

                (n, angle, Some(axis))
            }
            else {
                (n, N::zero(), None)
            }
        }
        else {
            (N::zero(), N::zero(), None)
        }
    }

    /// Compute the exponential of a quaternion.
    #[inline]
    pub fn exp(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        let v  = self.vector();
        let nn = v.norm_squared();

        if relative_eq!(nn, N::zero()) {
            QuaternionBase::identity()
        }
        else {
            let w_exp = self.scalar().exp();
            let n  = nn.sqrt();
            let nv = v * (w_exp * n.sin() / n);

            QuaternionBase::from_parts(n.cos(), nv)
        }
    }

    /// Compute the natural logarithm of a quaternion.
    #[inline]
    pub fn ln(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        let n = self.norm();
        let v = self.vector();
        let s = self.scalar();

        QuaternionBase::from_parts(n.ln(), v.normalize() *  (s / n).acos())
    }

    /// Raise the quaternion to a given floating power.
    #[inline]
    pub fn powf(&self, n: N) -> OwnedQuaternionBase<N, S::Alloc> {
        (self.ln() * n).exp()
    }
}

impl<N, S> QuaternionBase<N, S>
    where N: Real,
          S: StorageMut<N, U4, U1> {
    /// Transforms this quaternion into its 4D vector form (Vector part, Scalar part).
    #[inline]
    pub fn as_vector_mut(&mut self) -> &mut ColumnVector<N, U4, S> {
        &mut self.coords
    }

    /// The mutable vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector_mut(&mut self) -> MatrixSliceMut<N, U3, U1, S::RStride, S::CStride, S::Alloc> {
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
        }
        else {
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

impl<N, S> ApproxEq for QuaternionBase<N, S>
    where N: Real + ApproxEq<Epsilon = N>,
          S: Storage<N, U4, U1> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.as_vector().relative_eq(other.as_vector(), epsilon, max_relative) ||
        // Account for the double-covering of S², i.e. q = -q
       self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.relative_eq(&-*b, epsilon, max_relative))
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_vector().ulps_eq(other.as_vector(), epsilon, max_ulps) ||
        // Account for the double-covering of S², i.e. q = -q.
       self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.ulps_eq(&-*b, epsilon, max_ulps))
    }
}


impl<N, S> fmt::Display for QuaternionBase<N, S>
    where N: Real + fmt::Display,
          S: Storage<N, U4, U1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quaternion {} − ({}, {}, {})", self[3], self[0], self[1], self[2])
    }
}

/// A unit quaternions. May be used to represent a rotation.
pub type UnitQuaternionBase<N, S> = Unit<QuaternionBase<N, S>>;



impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OwnedUnitQuaternionBase<N, S::Alloc> {
        UnitQuaternionBase::new_unchecked(self.unwrap().into_owned())
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OwnedUnitQuaternionBase<N, S::Alloc> {
        UnitQuaternionBase::new_unchecked(self.as_ref().clone_owned())
    }

    /// The rotation angle in [0; pi] of this unit quaternion.
    #[inline]
    pub fn angle(&self) -> N {
        let w = self.quaternion().scalar().abs();

        // Handle innacuracies that make break `.acos`.
        if w >= N::one() {
            N::zero()
        }
        else {
            w.acos() * ::convert(2.0f64)
        }
    }

    /// The underlying quaternion.
    ///
    /// Same as `self.as_ref()`.
    #[inline]
    pub fn quaternion(&self) -> &QuaternionBase<N, S> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit quaternion.
    #[inline]
    pub fn conjugate(&self) -> OwnedUnitQuaternionBase<N, S::Alloc> {
        UnitQuaternionBase::new_unchecked(self.as_ref().conjugate())
    }

    /// Inverts this quaternion if it is not zero.
    #[inline]
    pub fn inverse(&self) -> OwnedUnitQuaternionBase<N, S::Alloc> {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to<S2>(&self, other: &UnitQuaternionBase<N, S2>) -> N
        where S2: Storage<N, U4, U1> {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit quaternion needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to<S2>(&self, other: &UnitQuaternionBase<N, S2>) -> OwnedUnitQuaternionBase<N, S2::Alloc>
        where S2: Storage<N, U4, U1> {
        other / self
    }

    /// Linear interpolation between two unit quaternions.
    ///
    /// The result is not normalized.
    #[inline]
    pub fn lerp<S2>(&self, other: &UnitQuaternionBase<N, S2>, t: N) -> OwnedQuaternionBase<N, S::Alloc>
        where S2: Storage<N, U4, U1> {
        self.as_ref().lerp(other.as_ref(), t)
    }

    /// Normalized linear interpolation between two unit quaternions.
    #[inline]
    pub fn nlerp<S2>(&self, other: &UnitQuaternionBase<N, S2>, t: N) -> OwnedUnitQuaternionBase<N, S::Alloc>
        where S2: Storage<N, U4, U1> {
        let mut res = self.lerp(other, t);
        let _ = res.normalize_mut();

        UnitQuaternionBase::new_unchecked(res)
    }

    /// Spherical linear interpolation between two unit quaternions.
    ///
    /// Panics if the angle between both quaternion is 180 degrees (in which case the interpolation
    /// is not well-defined).
    #[inline]
    pub fn slerp<S2>(&self, other: &UnitQuaternionBase<N, S2>, t: N) -> OwnedUnitQuaternionBase<N, S::Alloc>
        where S2: Storage<N, U4, U1, Alloc = S::Alloc> {
        self.try_slerp(other, t, N::zero()).expect(
            "Unable to perform a spherical quaternion interpolation when they \
             are 180 degree apart (the result is not unique).")
    }

    /// Computes the spherical linear interpolation between two unit quaternions or returns `None`
    /// if both quaternions are approximately 180 degrees apart (in which case the interpolation is
    /// not well-defined).
    ///
    /// # Arguments
    /// * `self`: the first quaternion to interpolate from.
    /// * `other`: the second quaternion to interpolate toward.
    /// * `t`: the interpolation parameter. Should be between 0 and 1.
    /// * `epsilon`: the value bellow which the sinus of the angle separating both quaternion
    /// must be to return `None`.
    #[inline]
    pub fn try_slerp<S2>(&self, other: &UnitQuaternionBase<N, S2>, t: N, epsilon: N)
                         -> Option<OwnedUnitQuaternionBase<N, S::Alloc>>
        where S2: Storage<N, U4, U1, Alloc = S::Alloc> {

        let c_hang = self.coords.dot(&other.coords);

        // self == other
        if c_hang.abs() >= N::one() {
            return Some(self.clone_owned())
        }

        let hang   = c_hang.acos();
        let s_hang = (N::one() - c_hang * c_hang).sqrt();

        // FIXME: what if s_hang is 0.0 ? The result is not well-defined.
        if relative_eq!(s_hang, N::zero(), epsilon = epsilon) {
            None
        }
        else {
            let ta = ((N::one() - t) * hang).sin() / s_hang;
            let tb = (t * hang).sin() / s_hang; 
            let res = self.as_ref() * ta + other.as_ref() * tb;

            Some(UnitQuaternionBase::new_unchecked(res))
        }
    }
}

impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: StorageMut<N, U4, U1> {
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
}

impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1>,
          S::Alloc: Allocator<N, U3, U1> {
    /// The rotation axis of this unit quaternion or `None` if the rotation is zero.
    #[inline]
    pub fn axis(&self) -> Option<Unit<OwnedColumnVector<N, U3, S::Alloc>>> {
        let v =
            if self.quaternion().scalar() >= N::zero() {
                self.as_ref().vector().clone_owned()
            }
            else {
                -self.as_ref().vector()
            };

        Unit::try_new(v, N::zero())
    }


    /// The rotation axis of this unit quaternion multiplied by the rotation agle.
    #[inline]
    pub fn scaled_axis(&self) -> OwnedColumnVector<N, U3, S::Alloc> {
        if let Some(axis) = self.axis() {
            axis.unwrap() * self.angle()
        }
        else {
            ColumnVector::zero()
        }
    }

    /// Compute the exponential of a quaternion.
    ///
    /// Note that this function yields a `QuaternionBase<N>` because it looses the unit property.
    #[inline]
    pub fn exp(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        self.as_ref().exp()
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// Note that this function yields a `QuaternionBase<N>` because it looses the unit property.
    /// The vector part of the return value corresponds to the axis-angle representation (divided
    /// by 2.0) of this unit quaternion.
    #[inline]
    pub fn ln(&self) -> OwnedQuaternionBase<N, S::Alloc> {
        if let Some(v) = self.axis() {
            QuaternionBase::from_parts(N::zero(), v.unwrap() * self.angle())
        }
        else {
            QuaternionBase::zero()
        }
    }

    /// Raise the quaternion to a given floating power.
    ///
    /// This returns the unit quaternion that identifies a rotation with axis `self.axis()` and
    /// angle `self.angle() × n`.
    #[inline]
    pub fn powf(&self, n: N) -> OwnedUnitQuaternionBase<N, S::Alloc> {
        if let Some(v) = self.axis() {
            UnitQuaternionBase::from_axis_angle(&v, self.angle() * n)
        }
        else {
            UnitQuaternionBase::identity()
        }
    }
}

impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1>,
          S::Alloc: Allocator<N, U3, U3> {
    /// Builds a rotation matrix from this unit quaternion.
    #[inline]
    pub fn to_rotation_matrix(&self) -> OwnedRotation<N, U3, S::Alloc> {
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

        RotationBase::from_matrix_unchecked(
            SquareMatrix::<_, U3, _>::new(
                ww + ii - jj - kk, ij - wk,           wj + ik,
                wk + ij,           ww - ii + jj - kk, jk - wi,
                ik - wj,           wi + jk,           ww - ii - jj + kk
            )
        )
    }

    /// Converts this unit quaternion into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, U4, S::Alloc>
        where S::Alloc: Allocator<N, U4, U4> {
        self.to_rotation_matrix().to_homogeneous()
    }
}


impl<N, S> fmt::Display for UnitQuaternionBase<N, S>
    where N: Real + fmt::Display,
          S: Storage<N, U4, U1>,
          S::Alloc: Allocator<N, U3, U1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(axis) = self.axis() {
            let axis = axis.unwrap();
            write!(f, "UnitQuaternion angle: {} − axis: ({}, {}, {})", self.angle(), axis[0], axis[1], axis[2])
        }
        else {
            write!(f, "UnitQuaternion angle: {} − axis: (undefined)", self.angle())
        }
    }
}

impl<N, S> ApproxEq for UnitQuaternionBase<N, S>
    where N: Real + ApproxEq<Epsilon = N>,
          S: Storage<N, U4, U1> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.as_ref().relative_eq(other.as_ref(), epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
