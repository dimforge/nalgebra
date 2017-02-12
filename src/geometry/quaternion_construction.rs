#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use rand::{Rand, Rng};
use num::{Zero, One};

use alga::general::Real;

use core::{Unit, ColumnVector, Vector3};
use core::storage::{Storage, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use core::dimension::{U1, U3, U4};

use geometry::{QuaternionBase, UnitQuaternionBase, RotationBase, OwnedRotation};

impl<N, S> QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    /// Creates a quaternion from a 4D vector. The quaternion scalar part corresponds to the `w`
    /// vector component.
    #[inline]
    pub fn from_vector(vector: ColumnVector<N, U4, S>) -> Self {
        QuaternionBase {
            coords: vector
        }
    }
}

impl<N, S> QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    /// Creates a new quaternion from its individual components. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is `[ x, y, z, w ]`.
    #[inline]
    pub fn new(w: N, x: N, y: N, z: N) -> Self {
        let v = ColumnVector::<N, U4, S>::new(x, y, z, w);
        Self::from_vector(v)
    }

    /// Creates a new quaternion from its scalar and vector parts. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is [ vector, scalar ].
    #[inline]
    // FIXME: take a reference to `vector`?
    pub fn from_parts<SB>(scalar: N, vector: ColumnVector<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U1> {

        Self::new(scalar, vector[0], vector[1], vector[2])
    }

    /// Creates a new quaternion from its polar decomposition.
    ///
    /// Note that `axis` is assumed to be a unit vector.
    // FIXME: take a reference to `axis`?
    pub fn from_polar_decomposition<SB>(scale: N, theta: N, axis: Unit<ColumnVector<N, U3, SB>>) -> Self
        where SB: Storage<N, U3, U1> {
        let rot = UnitQuaternionBase::<N, S>::from_axis_angle(&axis, theta * ::convert(2.0f64));

        rot.unwrap() * scale
    }

    /// The quaternion multiplicative identity.
    #[inline]
    pub fn identity() -> Self {
        Self::new(N::one(), N::zero(), N::zero(), N::zero())
    }
}

impl<N, S> One for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, S> Zero for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn zero() -> Self {
        Self::new(N::zero(), N::zero(), N::zero(), N::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coords.is_zero()
    }
}

impl<N, S> Rand for QuaternionBase<N, S>
    where N: Real + Rand,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Self {
        QuaternionBase::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
    }
}

#[cfg(feature="arbitrary")]
impl<N, S> Arbitrary for QuaternionBase<N, S>
    where N: Real + Arbitrary,
          S: OwnedStorage<N, U4, U1> + Send,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        QuaternionBase::new(N::arbitrary(g), N::arbitrary(g),
                            N::arbitrary(g), N::arbitrary(g))
    }
}

impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    /// The quaternion multiplicative identity.
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(QuaternionBase::identity())
    }

    /// Creates a new quaternion from a unit vector (the rotation axis) and an angle
    /// (the rotation angle).
    #[inline]
    pub fn from_axis_angle<SB>(axis: &Unit<ColumnVector<N, U3, SB>>, angle: N) -> Self
        where SB: Storage<N, U3, U1> {
        let (sang, cang) = (angle / ::convert(2.0f64)).sin_cos();

        let q = QuaternionBase::from_parts(cang, axis.as_ref() * sang);
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion from a quaternion.
    ///
    /// The input quaternion will be normalized.
    #[inline]
    pub fn from_quaternion(q: QuaternionBase<N, S>) -> Self {
        Self::new_normalize(q)
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    #[inline]
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self {
        let (sr, cr) = (roll  * ::convert(0.5f64)).sin_cos();
        let (sp, cp) = (pitch * ::convert(0.5f64)).sin_cos();
        let (sy, cy) = (yaw   * ::convert(0.5f64)).sin_cos();

        let q = QuaternionBase::new(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy);

        Self::new_unchecked(q)
    }

    /// Builds an unit quaternion from a rotation matrix.
    #[inline]
    pub fn from_rotation_matrix<SB>(rotmat: &RotationBase<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U3>,
              SB::Alloc: Allocator<N, U3, U1> {

        // Robust matrix to quaternion transformation.
        // See http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion
        let tr = rotmat[(0, 0)] + rotmat[(1, 1)] + rotmat[(2, 2)];
        let res;

        let _0_25: N = ::convert(0.25);

        if tr > N::zero() {
          let denom = (tr + N::one()).sqrt() * ::convert(2.0);
          res = QuaternionBase::new(_0_25 * denom,
                                    (rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                                    (rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                                    (rotmat[(1, 0)] - rotmat[(0, 1)]) / denom);
        }
        else if rotmat[(0, 0)] > rotmat[(1, 1)] && rotmat[(0, 0)] > rotmat[(2, 2)] {
          let denom = (N::one() + rotmat[(0, 0)] - rotmat[(1, 1)] - rotmat[(2, 2)]).sqrt() * ::convert(2.0);
          res = QuaternionBase::new((rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                                    _0_25 * denom,
                                    (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                                    (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom);
        }
        else if rotmat[(1, 1)] > rotmat[(2, 2)] {
          let denom = (N::one() + rotmat[(1, 1)] - rotmat[(0, 0)] - rotmat[(2, 2)]).sqrt() * ::convert(2.0);
          res = QuaternionBase::new((rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                                    (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                                    _0_25 * denom,
                                    (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom);
        }
        else {
          let denom = (N::one() + rotmat[(2, 2)] - rotmat[(0, 0)] - rotmat[(1, 1)]).sqrt() * ::convert(2.0);
          res = QuaternionBase::new((rotmat[(1, 0)] - rotmat[(0, 1)]) / denom,
                                    (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom,
                                    (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom,
                                    _0_25 * denom);
        }

        Self::new_unchecked(res)
    }

    /// The unit quaternion needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    #[inline]
    pub fn rotation_between<SB, SC>(a: &ColumnVector<N, U3, SB>, b: &ColumnVector<N, U3, SC>) -> Option<Self>
        where SB: Storage<N, U3, U1>,
              SC: Storage<N, U3, U1> {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(a: &ColumnVector<N, U3, SB>, b: &ColumnVector<N, U3, SC>, s: N) -> Option<Self>
        where SB: Storage<N, U3, U1>,
              SC: Storage<N, U3, U1> {
        // FIXME: code duplication with RotationBase.
        if let (Some(na), Some(nb)) = (a.try_normalize(N::zero()), b.try_normalize(N::zero())) {
            let c = na.cross(&nb);

            if let Some(axis) = Unit::try_new(c, N::default_epsilon()) {
                return Some(Self::from_axis_angle(&axis, na.dot(&nb).acos() * s))
            }

            // Zero or PI.
            if na.dot(&nb) < N::zero() {
                // PI
                //
                // The rotation axis is undefined but the angle not zero. This is not a
                // simple rotation.
                return None;
            }
        }

        Some(Self::identity())
    }


    /// Creates an unit quaternion that corresponds to the local frame of an observer standing at the
    /// origin and looking toward `dir`.
    ///
    /// It maps the view direction `dir` to the positive `z` axis.
    ///
    /// # Arguments
    ///   * dir - The look direction, that is, direction the matrix `z` axis will be aligned with.
    ///   * up - The vertical direction. The only requirement of this parameter is to not be
    ///   collinear
    ///   to `dir`. Non-collinearity is not checked.
    #[inline]
    pub fn new_observer_frame<SB, SC>(dir: &ColumnVector<N, U3, SB>, up: &ColumnVector<N, U3, SC>) -> Self
    where SB: Storage<N, U3, U1>,
          SC: Storage<N, U3, U1>,
          S::Alloc: Allocator<N, U3, U3> +
                    Allocator<N, U3, U1> {
        Self::from_rotation_matrix(&OwnedRotation::<N, U3, S::Alloc>::new_observer_frame(dir, up))
    }


    /// Builds a right-handed look-at view matrix without translation.
    ///
    /// This conforms to the common notion of right handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * eye - The eye position.
    ///   * target - The target position.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `target - eye`.
    #[inline]
    pub fn look_at_rh<SB, SC>(dir: &ColumnVector<N, U3, SB>, up: &ColumnVector<N, U3, SC>) -> Self
    where SB: Storage<N, U3, U1>,
          SC: Storage<N, U3, U1>,
          S::Alloc: Allocator<N, U3, U3> +
                    Allocator<N, U3, U1> {
        Self::new_observer_frame(&-dir, up).inverse()
    }

    /// Builds a left-handed look-at view matrix without translation.
    ///
    /// This conforms to the common notion of left handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * eye - The eye position.
    ///   * target - The target position.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `target - eye`.
    #[inline]
    pub fn look_at_lh<SB, SC>(dir: &ColumnVector<N, U3, SB>, up: &ColumnVector<N, U3, SC>) -> Self
    where SB: Storage<N, U3, U1>,
          SC: Storage<N, U3, U1>,
          S::Alloc: Allocator<N, U3, U3> +
                    Allocator<N, U3, U1> {
            Self::new_observer_frame(dir, up).inverse()
    }
}

impl<N, S> UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> +
                    Allocator<N, U3, U1> {
    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` is zero, this returns the indentity rotation.
    #[inline]
    pub fn new<SB>(axisangle: ColumnVector<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U1> {
        let two: N = ::convert(2.0f64);
        let q = QuaternionBase::<N, S>::from_parts(N::zero(), axisangle / two).exp();
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` is zero, this returns the indentity rotation.
    /// Same as `Self::new(axisangle)`.
    #[inline]
    pub fn from_scaled_axis<SB>(axisangle: ColumnVector<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U1> {
        Self::new(axisangle)
    }
}

impl<N, S> One for UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, S> Rand for UnitQuaternionBase<N, S>
    where N: Real + Rand,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> +
                    Allocator<N, U3, U1> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let axisangle = Vector3::rand(rng);
        UnitQuaternionBase::from_scaled_axis(axisangle)
    }
}

#[cfg(feature="arbitrary")]
impl<N, S> Arbitrary for UnitQuaternionBase<N, S>
    where N: Real + Arbitrary,
          S: OwnedStorage<N, U4, U1> + Send,
          S::Alloc: OwnedAllocator<N, U4, U1, S> +
                    Allocator<N, U3, U1> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let axisangle = Vector3::arbitrary(g);
        UnitQuaternionBase::from_scaled_axis(axisangle)

    }
}
