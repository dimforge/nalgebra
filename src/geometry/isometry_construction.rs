#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use rand::{Rng, Rand};

use alga::general::Real;
use alga::linear::Rotation as AlgaRotation;

use core::ColumnVector;
use core::dimension::{DimName, U1, U2, U3, U4};
use core::allocator::{OwnedAllocator, Allocator};
use core::storage::OwnedStorage;

use geometry::{PointBase, TranslationBase, RotationBase, IsometryBase, UnitQuaternionBase, UnitComplex};


impl<N, D: DimName, S, R> IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new identity isometry.
    #[inline]
    pub fn identity() -> Self {
        Self::from_parts(TranslationBase::identity(), R::identity())
    }
}

impl<N, D: DimName, S, R> One for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new identity isometry.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S, R> Rand for IsometryBase<N, D, S, R>
    where N: Real + Rand,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>> + Rand,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        Self::from_parts(rng.gen(), rng.gen())
    }
}

impl<N, D: DimName, S, R> IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// The isometry that applies the rotation `r` with its axis passing through the point `p`.
    /// This effectively lets `p` invariant.
    #[inline]
    pub fn rotation_wrt_point(r: R, p: PointBase<N, D, S>) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(TranslationBase::from_vector(shift + p.coords), r)
    }
}

#[cfg(feature = "arbitrary")]
impl<N, D: DimName, S, R> Arbitrary for IsometryBase<N, D, S, R>
    where N: Real + Arbitrary + Send,
          S: OwnedStorage<N, D, U1> + Send,
          R: AlgaRotation<PointBase<N, D, S>> + Arbitrary + Send,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        Self::from_parts(Arbitrary::arbitrary(rng), Arbitrary::arbitrary(rng))
    }
}

/*
 *
 * Constructors for various static dimensions.
 *
 */

// 2D rotation.
impl<N, S, SR> IsometryBase<N, U2, S, RotationBase<N, U2, SR>>
    where N: Real,
          S:  OwnedStorage<N, U2, U1, Alloc = SR::Alloc>,
          SR: OwnedStorage<N, U2, U2>,
          S::Alloc:  OwnedAllocator<N, U2, U1, S>,
          SR::Alloc: OwnedAllocator<N, U2, U2, SR> {
    /// Creates a new isometry from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: ColumnVector<N, U2, S>, angle: N) -> Self {
        Self::from_parts(TranslationBase::from_vector(translation), RotationBase::<N, U2, SR>::new(angle))
    }
}

impl<N, S> IsometryBase<N, U2, S, UnitComplex<N>>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc:  OwnedAllocator<N, U2, U1, S> {
    /// Creates a new isometry from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: ColumnVector<N, U2, S>, angle: N) -> Self {
        Self::from_parts(TranslationBase::from_vector(translation), UnitComplex::from_angle(angle))
    }
}

// 3D rotation.
macro_rules! isometry_construction_impl(
    ($RotId: ident < $($RotParams: ident),*>, $RRDim: ty, $RCDim: ty) => {
        impl<N, S, SR> IsometryBase<N, U3, S, $RotId<$($RotParams),*>>
            where N: Real,
                  S:  OwnedStorage<N, U3, U1, Alloc = SR::Alloc>,
                  SR: OwnedStorage<N, $RRDim, $RCDim>,
                  S::Alloc:  OwnedAllocator<N, U3, U1, S>,
                  SR::Alloc: OwnedAllocator<N, $RRDim, $RCDim, SR> +
                             Allocator<N, U3, U3> {
            /// Creates a new isometry from a translation and a rotation axis-angle.
            #[inline]
            pub fn new(translation: ColumnVector<N, U3, S>, axisangle: ColumnVector<N, U3, S>) -> Self {
                Self::from_parts(
                    TranslationBase::from_vector(translation),
                    $RotId::<$($RotParams),*>::from_scaled_axis(axisangle))
            }

            /// Creates an isometry that corresponds to the local frame of an observer standing at the
            /// point `eye` and looking toward `target`.
            ///
            /// It maps the view direction `target - eye` to the positive `z` axis and the origin to the
            /// `eye`.
            ///
            /// # Arguments
            ///   * eye - The observer position.
            ///   * target - The target position.
            ///   * up - Vertical direction. The only requirement of this parameter is to not be collinear
            ///   to `eye - at`. Non-collinearity is not checked.
            #[inline]
            pub fn new_observer_frame(eye:    &PointBase<N, U3, S>,
                                      target: &PointBase<N, U3, S>,
                                      up:     &ColumnVector<N, U3, S>)
                                      -> Self {
                Self::from_parts(
                    TranslationBase::from_vector(eye.coords.clone()),
                    $RotId::new_observer_frame(&(target - eye), up))
            }

            /// Builds a right-handed look-at view matrix.
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
            pub fn look_at_rh(eye:    &PointBase<N, U3, S>,
                              target: &PointBase<N, U3, S>,
                              up:     &ColumnVector<N, U3, S>)
                              -> Self {
                let rotation = $RotId::look_at_rh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(TranslationBase::from_vector(trans.coords), rotation)
            }

            /// Builds a left-handed look-at view matrix.
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
            pub fn look_at_lh(eye:    &PointBase<N, U3, S>,
                              target: &PointBase<N, U3, S>,
                              up:     &ColumnVector<N, U3, S>)
                              -> Self {
                let rotation = $RotId::look_at_lh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(TranslationBase::from_vector(trans.coords), rotation)
            }
        }
    }
);

isometry_construction_impl!(RotationBase<N, U3, SR>, U3, U3);
isometry_construction_impl!(UnitQuaternionBase<N, SR>, U4, U1);
