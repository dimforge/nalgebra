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

use geometry::{PointBase, TranslationBase, RotationBase, SimilarityBase,
               UnitComplex, UnitQuaternionBase, IsometryBase};


impl<N, D: DimName, S, R> SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new identity similarity.
    #[inline]
    pub fn identity() -> Self {
        Self::from_isometry(IsometryBase::identity(), N::one())
    }
}

impl<N, D: DimName, S, R> One for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new identity similarity.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S, R> Rand for SimilarityBase<N, D, S, R>
    where N: Real + Rand,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>> + Rand,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        let mut s = rng.gen();
        while relative_eq!(s, N::zero()) {
            s = rng.gen()
        }

        Self::from_isometry(rng.gen(), s)
    }
}

impl<N, D: DimName, S, R> SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: AlgaRotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// The similarity that applies tha scaling factor `scaling`, followed by the rotation `r` with
    /// its axis passing through the point `p`.
    #[inline]
    pub fn rotation_wrt_point(r: R, p: PointBase<N, D, S>, scaling: N) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(TranslationBase::from_vector(shift + p.coords), r, scaling)
    }
}

#[cfg(feature = "arbitrary")]
impl<N, D: DimName, S, R> Arbitrary for SimilarityBase<N, D, S, R>
    where N: Real + Arbitrary + Send,
          S: OwnedStorage<N, D, U1> + Send,
          R: AlgaRotation<PointBase<N, D, S>> + Arbitrary + Send,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        let mut s = Arbitrary::arbitrary(rng);
        while relative_eq!(s, N::zero()) {
            s = Arbitrary::arbitrary(rng)
        }

        Self::from_isometry(Arbitrary::arbitrary(rng), s)
    }
}

/*
 *
 * Constructors for various static dimensions.
 *
 */

// 2D rotation.
impl<N, S, SR> SimilarityBase<N, U2, S, RotationBase<N, U2, SR>>
    where N: Real,
          S:  OwnedStorage<N, U2, U1, Alloc = SR::Alloc>,
          SR: OwnedStorage<N, U2, U2>,
          S::Alloc:  OwnedAllocator<N, U2, U1, S>,
          SR::Alloc: OwnedAllocator<N, U2, U2, SR> {
    /// Creates a new similarity from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: ColumnVector<N, U2, S>, angle: N, scaling: N) -> Self {
        Self::from_parts(TranslationBase::from_vector(translation), RotationBase::<N, U2, SR>::new(angle), scaling)
    }
}

impl<N, S> SimilarityBase<N, U2, S, UnitComplex<N>>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    /// Creates a new similarity from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: ColumnVector<N, U2, S>, angle: N, scaling: N) -> Self {
        Self::from_parts(TranslationBase::from_vector(translation), UnitComplex::new(angle), scaling)
    }
}

// 3D rotation.
macro_rules! similarity_construction_impl(
    ($Rot: ty, $RotId: ident, $RRDim: ty, $RCDim: ty) => {
        impl<N, S, SR> SimilarityBase<N, U3, S, $Rot>
            where N: Real,
                  S:  OwnedStorage<N, U3, U1, Alloc = SR::Alloc>,
                  SR: OwnedStorage<N, $RRDim, $RCDim>,
                  S::Alloc:  OwnedAllocator<N, U3, U1, S>,
                  SR::Alloc: OwnedAllocator<N, $RRDim, $RCDim, SR> +
                             Allocator<N, U3, U3> {
            /// Creates a new similarity from a translation, rotation axis-angle, and scaling
            /// factor.
            #[inline]
            pub fn new(translation: ColumnVector<N, U3, S>, axisangle: ColumnVector<N, U3, S>, scaling: N) -> Self {
                Self::from_isometry(IsometryBase::<_, _, _, $Rot>::new(translation, axisangle), scaling)
            }

            /// Creates an similarity that corresponds to the a scaling factor and a local frame of
            /// an observer standing at the point `eye` and looking toward `target`.
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
                                      up:     &ColumnVector<N, U3, S>,
                                      scaling: N)
                                      -> Self {
                Self::from_isometry(IsometryBase::<_, _, _, $Rot>::new_observer_frame(eye, target, up), scaling)
            }

            /// Builds a right-handed look-at view matrix including scaling factor.
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
            pub fn look_at_rh(eye:     &PointBase<N, U3, S>,
                              target:  &PointBase<N, U3, S>,
                              up:      &ColumnVector<N, U3, S>,
                              scaling: N)
                              -> Self {
                Self::from_isometry(IsometryBase::<_, _, _, $Rot>::look_at_rh(eye, target, up), scaling)
            }

            /// Builds a left-handed look-at view matrix including a scaling factor.
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
            pub fn look_at_lh(eye:     &PointBase<N, U3, S>,
                              target:  &PointBase<N, U3, S>,
                              up:      &ColumnVector<N, U3, S>,
                              scaling: N)
                              -> Self {
                Self::from_isometry(IsometryBase::<_, _, _, $Rot>::look_at_lh(eye, target, up), scaling)
            }
        }
    }
);

similarity_construction_impl!(RotationBase<N, U3, SR>, RotationBase, U3, U3);
similarity_construction_impl!(UnitQuaternionBase<N, SR>, UnitQuaternionBase, U4, U1);
