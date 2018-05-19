#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
#[cfg(feature = "arbitrary")]
use base::storage::Owned;

use num::One;
use rand::{Rand, Rng};

use alga::general::Real;
use alga::linear::Rotation as AlgaRotation;

use base::{DefaultAllocator, Vector2, Vector3};
use base::dimension::{DimName, U2, U3};
use base::allocator::Allocator;

use geometry::{Isometry, Point, Point3, Rotation2, Rotation3, Similarity, Translation,
               UnitComplex, UnitQuaternion};

impl<N: Real, D: DimName, R> Similarity<N, D, R>
where
    R: AlgaRotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new identity similarity.
    #[inline]
    pub fn identity() -> Self {
        Self::from_isometry(Isometry::identity(), N::one())
    }
}

impl<N: Real, D: DimName, R> One for Similarity<N, D, R>
where
    R: AlgaRotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new identity similarity.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real + Rand, D: DimName, R> Rand for Similarity<N, D, R>
where
    R: AlgaRotation<Point<N, D>> + Rand,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        let mut s = rng.gen();
        while relative_eq!(s, N::zero()) {
            s = rng.gen()
        }

        Self::from_isometry(rng.gen(), s)
    }
}

impl<N: Real, D: DimName, R> Similarity<N, D, R>
where
    R: AlgaRotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    /// The similarity that applies tha scaling factor `scaling`, followed by the rotation `r` with
    /// its axis passing through the point `p`.
    #[inline]
    pub fn rotation_wrt_point(r: R, p: Point<N, D>, scaling: N) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(Translation::from_vector(shift + p.coords), r, scaling)
    }
}

#[cfg(feature = "arbitrary")]
impl<N, D: DimName, R> Arbitrary for Similarity<N, D, R>
where
    N: Real + Arbitrary + Send,
    R: AlgaRotation<Point<N, D>> + Arbitrary + Send,
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Send,
{
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
impl<N: Real> Similarity<N, U2, Rotation2<N>> {
    /// Creates a new similarity from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N, scaling: N) -> Self {
        Self::from_parts(
            Translation::from_vector(translation),
            Rotation2::new(angle),
            scaling,
        )
    }
}

impl<N: Real> Similarity<N, U2, UnitComplex<N>> {
    /// Creates a new similarity from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N, scaling: N) -> Self {
        Self::from_parts(
            Translation::from_vector(translation),
            UnitComplex::new(angle),
            scaling,
        )
    }
}

// 3D rotation.
macro_rules! similarity_construction_impl(
    ($Rot: ty) => {
        impl<N: Real> Similarity<N, U3, $Rot> {
            /// Creates a new similarity from a translation, rotation axis-angle, and scaling
            /// factor.
            #[inline]
            pub fn new(translation: Vector3<N>, axisangle: Vector3<N>, scaling: N) -> Self {
                Self::from_isometry(Isometry::<_, U3, $Rot>::new(translation, axisangle), scaling)
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
            pub fn new_observer_frame(eye:    &Point3<N>,
                                      target: &Point3<N>,
                                      up:     &Vector3<N>,
                                      scaling: N)
                                      -> Self {
                Self::from_isometry(Isometry::<_, U3, $Rot>::new_observer_frame(eye, target, up), scaling)
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
            pub fn look_at_rh(eye:     &Point3<N>,
                              target:  &Point3<N>,
                              up:      &Vector3<N>,
                              scaling: N)
                              -> Self {
                Self::from_isometry(Isometry::<_, U3, $Rot>::look_at_rh(eye, target, up), scaling)
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
            pub fn look_at_lh(eye:     &Point3<N>,
                              target:  &Point3<N>,
                              up:      &Vector3<N>,
                              scaling: N)
                              -> Self {
                Self::from_isometry(Isometry::<_, _, $Rot>::look_at_lh(eye, target, up), scaling)
            }
        }
    }
);

similarity_construction_impl!(Rotation3<N>);
similarity_construction_impl!(UnitQuaternion<N>);
