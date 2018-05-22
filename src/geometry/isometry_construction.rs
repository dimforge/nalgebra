#[cfg(feature = "arbitrary")]
use base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use alga::general::Real;
use alga::linear::Rotation as AlgaRotation;

use base::allocator::Allocator;
use base::dimension::{DimName, U2, U3};
use base::{DefaultAllocator, Vector2, Vector3};

use geometry::{
    Isometry, Point, Point3, Rotation, Rotation2, Rotation3, Translation, UnitComplex,
    UnitQuaternion,
};

impl<N: Real, D: DimName, R: AlgaRotation<Point<N, D>>> Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new identity isometry.
    #[inline]
    pub fn identity() -> Self {
        Self::from_parts(Translation::identity(), R::identity())
    }

    /// The isometry that applies the rotation `r` with its axis passing through the point `p`.
    /// This effectively lets `p` invariant.
    #[inline]
    pub fn rotation_wrt_point(r: R, p: Point<N, D>) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(Translation::from_vector(shift + p.coords), r)
    }
}

impl<N: Real, D: DimName, R: AlgaRotation<Point<N, D>>> One for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new identity isometry.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real, D: DimName, R> Distribution<Isometry<N, D, R>> for Standard
where
    R: AlgaRotation<Point<N, D>>,
    Standard: Distribution<N> + Distribution<R>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> Isometry<N, D, R> {
        Isometry::from_parts(rng.gen(), rng.gen())
    }
}

#[cfg(feature = "arbitrary")]
impl<N, D: DimName, R> Arbitrary for Isometry<N, D, R>
where
    N: Real + Arbitrary + Send,
    R: AlgaRotation<Point<N, D>> + Arbitrary + Send,
    Owned<N, D>: Send,
    DefaultAllocator: Allocator<N, D>,
{
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
impl<N: Real> Isometry<N, U2, Rotation2<N>> {
    /// Creates a new isometry from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N) -> Self {
        Self::from_parts(
            Translation::from_vector(translation),
            Rotation::<N, U2>::new(angle),
        )
    }
}

impl<N: Real> Isometry<N, U2, UnitComplex<N>> {
    /// Creates a new isometry from a translation and a rotation angle.
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N) -> Self {
        Self::from_parts(
            Translation::from_vector(translation),
            UnitComplex::from_angle(angle),
        )
    }
}

// 3D rotation.
macro_rules! isometry_construction_impl(
    ($RotId: ident < $($RotParams: ident),*>, $RRDim: ty, $RCDim: ty) => {
        impl<N: Real> Isometry<N, U3, $RotId<$($RotParams),*>> {
            /// Creates a new isometry from a translation and a rotation axis-angle.
            #[inline]
            pub fn new(translation: Vector3<N>, axisangle: Vector3<N>) -> Self {
                Self::from_parts(
                    Translation::from_vector(translation),
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
            pub fn new_observer_frame(eye:    &Point3<N>,
                                      target: &Point3<N>,
                                      up:     &Vector3<N>)
                                      -> Self {
                Self::from_parts(
                    Translation::from_vector(eye.coords.clone()),
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
            pub fn look_at_rh(eye:    &Point3<N>,
                              target: &Point3<N>,
                              up:     &Vector3<N>)
                              -> Self {
                let rotation = $RotId::look_at_rh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(Translation::from_vector(trans.coords), rotation)
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
            pub fn look_at_lh(eye:    &Point3<N>,
                              target: &Point3<N>,
                              up:     &Vector3<N>)
                              -> Self {
                let rotation = $RotId::look_at_lh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(Translation::from_vector(trans.coords), rotation)
            }
        }
    }
);

isometry_construction_impl!(Rotation3<N>, U3, U3);
isometry_construction_impl!(UnitQuaternion<N>, U4, U1);
