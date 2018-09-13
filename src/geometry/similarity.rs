use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{Real, SubsetOf};
use alga::linear::Rotation;

use base::allocator::Allocator;
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use base::storage::Owned;
use base::{DefaultAllocator, MatrixN};
use geometry::{Isometry, Point, Translation};

/// A similarity, i.e., an uniform scaling, followed by a rotation, followed by a translation.
#[repr(C)]
#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            serialize = "N: Serialize,
                     R: Serialize,
                     DefaultAllocator: Allocator<N, D>,
                     Owned<N, D>: Serialize"
        )
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            deserialize = "N: Deserialize<'de>,
                       R: Deserialize<'de>,
                       DefaultAllocator: Allocator<N, D>,
                       Owned<N, D>: Deserialize<'de>"
        )
    )
)]
pub struct Similarity<N: Real, D: DimName, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// The part of this similarity that does not include the scaling factor.
    pub isometry: Isometry<N, D, R>,
    scaling: N,
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Real, D: DimName, R> Abomonation for Similarity<N, D, R>
where
    Isometry<N, D, R>: Abomonation,
    DefaultAllocator: Allocator<N, D>,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.isometry.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.isometry.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.isometry.exhume(bytes)
    }
}

impl<N: Real + hash::Hash, D: DimName + hash::Hash, R: hash::Hash> hash::Hash
    for Similarity<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.isometry.hash(state);
        self.scaling.hash(state);
    }
}

impl<N: Real, D: DimName + Copy, R: Rotation<Point<N, D>> + Copy> Copy for Similarity<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Copy,
{
}

impl<N: Real, D: DimName, R: Rotation<Point<N, D>> + Clone> Clone for Similarity<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn clone(&self) -> Self {
        Similarity::from_isometry(self.isometry.clone(), self.scaling)
    }
}

impl<N: Real, D: DimName, R> Similarity<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new similarity from its rotational and translational parts.
    #[inline]
    pub fn from_parts(
        translation: Translation<N, D>,
        rotation: R,
        scaling: N,
    ) -> Similarity<N, D, R> {
        Similarity::from_isometry(Isometry::from_parts(translation, rotation), scaling)
    }

    /// Creates a new similarity from its rotational and translational parts.
    #[inline]
    pub fn from_isometry(isometry: Isometry<N, D, R>, scaling: N) -> Similarity<N, D, R> {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The scaling factor must not be zero."
        );

        Similarity {
            isometry: isometry,
            scaling: scaling,
        }
    }

    /// Creates a new similarity that applies only a scaling factor.
    #[inline]
    pub fn from_scaling(scaling: N) -> Similarity<N, D, R> {
        Self::from_isometry(Isometry::identity(), scaling)
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> Similarity<N, D, R> {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts `self` in-place.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.scaling = N::one() / self.scaling;
        self.isometry.inverse_mut();
        self.isometry.translation.vector *= self.scaling;
    }

    /// The scaling factor of this similarity transformation.
    #[inline]
    pub fn set_scaling(&mut self, scaling: N) {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The similarity scaling factor must not be zero."
        );

        self.scaling = scaling;
    }

    /// The scaling factor of this similarity transformation.
    #[inline]
    pub fn scaling(&self) -> N {
        self.scaling
    }

    /// The similarity transformation that applies a scaling factor `scaling` before `self`.
    #[inline]
    pub fn prepend_scaling(&self, scaling: N) -> Self {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The similarity scaling factor must not be zero."
        );

        Self::from_isometry(self.isometry.clone(), self.scaling * scaling)
    }

    /// The similarity transformation that applies a scaling factor `scaling` after `self`.
    #[inline]
    pub fn append_scaling(&self, scaling: N) -> Self {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The similarity scaling factor must not be zero."
        );

        Self::from_parts(
            Translation::from_vector(&self.isometry.translation.vector * scaling),
            self.isometry.rotation.clone(),
            self.scaling * scaling,
        )
    }

    /// Sets `self` to the similarity transformation that applies a scaling factor `scaling` before `self`.
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: N) {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The similarity scaling factor must not be zero."
        );

        self.scaling *= scaling
    }

    /// Sets `self` to the similarity transformation that applies a scaling factor `scaling` after `self`.
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: N) {
        assert!(
            !relative_eq!(scaling, N::zero()),
            "The similarity scaling factor must not be zero."
        );

        self.isometry.translation.vector *= scaling;
        self.scaling *= scaling;
    }

    /// Appends to `self` the given translation in-place.
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<N, D>) {
        self.isometry.append_translation_mut(t)
    }

    /// Appends to `self` the given rotation in-place.
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.isometry.append_rotation_mut(r)
    }

    /// Appends in-place to `self` a rotation centered at the point `p`, i.e., the rotation that
    /// lets `p` invariant.
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<N, D>) {
        self.isometry.append_rotation_wrt_point_mut(r, p)
    }

    /// Appends in-place to `self` a rotation centered at the point with coordinates
    /// `self.translation`.
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        self.isometry.append_rotation_wrt_center_mut(r)
    }
}

// NOTE: we don't require `R: Rotation<...>` here becaus this is not useful for the implementation
// and makes it harde to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the private scaling factor).
impl<N: Real, D: DimName, R> Similarity<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Converts this similarity into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        D: DimNameAdd<U1>,
        R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        let mut res = self.isometry.to_homogeneous();

        for e in res.fixed_slice_mut::<D, D>(0, 0).iter_mut() {
            *e *= self.scaling
        }

        res
    }
}

impl<N: Real, D: DimName, R> Eq for Similarity<N, D, R>
where
    R: Rotation<Point<N, D>> + Eq,
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N: Real, D: DimName, R> PartialEq for Similarity<N, D, R>
where
    R: Rotation<Point<N, D>> + PartialEq,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn eq(&self, right: &Similarity<N, D, R>) -> bool {
        self.isometry == right.isometry && self.scaling == right.scaling
    }
}

impl<N: Real, D: DimName, R> AbsDiffEq for Similarity<N, D, R>
where
    R: Rotation<Point<N, D>> + AbsDiffEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.isometry.abs_diff_eq(&other.isometry, epsilon)
            && self.scaling.abs_diff_eq(&other.scaling, epsilon)
    }
}

impl<N: Real, D: DimName, R> RelativeEq for Similarity<N, D, R>
where
    R: Rotation<Point<N, D>> + RelativeEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
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
        self.isometry
            .relative_eq(&other.isometry, epsilon, max_relative)
            && self
                .scaling
                .relative_eq(&other.scaling, epsilon, max_relative)
    }
}

impl<N: Real, D: DimName, R> UlpsEq for Similarity<N, D, R>
where
    R: Rotation<Point<N, D>> + UlpsEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.isometry.ulps_eq(&other.isometry, epsilon, max_ulps)
            && self.scaling.ulps_eq(&other.scaling, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName, R> fmt::Display for Similarity<N, D, R>
where
    N: Real + fmt::Display,
    R: Rotation<Point<N, D>> + fmt::Display,
    DefaultAllocator: Allocator<N, D> + Allocator<usize, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "Similarity {{"));
        try!(write!(f, "{:.*}", precision, self.isometry));
        try!(write!(f, "Scaling: {:.*}", precision, self.scaling));
        writeln!(f, "}}")
    }
}
