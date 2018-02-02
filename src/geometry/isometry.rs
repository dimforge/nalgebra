use std::fmt;
use std::hash;
use std::marker::PhantomData;
use approx::ApproxEq;

#[cfg(feature = "serde-serialize")]
use serde;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{Real, SubsetOf};
use alga::linear::Rotation;

use core::{DefaultAllocator, MatrixN};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use core::storage::Owned;
use core::allocator::Allocator;
use geometry::{Point, Translation};

/// A direct isometry, i.e., a rotation followed by a translation.
#[repr(C)]
#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde-serialize",
           serde(bound(serialize = "R: serde::Serialize,
                     DefaultAllocator: Allocator<N, D>,
                     Owned<N, D>: serde::Serialize")))]
#[cfg_attr(feature = "serde-serialize",
           serde(bound(deserialize = "R: serde::Deserialize<'de>,
                       DefaultAllocator: Allocator<N, D>,
                       Owned<N, D>: serde::Deserialize<'de>")))]
pub struct Isometry<N: Real, D: DimName, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// The pure rotational part of this isometry.
    pub rotation: R,
    /// The pure translational part of this isometry.
    pub translation: Translation<N, D>,

    // One dummy private field just to prevent explicit construction.
    #[cfg_attr(feature = "serde-serialize", serde(skip_serializing, skip_deserializing))]
    _noconstruct: PhantomData<N>,
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D, R> Abomonation for Isometry<N, D, R>
where
    N: Real,
    D: DimName,
    R: Abomonation,
    Translation<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D>,
{
    unsafe fn entomb(&self, writer: &mut Vec<u8>) {
        self.rotation.entomb(writer);
        self.translation.entomb(writer);
    }

    unsafe fn embalm(&mut self) {
        self.rotation.embalm();
        self.translation.embalm();
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.rotation
            .exhume(bytes)
            .and_then(|bytes| self.translation.exhume(bytes))
    }
}

impl<N: Real + hash::Hash, D: DimName + hash::Hash, R: hash::Hash> hash::Hash for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.translation.hash(state);
        self.rotation.hash(state);
    }
}

impl<N: Real, D: DimName + Copy, R: Rotation<Point<N, D>> + Copy> Copy for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Copy,
{
}

impl<N: Real, D: DimName, R: Rotation<Point<N, D>> + Clone> Clone for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn clone(&self) -> Self {
        Isometry::from_parts(self.translation.clone(), self.rotation.clone())
    }
}

impl<N: Real, D: DimName, R: Rotation<Point<N, D>>> Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new isometry from its rotational and translational parts.
    #[inline]
    pub fn from_parts(translation: Translation<N, D>, rotation: R) -> Isometry<N, D, R> {
        Isometry {
            rotation: rotation,
            translation: translation,
            _noconstruct: PhantomData,
        }
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> Isometry<N, D, R> {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.transform_vector(&self.translation.vector);
    }

    /// Appends to `self` the given translation in-place.
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<N, D>) {
        self.translation.vector += &t.vector
    }

    /// Appends to `self` the given rotation in-place.
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.rotation = self.rotation.append_rotation(&r);
        self.translation.vector = r.transform_vector(&self.translation.vector);
    }

    /// Appends in-place to `self` a rotation centered at the point `p`, i.e., the rotation that
    /// lets `p` invariant.
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<N, D>) {
        self.translation.vector -= &p.coords;
        self.append_rotation_mut(r);
        self.translation.vector += &p.coords;
    }

    /// Appends in-place to `self` a rotation centered at the point with coordinates
    /// `self.translation`.
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        let center = Point::from_coordinates(self.translation.vector.clone());
        self.append_rotation_wrt_point_mut(r, &center)
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it hard to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the dummy ZST field).
impl<N: Real, D: DimName, R> Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        D: DimNameAdd<U1>,
        R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        let mut res: MatrixN<N, _> = ::convert_ref(&self.rotation);
        res.fixed_slice_mut::<D, U1>(0, D::dim())
            .copy_from(&self.translation.vector);

        res
    }
}

impl<N: Real, D: DimName, R> Eq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + Eq,
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N: Real, D: DimName, R> PartialEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + PartialEq,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn eq(&self, right: &Isometry<N, D, R>) -> bool {
        self.translation == right.translation && self.rotation == right.rotation
    }
}

impl<N: Real, D: DimName, R> ApproxEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + ApproxEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

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
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.translation
            .relative_eq(&other.translation, epsilon, max_relative)
            && self.rotation
                .relative_eq(&other.rotation, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.translation
            .ulps_eq(&other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(&other.rotation, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N: Real + fmt::Display, D: DimName, R> fmt::Display for Isometry<N, D, R>
where
    R: fmt::Display,
    DefaultAllocator: Allocator<N, D> + Allocator<usize, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "Isometry {{"));
        try!(write!(f, "{:.*}", precision, self.translation));
        try!(write!(f, "{:.*}", precision, self.rotation));
        writeln!(f, "}}")
    }
}
