use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::One;
use std::cmp::Ordering;
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use base::allocator::Allocator;
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use base::iter::{MatrixIter, MatrixIterMut};
use base::{DefaultAllocator, Scalar, VectorN};

/// A point in a n-dimensional euclidean space.
#[repr(C)]
#[derive(Debug)]
pub struct Point<N: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// The coordinates of this point, i.e., the shift from the origin.
    pub coords: VectorN<N, D>,
}

impl<N: Scalar + hash::Hash, D: DimName + hash::Hash> hash::Hash for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<N: Scalar, D: DimName> Copy for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Copy,
{
}

impl<N: Scalar, D: DimName> Clone for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Point::from_coordinates(self.coords.clone())
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Scalar, D: DimName> Serialize for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Scalar, D: DimName> Deserialize<'a> for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = VectorN::<N, D>::deserialize(deserializer)?;

        Ok(Point::from_coordinates(coords))
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D> Abomonation for Point<N, D>
where
    N: Scalar,
    D: DimName,
    VectorN<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D>,
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

impl<N: Scalar, D: DimName> Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Clones this point into one that owns its data.
    #[inline]
    pub fn clone(&self) -> Point<N, D> {
        Point::from_coordinates(self.coords.clone_owned())
    }

    /// Converts this point into a vector in homogeneous coordinates, i.e., appends a `1` at the
    /// end of it.
    #[inline]
    pub fn to_homogeneous(&self) -> VectorN<N, DimNameSum<D, U1>>
    where
        N: One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>>,
    {
        let mut res = unsafe { VectorN::<_, DimNameSum<D, U1>>::new_uninitialized() };
        res.fixed_slice_mut::<D, U1>(0, 0).copy_from(&self.coords);
        res[(D::dim(), 0)] = N::one();

        res
    }

    /// Creates a new point with the given coordinates.
    #[inline]
    pub fn from_coordinates(coords: VectorN<N, D>) -> Point<N, D> {
        Point { coords: coords }
    }

    /// The dimension of this point.
    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// The stride of this point. This is the number of buffer element separating each component of
    /// this point.
    #[inline]
    pub fn stride(&self) -> usize {
        self.coords.strides().0
    }

    /// Iterates through this point coordinates.
    #[inline]
    pub fn iter(&self) -> MatrixIter<N, D, U1, <DefaultAllocator as Allocator<N, D>>::Buffer> {
        self.coords.iter()
    }

    /// Gets a reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &N {
        self.coords.vget_unchecked(i)
    }

    /// Mutably iterates through this point coordinates.
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> MatrixIterMut<N, D, U1, <DefaultAllocator as Allocator<N, D>>::Buffer> {
        self.coords.iter_mut()
    }

    /// Gets a mutable reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.coords.vget_unchecked_mut(i)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, i1: usize, i2: usize) {
        self.coords.swap_unchecked((i1, 0), (i2, 0))
    }
}

impl<N: Scalar + AbsDiffEq, D: DimName> AbsDiffEq for Point<N, D>
where
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
        self.coords.abs_diff_eq(&other.coords, epsilon)
    }
}

impl<N: Scalar + RelativeEq, D: DimName> RelativeEq for Point<N, D>
where
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
        self.coords
            .relative_eq(&other.coords, epsilon, max_relative)
    }
}

impl<N: Scalar + UlpsEq, D: DimName> UlpsEq for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.coords.ulps_eq(&other.coords, epsilon, max_ulps)
    }
}

impl<N: Scalar + Eq, D: DimName> Eq for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N: Scalar, D: DimName> PartialEq for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<N: Scalar + PartialOrd, D: DimName> PartialOrd for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coords.partial_cmp(&other.coords)
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        self.coords.lt(&right.coords)
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        self.coords.le(&right.coords)
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        self.coords.gt(&right.coords)
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        self.coords.ge(&right.coords)
    }
}

/*
 *
 * Display
 *
 */
impl<N: Scalar + fmt::Display, D: DimName> fmt::Display for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        let mut it = self.coords.iter();

        try!(write!(f, "{}", *it.next().unwrap()));

        for comp in it {
            try!(write!(f, ", {}", *comp));
        }

        write!(f, "}}")
    }
}
