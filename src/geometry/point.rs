use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::One;
use std::cmp::Ordering;
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use simba::simd::SimdPartialOrd;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::{DefaultAllocator, Scalar, VectorN};

/// A point in an euclidean space.
///
/// The difference between a point and a vector is only semantic. See [the user guide](https://www.nalgebra.org/points_and_transformations/)
/// for details on the distinction. The most notable difference that vectors ignore translations.
/// In particular, an [`Isometry2`](crate::Isometry2) or [`Isometry3`](crate::Isometry3) will
/// transform points by applying a rotation and a translation on them. However, these isometries
/// will only apply rotations to vectors (when doing `isometry * vector`, the translation part of
/// the isometry is ignored).
///
/// # Construction
/// * [From individual components <span style="float:right;">`new`…</span>](#construction-from-individual-components)
/// * [Swizzling <span style="float:right;">`xx`, `yxz`…</span>](#swizzling)
/// * [Other construction methods <span style="float:right;">`origin`, `from_slice`, `from_homogeneous`…</span>](#other-construction-methods)
///
/// # Transformation
/// Transforming a point by an [Isometry](crate::Isometry), [rotation](crate::Rotation), etc. can be
/// achieved by multiplication, e.g., `isometry * point` or `rotation * point`. Some of these transformation
/// may have some other methods, e.g., `isometry.inverse_transform_point(&point)`. See the documentation
/// of said transformations for details.
#[repr(C)]
#[derive(Debug, Clone)]
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

impl<N: Scalar + Copy, D: DimName> Copy for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Copy,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<N: Scalar, D: DimName> bytemuck::Zeroable for Point<N, D>
where
    VectorN<N, D>: bytemuck::Zeroable,
    DefaultAllocator: Allocator<N, D>,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<N: Scalar, D: DimName> bytemuck::Pod for Point<N, D>
where
    N: Copy,
    VectorN<N, D>: bytemuck::Pod,
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Copy,
{
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

        Ok(Self::from(coords))
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
    /// Returns a point containing the result of `f` applied to each of its entries.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let p = Point2::new(1.0, 2.0);
    /// assert_eq!(p.map(|e| e * 10.0), Point2::new(10.0, 20.0));
    ///
    /// // This works in any dimension.
    /// let p = Point3::new(1.1, 2.1, 3.1);
    /// assert_eq!(p.map(|e| e as u32), Point3::new(1, 2, 3));
    /// ```
    #[inline]
    pub fn map<N2: Scalar, F: FnMut(N) -> N2>(&self, f: F) -> Point<N2, D>
    where
        DefaultAllocator: Allocator<N2, D>,
    {
        self.coords.map(f).into()
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on it.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let mut p = Point2::new(1.0, 2.0);
    /// p.apply(|e| e * 10.0);
    /// assert_eq!(p, Point2::new(10.0, 20.0));
    ///
    /// // This works in any dimension.
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    /// p.apply(|e| e * 10.0);
    /// assert_eq!(p, Point3::new(10.0, 20.0, 30.0));
    /// ```
    #[inline]
    pub fn apply<F: FnMut(N) -> N>(&mut self, f: F) {
        self.coords.apply(f)
    }

    /// Converts this point into a vector in homogeneous coordinates, i.e., appends a `1` at the
    /// end of it.
    ///
    /// This is the same as `.into()`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3, Vector3, Vector4};
    /// let p = Point2::new(10.0, 20.0);
    /// assert_eq!(p.to_homogeneous(), Vector3::new(10.0, 20.0, 1.0));
    ///
    /// // This works in any dimension.
    /// let p = Point3::new(10.0, 20.0, 30.0);
    /// assert_eq!(p.to_homogeneous(), Vector4::new(10.0, 20.0, 30.0, 1.0));
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> VectorN<N, DimNameSum<D, U1>>
    where
        N: One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>>,
    {
        let mut res = unsafe {
            crate::unimplemented_or_uninitialized_generic!(
                <DimNameSum<D, U1> as DimName>::name(),
                U1
            )
        };
        res.fixed_slice_mut::<D, U1>(0, 0).copy_from(&self.coords);
        res[(D::dim(), 0)] = N::one();

        res
    }

    /// Creates a new point with the given coordinates.
    #[deprecated(note = "Use Point::from(vector) instead.")]
    #[inline]
    pub fn from_coordinates(coords: VectorN<N, D>) -> Self {
        Self { coords }
    }

    /// The dimension of this point.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let p = Point2::new(1.0, 2.0);
    /// assert_eq!(p.len(), 2);
    ///
    /// // This works in any dimension.
    /// let p = Point3::new(10.0, 20.0, 30.0);
    /// assert_eq!(p.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Returns true if the point contains no elements.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let p = Point2::new(1.0, 2.0);
    /// assert!(!p.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stride of this point. This is the number of buffer element separating each component of
    /// this point.
    #[inline]
    #[deprecated(note = "This methods is no longer significant and will always return 1.")]
    pub fn stride(&self) -> usize {
        self.coords.strides().0
    }

    /// Iterates through this point coordinates.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Point3;
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// let mut it = p.iter().cloned();
    ///
    /// assert_eq!(it.next(), Some(1.0));
    /// assert_eq!(it.next(), Some(2.0));
    /// assert_eq!(it.next(), Some(3.0));
    /// assert_eq!(it.next(), None);
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
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Point3;
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    ///
    /// for e in p.iter_mut() {
    ///     *e *= 10.0;
    /// }
    ///
    /// assert_eq!(p, Point3::new(10.0, 20.0, 30.0));
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

impl<N: Scalar + Eq, D: DimName> Eq for Point<N, D> where DefaultAllocator: Allocator<N, D> {}

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
 * inf/sup
 */
impl<N: Scalar + SimdPartialOrd, D: DimName> Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Computes the infimum (aka. componentwise min) of two points.
    #[inline]
    pub fn inf(&self, other: &Self) -> Point<N, D> {
        self.coords.inf(&other.coords).into()
    }

    /// Computes the supremum (aka. componentwise max) of two points.
    #[inline]
    pub fn sup(&self, other: &Self) -> Point<N, D> {
        self.coords.sup(&other.coords).into()
    }

    /// Computes the (infimum, supremum) of two points.
    #[inline]
    pub fn inf_sup(&self, other: &Self) -> (Point<N, D>, Point<N, D>) {
        let (inf, sup) = self.coords.inf_sup(&other.coords);
        (inf.into(), sup.into())
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
        write!(f, "{{")?;

        let mut it = self.coords.iter();

        write!(f, "{}", *it.next().unwrap())?;

        for comp in it {
            write!(f, ", {}", *comp)?;
        }

        write!(f, "}}")
    }
}
