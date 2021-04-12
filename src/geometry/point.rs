use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::One;
use std::cmp::Ordering;
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use simba::simd::SimdPartialOrd;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::{Const, DefaultAllocator, OVector, SVector, Scalar};

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
pub struct Point<T, const D: usize> {
    /// The coordinates of this point, i.e., the shift from the origin.
    pub coords: SVector<T, D>,
}

impl<T: Scalar + hash::Hash, const D: usize> hash::Hash for Point<T, D> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<T: Scalar + Copy, const D: usize> Copy for Point<T, D> {}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, const D: usize> bytemuck::Zeroable for Point<T, D> where
    SVector<T, D>: bytemuck::Zeroable
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, const D: usize> bytemuck::Pod for Point<T, D>
where
    T: Copy,
    SVector<T, D>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar + Serialize, const D: usize> Serialize for Point<T, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar + Deserialize<'a>, const D: usize> Deserialize<'a> for Point<T, D> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = SVector::<T, D>::deserialize(deserializer)?;

        Ok(Self::from(coords))
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<T, const D: usize> Abomonation for Point<T, D>
where
    T: Scalar,
    SVector<T, D>: Abomonation,
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

impl<T: Scalar, const D: usize> Point<T, D> {
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
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, f: F) -> Point<T2, D> {
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
    pub fn apply<F: FnMut(T) -> T>(&mut self, f: F) {
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
    pub fn to_homogeneous(&self) -> OVector<T, DimNameSum<Const<D>, U1>>
    where
        T: One,
        Const<D>: DimNameAdd<U1>,
        DefaultAllocator: Allocator<T, DimNameSum<Const<D>, U1>>,
    {
        let mut res = unsafe {
            crate::unimplemented_or_uninitialized_generic!(
                <DimNameSum<Const<D>, U1> as DimName>::name(),
                Const::<1>
            )
        };
        res.fixed_slice_mut::<D, 1>(0, 0).copy_from(&self.coords);
        res[(D, 0)] = T::one();

        res
    }

    /// Creates a new point with the given coordinates.
    #[deprecated(note = "Use Point::from(vector) instead.")]
    #[inline]
    pub fn from_coordinates(coords: SVector<T, D>) -> Self {
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
    pub fn iter(
        &self,
    ) -> MatrixIter<T, Const<D>, Const<1>, <DefaultAllocator as Allocator<T, Const<D>>>::Buffer>
    {
        self.coords.iter()
    }

    /// Gets a reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
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
    ) -> MatrixIterMut<T, Const<D>, Const<1>, <DefaultAllocator as Allocator<T, Const<D>>>::Buffer>
    {
        self.coords.iter_mut()
    }

    /// Gets a mutable reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        self.coords.vget_unchecked_mut(i)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, i1: usize, i2: usize) {
        self.coords.swap_unchecked((i1, 0), (i2, 0))
    }
}

impl<T: Scalar + AbsDiffEq, const D: usize> AbsDiffEq for Point<T, D>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.coords.abs_diff_eq(&other.coords, epsilon)
    }
}

impl<T: Scalar + RelativeEq, const D: usize> RelativeEq for Point<T, D>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
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

impl<T: Scalar + UlpsEq, const D: usize> UlpsEq for Point<T, D>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.coords.ulps_eq(&other.coords, epsilon, max_ulps)
    }
}

impl<T: Scalar + Eq, const D: usize> Eq for Point<T, D> {}

impl<T: Scalar, const D: usize> PartialEq for Point<T, D> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<T: Scalar + PartialOrd, const D: usize> PartialOrd for Point<T, D> {
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
impl<T: Scalar + SimdPartialOrd, const D: usize> Point<T, D> {
    /// Computes the infimum (aka. componentwise min) of two points.
    #[inline]
    pub fn inf(&self, other: &Self) -> Point<T, D> {
        self.coords.inf(&other.coords).into()
    }

    /// Computes the supremum (aka. componentwise max) of two points.
    #[inline]
    pub fn sup(&self, other: &Self) -> Point<T, D> {
        self.coords.sup(&other.coords).into()
    }

    /// Computes the (infimum, supremum) of two points.
    #[inline]
    pub fn inf_sup(&self, other: &Self) -> (Point<T, D>, Point<T, D>) {
        let (inf, sup) = self.coords.inf_sup(&other.coords);
        (inf.into(), sup.into())
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, const D: usize> fmt::Display for Point<T, D> {
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
