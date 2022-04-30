use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::One;
use std::cmp::Ordering;
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use simba::simd::SimdPartialOrd;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::{Const, DefaultAllocator, OVector, Scalar};
use std::mem::MaybeUninit;

/// A point in an euclidean space.
///
/// The difference between a point and a vector is only semantic. See [the user guide](https://www.nalgebra.org/docs/user_guide/points_and_transformations)
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
#[derive(Clone)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
pub struct OPoint<T: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// The coordinates of this point, i.e., the shift from the origin.
    pub coords: OVector<T, D>,
}

impl<T: Scalar + fmt::Debug, D: DimName> fmt::Debug for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.coords.as_slice().fmt(formatter)
    }
}

impl<T: Scalar + hash::Hash, D: DimName> hash::Hash for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<T: Scalar + Copy, D: DimName> Copy for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    OVector<T, D>: Copy,
{
}

#[cfg(feature = "cuda")]
unsafe impl<T: Scalar + cust_core::DeviceCopy, D: DimName> cust_core::DeviceCopy for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    OVector<T, D>: cust_core::DeviceCopy,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, D: DimName> bytemuck::Zeroable for OPoint<T, D>
where
    OVector<T, D>: bytemuck::Zeroable,
    DefaultAllocator: Allocator<T, D>,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, D: DimName> bytemuck::Pod for OPoint<T, D>
where
    T: Copy,
    OVector<T, D>: bytemuck::Pod,
    DefaultAllocator: Allocator<T, D>,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar, D: DimName> Serialize for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar, D: DimName> Deserialize<'a> for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = OVector::<T, D>::deserialize(deserializer)?;

        Ok(Self::from(coords))
    }
}

impl<T: Scalar, D: DimName> OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
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
    #[must_use]
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, f: F) -> OPoint<T2, D>
    where
        DefaultAllocator: Allocator<T2, D>,
    {
        self.coords.map(f).into()
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on it.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let mut p = Point2::new(1.0, 2.0);
    /// p.apply(|e| *e = *e * 10.0);
    /// assert_eq!(p, Point2::new(10.0, 20.0));
    ///
    /// // This works in any dimension.
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    /// p.apply(|e| *e = *e * 10.0);
    /// assert_eq!(p, Point3::new(10.0, 20.0, 30.0));
    /// ```
    #[inline]
    pub fn apply<F: FnMut(&mut T)>(&mut self, f: F) {
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
    #[must_use]
    pub fn to_homogeneous(&self) -> OVector<T, DimNameSum<D, U1>>
    where
        T: One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<T, DimNameSum<D, U1>>,
    {
        // TODO: this is mostly a copy-past from Vector::push.
        //       But we can’t use Vector::push because of the DimAdd bound
        //       (which we don’t use because we use DimNameAdd).
        //       We should find a way to re-use Vector::push.
        let len = self.len();
        let mut res = crate::Matrix::uninit(DimNameSum::<D, U1>::name(), Const::<1>);
        // This is basically a copy_from except that we warp the copied
        // values into MaybeUninit.
        res.generic_slice_mut((0, 0), self.coords.shape_generic())
            .zip_apply(&self.coords, |out, e| *out = MaybeUninit::new(e));
        res[(len, 0)] = MaybeUninit::new(T::one());

        // Safety: res has been fully initialized.
        unsafe { res.assume_init() }
    }

    /// Creates a new point with the given coordinates.
    #[deprecated(note = "Use Point::from(vector) instead.")]
    #[inline]
    pub fn from_coordinates(coords: OVector<T, D>) -> Self {
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
    #[must_use]
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
    #[must_use]
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
    /// ```
    #[inline]
    pub fn iter(
        &self,
    ) -> MatrixIter<'_, T, D, Const<1>, <DefaultAllocator as Allocator<T, D>>::Buffer> {
        self.coords.iter()
    }

    /// Gets a reference to i-th element of this point without bound-checking.
    #[inline]
    #[must_use]
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
    /// ```
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> MatrixIterMut<'_, T, D, Const<1>, <DefaultAllocator as Allocator<T, D>>::Buffer> {
        self.coords.iter_mut()
    }

    /// Gets a mutable reference to i-th element of this point without bound-checking.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        self.coords.vget_unchecked_mut(i)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, i1: usize, i2: usize) {
        self.coords.swap_unchecked((i1, 0), (i2, 0))
    }
}

impl<T: Scalar + AbsDiffEq, D: DimName> AbsDiffEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<T, D>,
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

impl<T: Scalar + RelativeEq, D: DimName> RelativeEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<T, D>,
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

impl<T: Scalar + UlpsEq, D: DimName> UlpsEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<T, D>,
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

impl<T: Scalar + Eq, D: DimName> Eq for OPoint<T, D> where DefaultAllocator: Allocator<T, D> {}

impl<T: Scalar, D: DimName> PartialEq for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<T: Scalar + PartialOrd, D: DimName> PartialOrd for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
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
impl<T: Scalar + SimdPartialOrd, D: DimName> OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Computes the infimum (aka. componentwise min) of two points.
    #[inline]
    #[must_use]
    pub fn inf(&self, other: &Self) -> OPoint<T, D> {
        self.coords.inf(&other.coords).into()
    }

    /// Computes the supremum (aka. componentwise max) of two points.
    #[inline]
    #[must_use]
    pub fn sup(&self, other: &Self) -> OPoint<T, D> {
        self.coords.sup(&other.coords).into()
    }

    /// Computes the (infimum, supremum) of two points.
    #[inline]
    #[must_use]
    pub fn inf_sup(&self, other: &Self) -> (OPoint<T, D>, OPoint<T, D>) {
        let (inf, sup) = self.coords.inf_sup(&other.coords);
        (inf.into(), sup.into())
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, D: DimName> fmt::Display for OPoint<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;

        let mut it = self.coords.iter();

        write!(f, "{}", *it.next().unwrap())?;

        for comp in it {
            write!(f, ", {}", *comp)?;
        }

        write!(f, "}}")
    }
}
