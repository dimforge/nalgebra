use num::One;
use std::fmt;
use std::cmp::Ordering;
use approx::ApproxEq;

use core::{Scalar, ColumnVector, OwnedColumnVector};
use core::iter::{MatrixIter, MatrixIterMut};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::{Storage, StorageMut, MulStorage};
use core::allocator::{Allocator, SameShapeR};

// XXX Bad name: we can't even add points…
/// The type of the result of the sum of a point with a vector.
pub type PointSum<N, D1, D2, SA> =
    PointBase<N, SameShapeR<D1, D2>,
    <<SA as Storage<N, D1, U1>>::Alloc as Allocator<N, SameShapeR<D1, D2>, U1>>::Buffer>;

/// The type of the result of the multiplication of a point by a matrix.
pub type PointMul<N, R1, C1, SA> = PointBase<N, R1, MulStorage<N, R1, C1, U1, SA>>;

/// A point with an owned storage.
pub type OwnedPoint<N, D, A> = PointBase<N, D, <A as Allocator<N, D, U1>>::Buffer>;

/// A point in a n-dimensional euclidean space.
#[repr(C)]
#[derive(Hash, Debug, Serialize, Deserialize)]
pub struct PointBase<N: Scalar, D: DimName, S: Storage<N, D, U1>> {
    /// The coordinates of this point, i.e., the shift from the origin.
    pub coords: ColumnVector<N, D, S>
}

impl<N, D, S> Copy for PointBase<N, D, S>
    where N: Scalar,
          D: DimName,
          S: Storage<N, D, U1> + Copy { }

impl<N, D, S> Clone for PointBase<N, D, S>
    where N: Scalar,
          D: DimName,
          S: Storage<N, D, U1> + Clone {
    #[inline]
    fn clone(&self) -> Self {
        PointBase::from_coordinates(self.coords.clone())
    }
}

impl<N: Scalar, D: DimName, S: Storage<N, D, U1>> PointBase<N, D, S> {
    /// Creates a new point with the given coordinates.
    #[inline]
    pub fn from_coordinates(coords: ColumnVector<N, D, S>) -> PointBase<N, D, S> {
        PointBase {
            coords: coords
        }
    }
}

impl<N: Scalar, D: DimName, S: Storage<N, D, U1>> PointBase<N, D, S> {
    /// Moves this point into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OwnedPoint<N, D, S::Alloc> {
        PointBase::from_coordinates(self.coords.into_owned())
    }

    /// Clones this point into one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OwnedPoint<N, D, S::Alloc> {
        PointBase::from_coordinates(self.coords.clone_owned())
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
    pub fn iter(&self) -> MatrixIter<N, D, U1, S> {
        self.coords.iter()
    }

    /// Gets a reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &N {
        self.coords.get_unchecked(i, 0)
    }


    /// Converts this point into a vector in homogeneous coordinates, i.e., appends a `1` at the
    /// end of it.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedColumnVector<N, DimNameSum<D, U1>, S::Alloc>
        where N: One,
              D: DimNameAdd<U1>,
              S::Alloc: Allocator<N, DimNameSum<D, U1>, U1> {

        let mut res = unsafe { OwnedColumnVector::<N, _, S::Alloc>::new_uninitialized() };
        res.fixed_slice_mut::<D, U1>(0, 0).copy_from(&self.coords);
        res[(D::dim(), 0)] = N::one();

        res
    }
}

impl<N: Scalar, D: DimName, S: StorageMut<N, D, U1>> PointBase<N, D, S> {
    /// Mutably iterates through this point coordinates.
    #[inline]
    pub fn iter_mut(&mut self) -> MatrixIterMut<N, D, U1, S> {
        self.coords.iter_mut()
    }

    /// Gets a mutable reference to i-th element of this point without bound-checking.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.coords.get_unchecked_mut(i, 0)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, i1: usize, i2: usize) {
        self.coords.swap_unchecked((i1, 0), (i2, 0))
    }
}

impl<N, D: DimName, S> ApproxEq for PointBase<N, D, S>
    where N: Scalar + ApproxEq,
          S: Storage<N, D, U1>,
          N::Epsilon: Copy {
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
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.coords.relative_eq(&other.coords, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.coords.ulps_eq(&other.coords, epsilon, max_ulps)
    }
}

impl<N, D: DimName, S> Eq for PointBase<N, D, S>
    where N: Scalar + Eq,
          S: Storage<N, D, U1> { }

impl<N, D: DimName, S> PartialEq for PointBase<N, D, S>
    where N: Scalar,
          S: Storage<N, D, U1> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<N, D: DimName, S> PartialOrd for PointBase<N, D, S>
    where N: Scalar + PartialOrd,
          S: Storage<N, D, U1> {
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
impl<N, D: DimName, S> fmt::Display for PointBase<N, D, S>
    where N: Scalar + fmt::Display,
          S: Storage<N, D, U1> {
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
