use std::fmt;
use std::marker::PhantomData;
use approx::ApproxEq;

use alga::general::{Real, SubsetOf};
use alga::linear::Rotation;

use core::{Scalar, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::{Storage, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use geometry::{TranslationBase, PointBase};

/// A direct isometry, i.e., a rotation followed by a translation.
#[repr(C)]
#[derive(Hash, Debug, Clone, Copy)]
pub struct IsometryBase<N: Scalar, D: DimName, S, R> {
    pub rotation:    R,
    pub translation: TranslationBase<N, D, S>,
    // One private field just to prevent explicit construction.
    _noconstruct:    PhantomData<N>
}

impl<N, D: DimName, S, R> IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new isometry from its rotational and translational parts.
    #[inline]
    pub fn from_parts(translation: TranslationBase<N, D, S>, rotation: R) -> IsometryBase<N, D, S, R> {
        IsometryBase {
            rotation:     rotation,
            translation:  translation,
            _noconstruct: PhantomData
        }
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> IsometryBase<N, D, S, R> {
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
    pub fn append_translation_mut(&mut self, t: &TranslationBase<N, D, S>) {
        self.translation.vector += &t.vector
    }

    /// Appends to `self` the given rotation in-place.
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.rotation           = self.rotation.append_rotation(&r);
        self.translation.vector = r.transform_vector(&self.translation.vector);
    }

    /// Appends in-place to `self` a rotation centered at the point `p`, i.e., the rotation that
    /// lets `p` invariant.
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &PointBase<N, D, S>) {
        self.translation.vector -= &p.coords;
        self.append_rotation_mut(r);
        self.translation.vector += &p.coords;
    }

    /// Appends in-place to `self` a rotation centered at the point with coordinates
    /// `self.translation`.
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        let center = PointBase::from_coordinates(self.translation.vector.clone());
        self.append_rotation_wrt_point_mut(r, &center)
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it hard to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the dummy ZST field).
impl<N, D: DimName, S, R> IsometryBase<N, D, S, R>
    where N: Scalar,
          S: Storage<N, D, U1> {
    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>
        where D: DimNameAdd<U1>,
              R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>>,
              S::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
        let mut res: OwnedSquareMatrix<N, _, S::Alloc> = ::convert_ref(&self.rotation);
        res.fixed_slice_mut::<D, U1>(0, D::dim()).copy_from(&self.translation.vector);

        res
    }
}


impl<N, D: DimName, S, R> Eq for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + Eq,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
}

impl<N, D: DimName, S, R> PartialEq for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + PartialEq,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn eq(&self, right: &IsometryBase<N, D, S, R>) -> bool {
        self.translation == right.translation &&
        self.rotation    == right.rotation
    }
}

impl<N, D: DimName, S, R> ApproxEq for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + ApproxEq<Epsilon = N::Epsilon>,
          S::Alloc: OwnedAllocator<N, D, U1, S>,
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
        self.translation.relative_eq(&other.translation, epsilon, max_relative) &&
        self.rotation.relative_eq(&other.rotation, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.translation.ulps_eq(&other.translation, epsilon, max_ulps) &&
        self.rotation.ulps_eq(&other.rotation, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName, S, R> fmt::Display for IsometryBase<N, D, S, R>
    where N: Real + fmt::Display,
          S: OwnedStorage<N, D, U1>,
          R: fmt::Display,
          S::Alloc: OwnedAllocator<N, D, U1, S> + Allocator<usize, D, U1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "IsometryBase {{"));
        try!(write!(f, "{:.*}", precision, self.translation));
        try!(write!(f, "{:.*}", precision, self.rotation));
        writeln!(f, "}}")
    }
}


//         /*
//          *
//          * Absolute
//          *
//          */
//         impl<N: Absolute> Absolute for $t<N> {
//             type AbsoluteValue = $submatrix<N::AbsoluteValue>;
//
//             #[inline]
//             fn abs(m: &$t<N>) -> $submatrix<N::AbsoluteValue> {
//                 Absolute::abs(&m.submatrix)
//             }
//         }
