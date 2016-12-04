use std::fmt;
use approx::ApproxEq;

use alga::general::{ClosedMul, Real, SubsetOf};
use alga::linear::Rotation;

use core::{Scalar, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::{Storage, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use geometry::{PointBase, TranslationBase, IsometryBase};

/// A similarity, i.e., an uniform scaling, followed by a rotation, followed by a translation.
#[repr(C)]
#[derive(Hash, Debug, Clone, Copy)]
pub struct SimilarityBase<N: Scalar, D: DimName, S, R> {
    pub isometry: IsometryBase<N, D, S, R>,
    scaling:      N
}

impl<N, D: DimName, S, R> SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new similarity from its rotational and translational parts.
    #[inline]
    pub fn from_parts(translation: TranslationBase<N, D, S>, rotation: R, scaling: N) -> SimilarityBase<N, D, S, R> {
        SimilarityBase::from_isometry(IsometryBase::from_parts(translation, rotation), scaling)
    }

    /// Creates a new similarity from its rotational and translational parts.
    #[inline]
    pub fn from_isometry(isometry: IsometryBase<N, D, S, R>, scaling: N) -> SimilarityBase<N, D, S, R> {
        assert!(!relative_eq!(scaling, N::zero()), "The scaling factor must not be zero.");

        SimilarityBase {
            isometry: isometry,
            scaling:  scaling
        }
    }

    /// Creates a new similarity that applies only a scaling factor.
    #[inline]
    pub fn from_scaling(scaling: N) -> SimilarityBase<N, D, S, R> {
        Self::from_isometry(IsometryBase::identity(), scaling)
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> SimilarityBase<N, D, S, R> {
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
    pub fn scaling(&self) -> N {
        self.scaling
    }

    /// The scaling factor of this similarity transformation.
    #[inline]
    pub fn set_scaling(&mut self, scaling: N) {
        assert!(!relative_eq!(scaling, N::zero()), "The similarity scaling factor must not be zero.");

        self.scaling = scaling;
    }

    /// The similarity transformation that applies a scaling factor `scaling` before `self`.
    #[inline]
    pub fn prepend_scaling(&self, scaling: N) -> Self {
        assert!(!relative_eq!(scaling, N::zero()), "The similarity scaling factor must not be zero.");

        Self::from_isometry(self.isometry.clone(), self.scaling * scaling)
    }

    /// The similarity transformation that applies a scaling factor `scaling` after `self`.
    #[inline]
    pub fn append_scaling(&self, scaling: N) -> Self {
        assert!(!relative_eq!(scaling, N::zero()), "The similarity scaling factor must not be zero.");

        Self::from_parts(
            TranslationBase::from_vector(&self.isometry.translation.vector * scaling),
            self.isometry.rotation.clone(),
            self.scaling * scaling)
    }

    /// Sets `self` to the similarity transformation that applies a scaling factor `scaling` before `self`.
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: N) {
        assert!(!relative_eq!(scaling, N::zero()), "The similarity scaling factor must not be zero.");

        self.scaling *= scaling
    }

    /// Sets `self` to the similarity transformation that applies a scaling factor `scaling` after `self`.
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: N) {
        assert!(!relative_eq!(scaling, N::zero()), "The similarity scaling factor must not be zero.");

        self.isometry.translation.vector *= scaling;
        self.scaling *= scaling;
    }

    /// Appends to `self` the given translation in-place.
    #[inline]
    pub fn append_translation_mut(&mut self, t: &TranslationBase<N, D, S>) {
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
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &PointBase<N, D, S>) {
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
impl<N, D: DimName, S, R> SimilarityBase<N, D, S, R>
    where N: Scalar + ClosedMul,
          S: Storage<N, D, U1> {
    /// Converts this similarity into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>
        where D: DimNameAdd<U1>,
              R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>>,
              S::Alloc: Allocator<N, D, D> +
                        Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
        let mut res = self.isometry.to_homogeneous();

        for e in res.fixed_slice_mut::<D, D>(0, 0).iter_mut() {
            *e *= self.scaling
        }

        res
    }
}


impl<N, D: DimName, S, R> Eq for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + Eq,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
}

impl<N, D: DimName, S, R> PartialEq for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + PartialEq,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn eq(&self, right: &SimilarityBase<N, D, S, R>) -> bool {
        self.isometry == right.isometry && self.scaling == right.scaling
    }
}

impl<N, D: DimName, S, R> ApproxEq for SimilarityBase<N, D, S, R>
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
        self.isometry.relative_eq(&other.isometry, epsilon, max_relative) &&
        self.scaling.relative_eq(&other.scaling, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.isometry.ulps_eq(&other.isometry, epsilon, max_ulps) &&
        self.scaling.ulps_eq(&other.scaling, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName, S, R> fmt::Display for SimilarityBase<N, D, S, R>
    where N: Real + fmt::Display,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>> + fmt::Display,
          S::Alloc: OwnedAllocator<N, D, U1, S> + Allocator<usize, D, U1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "SimilarityBase {{"));
        try!(write!(f, "{:.*}", precision, self.isometry));
        try!(write!(f, "Scaling: {:.*}", precision, self.scaling));
        writeln!(f, "}}")
    }
}

/*
//         /*
//          *
//          * ToHomogeneous
//          *
//          */
//         impl<N: Real> ToHomogeneous<$homogeneous<N>> for $t<N> {
//             #[inline]
//             fn to_homogeneous(&self) -> $homogeneous<N> {
//                 self.vector.to_homogeneous()
//             }
//         }


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
*/
