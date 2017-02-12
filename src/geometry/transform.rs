use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use approx::ApproxEq;

use alga::general::Field;

use core::{Scalar, SquareMatrix, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use core::storage::{Storage, StorageMut};
use core::allocator::Allocator;

/// Trait implemented by phantom types identifying the projective transformation type.
///
/// NOTE: this trait is not intended to be implementable outside of the `nalgebra` crate.
pub trait TCategory: Any + Debug + Copy + PartialEq + Send {
    /// Indicates whether a `Transform` with the category `Self` has a bottom-row different from
    /// `0 0 .. 1`.
    #[inline]
    fn has_normalizer() -> bool {
        true
    }

    /// Checks that the given matrix is a valid homogeneous representation of an element of the
    /// category `Self`.
    fn check_homogeneous_invariants<N, D, S>(mat: &SquareMatrix<N, D, S>) -> bool
        where N: Scalar + Field + ApproxEq,
              D: DimName,
              S: Storage<N, D, D>,
              N::Epsilon: Copy;
}

/// Traits that gives the `Transform` category that is compatible with the result of the
/// multiplication of transformations with categories `Self` and `Other`.
pub trait TCategoryMul<Other: TCategory>: TCategory {
    /// The transform category that results from the multiplication of a `Transform<Self>` to a
    /// `Transform<Other>`. This is usually equal to `Self` or `Other`, whichever is the most
    /// general category.
    type Representative: TCategory;
}

/// Indicates that `Self` is a more general `Transform` category than `Other`.
pub trait SuperTCategoryOf<Other: TCategory>: TCategory { }

/// Indicates that `Self` is a more specific `Transform` category than `Other`.
///
/// Automatically implemented based on `SuperTCategoryOf`.
pub trait SubTCategoryOf<Other: TCategory>: TCategory { }
impl<T1, T2> SubTCategoryOf<T2> for T1
where T1: TCategory,
      T2: SuperTCategoryOf<T1> {
}

/// Tag representing the most general (not necessarily inversible) `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TGeneral { }

/// Tag representing the most general inversible `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TProjective { }

/// Tag representing an affine `Transform`. Its bottom-row is equal to `(0, 0 ... 0, 1)`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TAffine { }

impl TCategory for TGeneral {
    #[inline]
    fn check_homogeneous_invariants<N, D, S>(_: &SquareMatrix<N, D, S>) -> bool
        where N: Scalar + Field + ApproxEq,
              D: DimName,
              S: Storage<N, D, D>,
              N::Epsilon: Copy {
        true
    }
}

impl TCategory for TProjective {
    #[inline]
    fn check_homogeneous_invariants<N, D, S>(mat: &SquareMatrix<N, D, S>) -> bool
        where N: Scalar + Field + ApproxEq,
              D: DimName,
              S: Storage<N, D, D>,
              N::Epsilon: Copy {
        mat.is_invertible()
    }
}

impl TCategory for TAffine {
    #[inline]
    fn has_normalizer() -> bool {
        false
    }

    #[inline]
    fn check_homogeneous_invariants<N, D, S>(mat: &SquareMatrix<N, D, S>) -> bool
        where N: Scalar + Field + ApproxEq,
              D: DimName,
              S: Storage<N, D, D>,
              N::Epsilon: Copy {
        mat.is_invertible()                   &&
        mat[(D::dim(), D::dim())] == N::one() &&
        (0 .. D::dim()).all(|i| mat[(D::dim(), i)].is_zero())
    }
}

macro_rules! category_mul_impl(
    ($($a: ident * $b: ident => $c: ty);* $(;)*) => {$(
        impl TCategoryMul<$a> for $b {
            type Representative = $c;
        }
    )*}
);

// We require stability uppon multiplication.
impl<T: TCategory> TCategoryMul<T> for T {
    type Representative = T;
}

category_mul_impl!(
    // TGeneral * TGeneral     => TGeneral;
    TGeneral * TProjective  => TGeneral;
    TGeneral * TAffine      => TGeneral;

    TProjective * TGeneral     => TGeneral;
    // TProjective * TProjective  => TProjective;
    TProjective * TAffine      => TProjective;

    TAffine * TGeneral     => TGeneral;
    TAffine * TProjective  => TProjective;
    // TAffine * TAffine      => TAffine;
);

macro_rules! super_tcategory_impl(
    ($($a: ident >= $b: ident);* $(;)*) => {$(
        impl SuperTCategoryOf<$b> for $a { }
    )*}
);

impl<T: TCategory> SuperTCategoryOf<T> for T { }

super_tcategory_impl!(
    TGeneral >= TProjective;
    TGeneral >= TAffine;
    TProjective >= TAffine;
);


/// A transformation matrix that owns its data.
pub type OwnedTransform<N, D, A, C>
    = TransformBase<N, D, <A as Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>>::Buffer, C>;


/// A transformation matrix in homogeneous coordinates.
///
/// It is stored as a matrix with dimensions `(D + 1, D + 1)`, e.g., it stores a 4x4 matrix for a
/// 3D transformation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)] // FIXME: Hash
pub struct TransformBase<N: Scalar, D: DimNameAdd<U1>, S, C: TCategory> {
    matrix:   SquareMatrix<N, DimNameSum<D, U1>, S>,

    #[serde(skip_serializing, skip_deserializing)]
    _phantom: PhantomData<C>
}

// XXX: for some reasons, implementing Clone and Copy manually causes an ICE…

impl<N, D, S, C: TCategory> Eq for TransformBase<N, D, S, C>
    where N: Scalar + Eq,
          D: DimNameAdd<U1>,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> { }

impl<N, D, S, C: TCategory> PartialEq for TransformBase<N, D, S, C>
    where N: Scalar,
          D: DimNameAdd<U1>,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, D, S, C: TCategory> TransformBase<N, D, S, C>
    where N: Scalar,
          D: DimNameAdd<U1>,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    /// Creates a new transformation from the given homogeneous matrix. The transformation category
    /// of `Self` is not checked to be verified by the given matrix.
    #[inline]
    pub fn from_matrix_unchecked(matrix: SquareMatrix<N, DimNameSum<D, U1>, S>) -> Self {
        TransformBase {
            matrix:   matrix,
            _phantom: PhantomData
        }
    }

    /// Moves this transform into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OwnedTransform<N, D, S::Alloc, C> {
        TransformBase::from_matrix_unchecked(self.matrix.into_owned())
    }

    /// Clones this transform into one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OwnedTransform<N, D, S::Alloc, C> {
        TransformBase::from_matrix_unchecked(self.matrix.clone_owned())
    }

    /// The underlying matrix.
    #[inline]
    pub fn unwrap(self) -> SquareMatrix<N, DimNameSum<D, U1>, S> {
        self.matrix
    }

    /// A reference to the underlynig matrix.
    #[inline]
    pub fn matrix(&self) -> &SquareMatrix<N, DimNameSum<D, U1>, S> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix.
    ///
    /// It is `_unchecked` because direct modifications of this matrix may break invariants
    /// identified by this transformation category.
    #[inline]
    pub fn matrix_mut_unchecked(&mut self) -> &mut SquareMatrix<N, DimNameSum<D, U1>, S> {
        &mut self.matrix
    }

    /// Sets the category of this transform.
    ///
    /// This can be done only if the new category is more general than the current one, e.g., a
    /// transform with category `TProjective` cannot be converted to a transform with category
    /// `TAffine` because not all projective transformations are affine (the other way-round is
    /// valid though).
    #[inline]
    pub fn set_category<CNew: SuperTCategoryOf<C>>(self) -> TransformBase<N, D, S, CNew> {
        TransformBase::from_matrix_unchecked(self.matrix)
    }

    /// Converts this transform into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc> {
        self.matrix().clone_owned()
    }
}

impl<N, D, S, C> TransformBase<N, D, S, C>
    where N: Scalar + Field + ApproxEq,
          D: DimNameAdd<U1>,
          C: TCategory,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    /// Attempts to invert this transformation. You may use `.inverse` instead of this
    /// transformation has a subcategory of `TProjective`.
    #[inline]
    pub fn try_inverse(self) -> Option<OwnedTransform<N, D, S::Alloc, C>> {
        if let Some(m) = self.matrix.try_inverse() {
            Some(TransformBase::from_matrix_unchecked(m))
        }
        else {
            None
        }
    }

    /// Inverts this transformation. Use `.try_inverse` if this transform has the `TGeneral`
    /// category (it may not be invertible).
    #[inline]
    pub fn inverse(self) -> OwnedTransform<N, D, S::Alloc, C>
        where C: SubTCategoryOf<TProjective> {
        // FIXME: specialize for TAffine?
        TransformBase::from_matrix_unchecked(self.matrix.try_inverse().unwrap())
    }
}

impl<N, D, S, C> TransformBase<N, D, S, C>
    where N: Scalar + Field + ApproxEq,
          D: DimNameAdd<U1>,
          C: TCategory,
          S: StorageMut<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    /// Attempts to invert this transformation in-place. You may use `.inverse_mut` instead of this
    /// transformation has a subcategory of `TProjective`.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        self.matrix.try_inverse_mut()
    }

    /// Inverts this transformation in-place. Use `.try_inverse_mut` if this transform has the
    /// `TGeneral` category (it may not be invertible).
    #[inline]
    pub fn inverse_mut(&mut self)
        where C: SubTCategoryOf<TProjective> {
        let _ = self.matrix.try_inverse_mut();
    }
}

impl<N, D, S> TransformBase<N, D, S, TGeneral>
    where N: Scalar,
          D: DimNameAdd<U1>,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    /// A mutable reference to underlying matrix. Use `.matrix_mut_unchecked` instead if this
    /// transformation category is not `TGeneral`.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut SquareMatrix<N, DimNameSum<D, U1>, S> {
        self.matrix_mut_unchecked()
    }
}
