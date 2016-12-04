use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Real, Inverse, Multiplicative, Identity};
use alga::linear::{Transformation, AffineTransformation, Rotation, Similarity, ProjectiveTransformation};

use core::ColumnVector;
use core::dimension::{DimName, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{SimilarityBase, TranslationBase, PointBase};


/*
 *
 * Algebraic structures.
 *
 */
impl<N, D: DimName, S, R> Identity<Multiplicative> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S, R> Inverse<Multiplicative> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N, D: DimName, S, R> AbstractMagma<Multiplicative> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimName, S, R> $marker<$operator> for SimilarityBase<N, D, S, R>
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
                  R: Rotation<PointBase<N, D, S>>,
                  S::Alloc: OwnedAllocator<N, D, U1, S> { }
    )*}
);

impl_multiplicative_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

/*
 *
 * Transformation groups.
 *
 */
impl<N, D: DimName, S, R> Transformation<PointBase<N, D, S>> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        self * v
    }
}

impl<N, D: DimName, S, R> ProjectiveTransformation<PointBase<N, D, S>> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        self.isometry.inverse_transform_point(pt) / self.scaling()
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        self.isometry.inverse_transform_vector(v) / self.scaling()
    }
}

impl<N, D: DimName, S, R> AffineTransformation<PointBase<N, D, S>> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type NonUniformScaling = N;
    type Rotation          = R;
    type Translation       = TranslationBase<N, D, S>;

    #[inline]
    fn decompose(&self) -> (TranslationBase<N, D, S>, R, N, R) {
        (self.isometry.translation.clone(), self.isometry.rotation.clone(), self.scaling(), R::identity())
    }

    #[inline]
    fn append_translation(&self, t: &Self::Translation) -> Self {
        t * self
    }

    #[inline]
    fn prepend_translation(&self, t: &Self::Translation) -> Self {
        self * t
    }

    #[inline]
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        SimilarityBase::from_isometry(self.isometry.append_rotation(r), self.scaling())
    }

    #[inline]
    fn prepend_rotation(&self, r: &Self::Rotation) -> Self {
        self * r
    }

    #[inline]
    fn append_scaling(&self, s: &Self::NonUniformScaling) -> Self {
        self.append_scaling(*s)
    }

    #[inline]
    fn prepend_scaling(&self, s: &Self::NonUniformScaling) -> Self {
        self.prepend_scaling(*s)
    }

    #[inline]
    fn append_rotation_wrt_point(&self, r: &Self::Rotation, p: &PointBase<N, D, S>) -> Option<Self> {
        let mut res = self.clone();
        res.append_rotation_wrt_point_mut(r, p);
        Some(res)
    }
}

impl<N, D: DimName, S, R> Similarity<PointBase<N, D, S>> for SimilarityBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Scaling = N;

    #[inline]
    fn translation(&self) -> TranslationBase<N, D, S> {
        self.isometry.translation()
    }

    #[inline]
    fn rotation(&self) -> R {
        self.isometry.rotation()
    }

    #[inline]
    fn scaling(&self) -> N {
        self.scaling()
    }
}
