use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Real, Inverse, Multiplicative, Identity, Id};
use alga::linear::{Transformation, Similarity, AffineTransformation, Isometry, DirectIsometry,
                   OrthogonalTransformation, ProjectiveTransformation, Rotation};

use core::ColumnVector;
use core::dimension::{DimName, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{RotationBase, PointBase};



/*
 *
 * Algebraic structures.
 *
 */
impl<N, D: DimName, S> Identity<Multiplicative> for RotationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S> Inverse<Multiplicative> for RotationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn inverse(&self) -> Self {
        self.transpose()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

impl<N, D: DimName, S> AbstractMagma<Multiplicative> for RotationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimName, S> $marker<$operator> for RotationBase<N, D, S>
            where N: Real,
                  S: OwnedStorage<N, D, D>,
                  S::Alloc: OwnedAllocator<N, D, D, S> { }
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
impl<N, D: DimName, SA, SB> Transformation<PointBase<N, D, SB>> for RotationBase<N, D, SA>
    where N:  Real,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, D, SB>) -> PointBase<N, D, SB> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, D, SB>) -> ColumnVector<N, D, SB> {
        self * v
    }
}

impl<N, D: DimName, SA, SB> ProjectiveTransformation<PointBase<N, D, SB>> for RotationBase<N, D, SA>
    where N:  Real,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, D, SB>) -> PointBase<N, D, SB> {
        PointBase::from_coordinates(self.inverse_transform_vector(&pt.coords))
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, D, SB>) -> ColumnVector<N, D, SB> {
        self.matrix().tr_mul(v)
    }
}

impl<N, D: DimName, SA, SB> AffineTransformation<PointBase<N, D, SB>> for RotationBase<N, D, SA>
    where N:  Real,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    type Rotation          = Self;
    type NonUniformScaling = Id;
    type Translation       = Id;

    #[inline]
    fn decompose(&self) -> (Id, Self, Id, Self) {
        (Id::new(), self.clone(), Id::new(), Self::identity())
    }

    #[inline]
    fn append_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        r * self
    }

    #[inline]
    fn prepend_rotation(&self, r: &Self::Rotation) -> Self {
        self * r
    }

    #[inline]
    fn append_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }
}


impl<N, D: DimName, SA, SB> Similarity<PointBase<N, D, SB>> for RotationBase<N, D, SA>
    where N:  Real,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    type Scaling  = Id;

    #[inline]
    fn translation(&self) -> Id {
        Id::new()
    }

    #[inline]
    fn rotation(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N, D: DimName, SA, SB> $Trait<PointBase<N, D, SB>> for RotationBase<N, D, SA>
        where N:  Real,
              SA: OwnedStorage<N, D, D>,
              SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
              SA::Alloc: OwnedAllocator<N, D, D, SA>,
              SB::Alloc: OwnedAllocator<N, D, U1, SB> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);


/// Subgroups of the n-dimensional rotation group `SO(n)`.
impl<N, D: DimName, SA, SB> Rotation<PointBase<N, D, SB>> for RotationBase<N, D, SA>
    where N:  Real,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        // XXX: add the general case.
        unimplemented!()
    }

    #[inline]
    fn rotation_between(a: &ColumnVector<N, D, SB>, b: &ColumnVector<N, D, SB>) -> Option<Self> {
        // XXX: add the general case.
        unimplemented!()
    }

    #[inline]
    fn scaled_rotation_between(a: &ColumnVector<N, D, SB>, b: &ColumnVector<N, D, SB>, n: N) -> Option<Self> {
        // XXX: add the general case.
        unimplemented!()
    }
}

/*
impl<N: Real> Matrix for RotationBase<N> {
    type Field     = N;
    type Row       = Matrix<N>;
    type Column    = Matrix<N>;
    type Transpose = Self;

    #[inline]
    fn nrows(&self) -> usize {
        self.submatrix.nrows()
    }

    #[inline]
    fn ncolumns(&self) -> usize {
        self.submatrix.ncolumns()
    }

    #[inline]
    fn row(&self, i: usize) -> Self::Row {
        self.submatrix.row(i)
    }

    #[inline]
    fn column(&self, i: usize) -> Self::Column {
        self.submatrix.column(i)
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> Self::Field {
        self.submatrix[(i, j)]
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Field {
        self.submatrix.at_fast(i, j)
    }

    #[inline]
    fn transpose(&self) -> Self::Transpose {
        RotationBase::from_matrix_unchecked(self.submatrix.transpose())
    }
}

impl<N: Real> SquareMatrix for RotationBase<N> {
    type Vector = Matrix<N>;

    #[inline]
    fn diagonal(&self) -> Self::Coordinates {
        self.submatrix.diagonal()
    }

    #[inline]
    fn determinant(&self) -> Self::Field {
        ::one()
    }

    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        Some(::transpose(self))
    }

    #[inline]
    fn try_inverse_mut(&mut self) -> bool {
        self.transpose_mut();
        true
    }

    #[inline]
    fn transpose_mut(&mut self) {
        self.submatrix.transpose_mut()
    }
}

impl<N: Real> InversibleSquareMatrix for RotationBase<N> { }
*/


