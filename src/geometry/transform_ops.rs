use num::{Zero, One};
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign};
use approx::ApproxEq;

use alga::general::{Field, Real, ClosedAdd, ClosedMul, ClosedNeg, SubsetOf};

use core::{Scalar, ColumnVector, OwnedColumnVector, OwnedSquareMatrix};
use core::storage::{Storage, StorageMut, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1, U3, U4};

use geometry::{PointBase, OwnedPoint, TransformBase, OwnedTransform, TCategory, TCategoryMul,
               SubTCategoryOf, SuperTCategoryOf, TGeneral, TProjective, TAffine, RotationBase,
               UnitQuaternionBase, IsometryBase, SimilarityBase, TranslationBase};

/*
 *
 * In the following, we provide:
 * =========================
 *
 * Index<(usize, usize)>
 * IndexMut<(usize, usize)> (where TCategory == TGeneral)
 *
 * (Operators)
 *
 * TransformBase × IsometryBase
 * TransformBase × RotationBase
 * TransformBase × SimilarityBase
 * TransformBase × TransformBase
 * TransformBase × UnitQuaternion
 * FIXME: TransformBase × UnitComplex
 * TransformBase × TranslationBase
 * TransformBase × ColumnVector
 * TransformBase × PointBase
 *
 * IsometryBase       × TransformBase
 * RotationBase       × TransformBase
 * SimilarityBase     × TransformBase
 * TranslationBase    × TransformBase
 * UnitQuaternionBase × TransformBase
 * FIXME: UnitComplex × TransformBase
 *
 * FIXME: TransformBase ÷ IsometryBase
 * TransformBase ÷ RotationBase
 * FIXME: TransformBase ÷ SimilarityBase
 * TransformBase ÷ TransformBase
 * TransformBase ÷ UnitQuaternion
 * TransformBase ÷ TranslationBase
 *
 * FIXME: IsometryBase       ÷ TransformBase
 * RotationBase       ÷ TransformBase
 * FIXME: SimilarityBase     ÷ TransformBase
 * TranslationBase    ÷ TransformBase
 * UnitQuaternionBase ÷ TransformBase
 * FIXME: UnitComplex ÷ TransformBase
 *
 *
 * (Assignment Operators)
 *
 *
 * TransformBase ×= TransformBase
 * TransformBase ×= SimilarityBase
 * TransformBase ×= IsometryBase
 * TransformBase ×= RotationBase
 * TransformBase ×= UnitQuaternionBase
 * FIXME: TransformBase ×= UnitComplex
 * TransformBase ×= TranslationBase
 *
 * TransformBase ÷= TransformBase
 * FIXME: TransformBase ÷= SimilarityBase
 * FIXME: TransformBase ÷= IsometryBase
 * TransformBase ÷= RotationBase
 * TransformBase ÷= UnitQuaternionBase
 * FIXME: TransformBase ÷= UnitComplex
 *
 */

/*
 *
 * Indexing.
 *
 */
impl<N, D, S, C: TCategory> Index<(usize, usize)> for TransformBase<N, D, S, C>
    where N: Scalar,
          D: DimName + DimNameAdd<U1>,
          S: Storage<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    type Output = N;

    #[inline]
    fn index(&self, ij: (usize, usize)) -> &N {
        self.matrix().index(ij)
    }
}

// Only general transformations are mutably indexable.
impl<N, D, S> IndexMut<(usize, usize)> for TransformBase<N, D, S, TGeneral>
    where N: Scalar,
          D: DimName + DimNameAdd<U1>,
          S: StorageMut<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
    #[inline]
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut N {
        self.matrix_mut().index_mut(ij)
    }
}


// TransformBase × ColumnVector
md_impl_all!(
    Mul, mul where N: Field;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory
    where SA::Alloc: Allocator<N, D, D>
    where SA::Alloc: Allocator<N, D, U1>
    where SA::Alloc: Allocator<N, U1, D>;
    self: TransformBase<N, D, SA, C>, rhs: ColumnVector<N, D, SB>, Output = OwnedColumnVector<N, D, SA::Alloc>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform = self.matrix().fixed_slice::<D, D>(0, 0);

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, D>(D::dim(), 0);
            let n = normalizer.tr_dot(&rhs);

            if !n.is_zero() {
                return transform * (rhs / n);
            }
        }

        transform * rhs
    };
);


// TransformBase × PointBase
md_impl_all!(
    Mul, mul where N: Field;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory
    where SA::Alloc: Allocator<N, D, D>
    where SA::Alloc: Allocator<N, D, U1>
    where SA::Alloc: Allocator<N, U1, D>;
    self: TransformBase<N, D, SA, C>, rhs: PointBase<N, D, SB>, Output = OwnedPoint<N, D, SA::Alloc>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform   = self.matrix().fixed_slice::<D, D>(0, 0);
        let translation = self.matrix().fixed_slice::<D, U1>(0, D::dim());

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, D>(D::dim(), 0);
            let n = normalizer.tr_dot(&rhs.coords) + unsafe { *self.matrix().get_unchecked(D::dim(), D::dim()) };

            if !n.is_zero() {
                return transform * (rhs / n) + translation;
            }
        }

        transform * rhs + translation
    };
);


// TransformBase × TransformBase
md_impl_all!(
    Mul, mul;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: TCategory;
    self: TransformBase<N, D, SA, CA>, rhs: TransformBase<N, D, SB, CB>, Output = OwnedTransform<N, D, SA::Alloc, CA::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.matrix());
);


// TransformBase × RotationBase
md_impl_all!(
    Mul, mul where N: One;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: RotationBase<N, D, SB>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);


// RotationBase × TransformBase
md_impl_all!(
    Mul, mul where N: One;
    (D, D), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: RotationBase<N, D, SA>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);


// TransformBase × UnitQuaternionBase
md_impl_all!(
    Mul, mul where N: Real;
    (U4, U4), (U4, U1) for C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, U3, U3>
    where SB::Alloc: Allocator<N, U4, U4>;
    self: TransformBase<N, U3, SA, C>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedTransform<N, U3, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);


// UnitQuaternionBase × TransformBase
md_impl_all!(
    Mul, mul where N: Real;
    (U4, U1), (U4, U4) for C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, U3, U3>
    where SA::Alloc: Allocator<N, U4, U4>;
    self: UnitQuaternionBase<N, SA>, rhs: TransformBase<N, U3, SB, C>, Output = OwnedTransform<N, U3, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);



// TransformBase × IsometryBase
md_impl_all!(
    Mul, mul where N: Real;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: IsometryBase<N, D, SB, R>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// IsometryBase × TransformBase
md_impl_all!(
    Mul, mul where N: Real;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SA::Alloc> >
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: IsometryBase<N, D, SA, R>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);


// TransformBase × SimilarityBase
md_impl_all!(
    Mul, mul where N: Real;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
    where SB::Alloc: Allocator<N, D, D >
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: SimilarityBase<N, D, SB, R>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// SimilarityBase × TransformBase
md_impl_all!(
    Mul, mul where N: Real;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SA::Alloc> >
    where SA::Alloc: Allocator<N, D, D >
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: SimilarityBase<N, D, SA, R>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);



/*
 *
 * FIXME: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// TransformBase × TranslationBase
md_impl_all!(
    Mul, mul where N: Real;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: TranslationBase<N, D, SB>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// TranslationBase × TransformBase
md_impl_all!(
    Mul, mul where N: Real;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TranslationBase<N, D, SA>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);



// TransformBase ÷ TransformBase
md_impl_all!(
    Div, div where N: ApproxEq, Field;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: SubTCategoryOf<TProjective>;
    self: TransformBase<N, D, SA, CA>, rhs: TransformBase<N, D, SB, CB>, Output = OwnedTransform<N, D, SA::Alloc, CA::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.clone_owned().inverse();
    [ref ref] => self * rhs.clone_owned().inverse();
);

// TransformBase ÷ RotationBase
md_impl_all!(
    Div, div where N: One;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: RotationBase<N, D, SB>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);


// RotationBase ÷ TransformBase
md_impl_all!(
    Div, div where N: One;
    (D, D), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: RotationBase<N, D, SA>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);


// TransformBase ÷ UnitQuaternionBase
md_impl_all!(
    Div, div where N: Real;
    (U4, U4), (U4, U1) for C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, U3, U3>
    where SB::Alloc: Allocator<N, U4, U4>;
    self: TransformBase<N, U3, SA, C>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedTransform<N, U3, SA::Alloc, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);


// UnitQuaternionBase ÷ TransformBase
md_impl_all!(
    Div, div where N: Real;
    (U4, U1), (U4, U4) for C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, U3, U3>
    where SA::Alloc: Allocator<N, U4, U4>;
    self: UnitQuaternionBase<N, SA>, rhs: TransformBase<N, U3, SB, C>, Output = OwnedTransform<N, U3, SA::Alloc, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);



//      // TransformBase ÷ IsometryBase
//      md_impl_all!(
//          Div, div where N: Real;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: TransformBase<N, D, SA, C>, rhs: IsometryBase<N, D, SB, R>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.inverse().to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.inverse().to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//      );

//      // IsometryBase ÷ TransformBase
//      md_impl_all!(
//          Div, div where N: Real;
//          (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SA::Alloc> >
//          where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: IsometryBase<N, D, SA, R>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );


//      // TransformBase ÷ SimilarityBase
//      md_impl_all!(
//          Div, div where N: Real;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
//          where SB::Alloc: Allocator<N, D, D >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: TransformBase<N, D, SA, C>, rhs: SimilarityBase<N, D, SB, R>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.unwrap() * rhs.to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//      );

//      // SimilarityBase ÷ TransformBase
//      md_impl_all!(
//          Div, div where N: Real;
//          (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SA::Alloc> >
//          where SA::Alloc: Allocator<N, D, D >
//          where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: SimilarityBase<N, D, SA, R>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.unwrap());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );



// TransformBase ÷ TranslationBase
md_impl_all!(
    Div, div where N: Real;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: TranslationBase<N, D, SB>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// TranslationBase ÷ TransformBase
md_impl_all!(
    Div, div where N: Real;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>
    where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TranslationBase<N, D, SA>, rhs: TransformBase<N, D, SB, C>, Output = OwnedTransform<N, D, SA::Alloc, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);


// TransformBase ×= TransformBase
md_assign_impl_all!(
    MulAssign, mul_assign;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategory, CB: SubTCategoryOf<CA>;
    self: TransformBase<N, D, SA, CA>, rhs: TransformBase<N, D, SB, CB>;
    [val] => *self.matrix_mut_unchecked() *= rhs.unwrap();
    [ref] => *self.matrix_mut_unchecked() *= rhs.matrix();
);


// TransformBase ×= SimilarityBase
md_assign_impl_all!(
    MulAssign, mul_assign;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >
    where SB::Alloc: Allocator<N, D, D>;
    self: TransformBase<N, D, SA, C>, rhs: SimilarityBase<N, D, SB, R>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);


// TransformBase ×= IsometryBase
md_assign_impl_all!(
    MulAssign, mul_assign;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: IsometryBase<N, D, SB, R>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

/*
 *
 * FIXME: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// TransformBase ×= TranslationBase
md_assign_impl_all!(
    MulAssign, mul_assign where N: One;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: TranslationBase<N, D, SB>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);


// TransformBase ×= RotationBase
md_assign_impl_all!(
    MulAssign, mul_assign where N: One;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategory
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: RotationBase<N, D, SB>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);


// TransformBase ×= UnitQuaternionBase
md_assign_impl_all!(
    MulAssign, mul_assign where N: Real;
    (U4, U4), (U4, U1) for C: TCategory
    where SB::Alloc: Allocator<N, U3, U3>
    where SB::Alloc: Allocator<N, U4, U4>;
    self: TransformBase<N, U3, SA, C>, rhs: UnitQuaternionBase<N, SB>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);


// TransformBase ÷= TransformBase
md_assign_impl_all!(
    DivAssign, div_assign where N: Field, ApproxEq;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, CA: SuperTCategoryOf<CB>, CB: SubTCategoryOf<TProjective>;
    self: TransformBase<N, D, SA, CA>, rhs: TransformBase<N, D, SB, CB>;
    [val] => *self *= rhs.clone_owned().inverse();
    [ref] => *self *= rhs.clone_owned().inverse();
);


//      // TransformBase ÷= SimilarityBase
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >
//          where SB::Alloc: Allocator<N, D, D>;
//          self: TransformBase<N, D, SA, C>, rhs: SimilarityBase<N, D, SB, R>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );
//      
//      
//      // TransformBase ÷= IsometryBase
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<OwnedSquareMatrix<N, DimNameSum<D, U1>, SB::Alloc> >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: TransformBase<N, D, SA, C>, rhs: IsometryBase<N, D, SB, R>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );


// TransformBase ÷= TranslationBase
md_assign_impl_all!(
    DivAssign, div_assign where N: One, ClosedNeg;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: TranslationBase<N, D, SB>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);


// TransformBase ÷= RotationBase
md_assign_impl_all!(
    DivAssign, div_assign where N: One;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategory
    where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
    self: TransformBase<N, D, SA, C>, rhs: RotationBase<N, D, SB>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);


// TransformBase ÷= UnitQuaternionBase
md_assign_impl_all!(
    DivAssign, div_assign where N: Real;
    (U4, U4), (U4, U1) for C: TCategory
    where SB::Alloc: Allocator<N, U3, U3>
    where SB::Alloc: Allocator<N, U4, U4>;
    self: TransformBase<N, U3, SA, C>, rhs: UnitQuaternionBase<N, SB>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);
