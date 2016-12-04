use std::ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Index, IndexMut};
use num::Zero;

use alga::general::{ClosedNeg, ClosedAdd, ClosedSub, ClosedMul, ClosedDiv};

use core::{Scalar, ColumnVector, Matrix, ColumnVectorSum};
use core::dimension::{Dim, DimName, U1};
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns, AreMultipliable};
use core::storage::{Storage, StorageMut};
use core::allocator::{SameShapeAllocator, Allocator};

use geometry::{PointBase, OwnedPoint, PointMul};


/*
 *
 * Indexing.
 *
 */
impl<N, D: DimName, S> Index<usize> for PointBase<N, D, S>
    where N: Scalar,
          S: Storage<N, D, U1> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}

impl<N, D: DimName, S> IndexMut<usize> for PointBase<N, D, S>
    where N: Scalar,
          S: StorageMut<N, D, U1> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.coords[i]
    }
}

/*
 * Neg.
 *
 */
impl<N, D: DimName, S> Neg for PointBase<N, D, S>
    where N: Scalar + ClosedNeg,
          S: Storage<N, D, U1> {
    type Output = OwnedPoint<N, D, S::Alloc>;

    #[inline]
    fn neg(self) -> Self::Output {
        PointBase::from_coordinates(-self.coords)
    }
}

impl<'a, N, D: DimName, S> Neg for &'a PointBase<N, D, S>
    where N: Scalar + ClosedNeg,
          S: Storage<N, D, U1> {
    type Output = OwnedPoint<N, D, S::Alloc>;

    #[inline]
    fn neg(self) -> Self::Output {
        PointBase::from_coordinates(-&self.coords)
    }
}

/*
 *
 * Subtraction & Addition.
 *
 */

// PointBase - PointBase
add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: &'a PointBase<N, D, SA>, right: &'b PointBase<N, D, SB>, Output = ColumnVectorSum<N, D, D, SA>;
    &self.coords - &right.coords; 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: &'a PointBase<N, D, SB>, right: PointBase<N, D, SA>, Output = ColumnVectorSum<N, D, D, SA>;
    &self.coords - right.coords; 'a);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: PointBase<N, D, SA>, right: &'b PointBase<N, D, SB>, Output = ColumnVectorSum<N, D, D, SA>;
    self.coords - &right.coords; 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: PointBase<N, D, SA>, right: PointBase<N, D, SB>, Output = ColumnVectorSum<N, D, D, SA>;
    self.coords - right.coords; );

// PointBase - Vector
add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: &'a PointBase<N, D1, SA>, right: &'b ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(&self.coords - right); 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: &'a PointBase<N, D1, SA>, right: ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(&self.coords - &right); 'a); // FIXME: should not be a ref to `right`.

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: PointBase<N, D1, SA>, right: &'b ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(self.coords - right); 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: PointBase<N, D1, SA>, right: ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(self.coords - right); );


// PointBase + Vector
add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: &'a PointBase<N, D1, SA>, right: &'b ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(&self.coords + right); 'a, 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: &'a PointBase<N, D1, SA>, right: ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(&self.coords + &right); 'a); // FIXME: should not be a ref to `right`.

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: PointBase<N, D1, SA>, right: &'b ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(self.coords + right); 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim;
    self: PointBase<N, D1, SA>, right: ColumnVector<N, D2, SB>, Output = OwnedPoint<N, D1, SA::Alloc>;
    Self::Output::from_coordinates(self.coords + right); );


// XXX: replace by the shared macro: add_sub_assign_impl
macro_rules! op_assign_impl(
    ($($TraitAssign: ident, $method_assign: ident, $bound: ident);* $(;)*) => {$(
        impl<'b, N, D1: DimName, D2: Dim, SA, SB> $TraitAssign<&'b ColumnVector<N, D2, SB>> for PointBase<N, D1, SA>
            where N: Scalar + $bound,
                  SA: StorageMut<N, D1, U1>,
                  SB: Storage<N, D2, U1>,
                  ShapeConstraint: SameNumberOfRows<D1, D2> {

            #[inline]
            fn $method_assign(&mut self, right: &'b ColumnVector<N, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }

        impl<N, D1: DimName, D2: Dim, SA, SB> $TraitAssign<ColumnVector<N, D2, SB>> for PointBase<N, D1, SA>
            where N: Scalar + $bound,
                  SA: StorageMut<N, D1, U1>,
                  SB: Storage<N, D2, U1>,
                  ShapeConstraint: SameNumberOfRows<D1, D2> {

            #[inline]
            fn $method_assign(&mut self, right: ColumnVector<N, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }
    )*}
);

op_assign_impl!(
    AddAssign, add_assign, ClosedAdd;
    SubAssign, sub_assign, ClosedSub;
);


/*
 *
 * Matrix × PointBase
 *
 */
md_impl_all!(
    Mul, mul;
    (R1, C1), (D2, U1) for R1: DimName, C1: Dim, D2: DimName
    where SA::Alloc: Allocator<N, R1, U1>
    where ShapeConstraint: AreMultipliable<R1, C1, D2, U1>;
    self: Matrix<N, R1, C1, SA>, right: PointBase<N, D2, SB>, Output = PointMul<N, R1, C1, SA>;
    [val val] => PointBase::from_coordinates(self * right.coords);
    [ref val] => PointBase::from_coordinates(self * right.coords);
    [val ref] => PointBase::from_coordinates(self * &right.coords);
    [ref ref] => PointBase::from_coordinates(self * &right.coords);
);



/*
 *
 * PointBase ×/÷ Scalar
 *
 */
macro_rules! componentwise_scalarop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident) => {
        impl<N, D: DimName, S> $Trait<N> for PointBase<N, D, S>
            where N: Scalar + $bound,
                  S: Storage<N, D, U1> {
            type Output = OwnedPoint<N, D, S::Alloc>;

            #[inline]
            fn $method(self, right: N) -> Self::Output {
                PointBase::from_coordinates(self.coords.$method(right))
            }
        }

        impl<'a, N, D: DimName, S> $Trait<N> for &'a PointBase<N, D, S>
            where N: Scalar + $bound,
                  S: Storage<N, D, U1> {
            type Output = OwnedPoint<N, D, S::Alloc>;

            #[inline]
            fn $method(self, right: N) -> Self::Output {
                PointBase::from_coordinates((&self.coords).$method(right))
            }
        }

        impl<N, D: DimName, S> $TraitAssign<N> for PointBase<N, D, S>
            where N: Scalar + $bound,
                  S: StorageMut<N, D, U1> {
            #[inline]
            fn $method_assign(&mut self, right: N) {
                self.coords.$method_assign(right)
            }
        }
    }
);

componentwise_scalarop_impl!(Mul, mul, ClosedMul; MulAssign, mul_assign);
componentwise_scalarop_impl!(Div, div, ClosedDiv; DivAssign, div_assign);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl<D: DimName, S> Mul<PointBase<$T, D, S>> for $T
            where S: Storage<$T, D, U1> {
            type Output = OwnedPoint<$T, D, S::Alloc>;

            #[inline]
            fn mul(self, right: PointBase<$T, D, S>) -> Self::Output {
                PointBase::from_coordinates(self * right.coords)
            }
        }

        impl<'b, D: DimName, S> Mul<&'b PointBase<$T, D, S>> for $T
            where S: Storage<$T, D, U1> {
            type Output = OwnedPoint<$T, D, S::Alloc>;

            #[inline]
            fn mul(self, right: &'b PointBase<$T, D, S>) -> Self::Output {
                PointBase::from_coordinates(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(
    u8, u16, u32, u64, usize,
    i8, i16, i32, i64, isize,
    f32, f64
);
