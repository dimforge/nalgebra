use std::ops::{Mul, MulAssign, Div, DivAssign};

use alga::general::{ClosedAdd, ClosedSub};

use core::Scalar;
use core::dimension::{DimName, U1};
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};
use core::storage::{OwnedStorage, Storage};
use core::allocator::{OwnedAllocator, SameShapeAllocator};

use geometry::{PointBase, OwnedPoint, TranslationBase, OwnedTranslation};

// TranslationBase × TranslationBase
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(&self.vector + &right.vector); 'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SB>, right: TranslationBase<N, D, SA>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(&self.vector + right.vector); 'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(self.vector + &right.vector); 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SA>, right: TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(self.vector + right.vector); );

// TranslationBase ÷ TranslationBase
// FIXME: instead of calling inverse explicitely, could we just add a `mul_tr` or `mul_inv` method?
add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(&self.vector - &right.vector); 'a, 'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SB>, right: TranslationBase<N, D, SA>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(&self.vector - right.vector); 'a);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(self.vector - &right.vector); 'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SA>, right: TranslationBase<N, D, SB>, Output = OwnedTranslation<N, D, SA>;
    TranslationBase::from_vector(self.vector - right.vector); );


// TranslationBase × PointBase
// FIXME: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SB>, right: &'b PointBase<N, D, SA>, Output = OwnedPoint<N, D, SA::Alloc>;
    right + &self.vector; 'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a TranslationBase<N, D, SB>, right: PointBase<N, D, SA>, Output = OwnedPoint<N, D, SA::Alloc>;
    right + &self.vector; 'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SA>, right: &'b PointBase<N, D, SB>, Output = OwnedPoint<N, D, SB::Alloc>;
    right + self.vector; 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: TranslationBase<N, D, SB>, right: PointBase<N, D, SA>, Output = OwnedPoint<N, D, SA::Alloc>;
    right + self.vector; );


// TranslationBase *= TranslationBase
add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>;
    self.vector += &right.vector; 'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: TranslationBase<N, D, SA>, right: TranslationBase<N, D, SB>;
    self.vector += right.vector; );


add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: TranslationBase<N, D, SA>, right: &'b TranslationBase<N, D, SB>;
    self.vector -= &right.vector; 'b);

add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: TranslationBase<N, D, SA>, right: TranslationBase<N, D, SB>;
    self.vector -= right.vector; );

/*
// TranslationBase × Matrix
add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: &'a TranslationBase<N, D1, SA>, right: &'b Matrix<N, R2, C2, SB>, Output = MatrixMul<N, D1, D1, C2 , SA>;
         self.vector() * right; 'a, 'b);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: &'a TranslationBase<N, D1, SA>, right: Matrix<N, R2, C2, SB>, Output = MatrixMul<N, D1, D1, C2 , SA>;
         self.vector() * right; 'a);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: TranslationBase<N, D1, SA>, right: &'b Matrix<N, R2, C2, SB>, Output = MatrixMul<N, D1, D1, C2 , SA>;
         self.unwrap() * right; 'b);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: TranslationBase<N, D1, SA>, right: Matrix<N, R2, C2, SB>, Output = MatrixMul<N, D1, D1, C2 , SA>;
         self.unwrap() * right; );

// Matrix × TranslationBase
add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: &'a Matrix<N, R1, C1, SA>, right: &'b TranslationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
         self * right.vector(); 'a, 'b);

add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: &'a Matrix<N, R1, C1, SA>, right: TranslationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
         self * right.unwrap(); 'a);

add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: Matrix<N, R1, C1, SA>, right: &'b TranslationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
         self * right.vector(); 'b);


add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: Matrix<N, R1, C1, SA>, right: TranslationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
         self * right.unwrap(); );

// Matrix *= TranslationBase
md_assign_impl!(MulAssign, mul_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1, SA>, right: &'b TranslationBase<N, C1, SB>;
                self.mul_assign(right.vector()); 'b);

md_assign_impl!(MulAssign, mul_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1, SA>, right: TranslationBase<N, C1, SB>;
                self.mul_assign(right.unwrap()); );


md_assign_impl!(DivAssign, div_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1, SA>, right: &'b TranslationBase<N, C1, SB>;
                self.mul_assign(right.inverse().vector()); 'b);

md_assign_impl!(DivAssign, div_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1, SA>, right: TranslationBase<N, C1, SB>;
                self.mul_assign(right.inverse().unwrap()); );
*/
