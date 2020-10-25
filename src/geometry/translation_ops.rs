use std::ops::{Div, DivAssign, Mul, MulAssign};

use simba::scalar::{ClosedAdd, ClosedSub};

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{DimName, U1};
use crate::base::{DefaultAllocator, Scalar};

use crate::geometry::{Point, Translation};

// Translation × Translation
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector + &right.vector) };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector + right.vector) };
    'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector + &right.vector) };
    'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector + right.vector) }; );

// Translation ÷ Translation
// FIXME: instead of calling inverse explicitly, could we just add a `mul_tr` or `mul_inv` method?
add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector - &right.vector) };
    'a, 'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector - right.vector) };
    'a);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector - &right.vector) };
    'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector - right.vector) }; );

// Translation × Point
// FIXME: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Point<N, D>, Output = Point<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + &self.vector };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Point<N, D>, Output = Point<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + &self.vector };
    'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Point<N, D>, Output = Point<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + self.vector };
    'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Point<N, D>, Output = Point<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + self.vector }; );

// Translation *= Translation
add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self.vector += &right.vector };
    'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self.vector += right.vector }; );

add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self.vector -= &right.vector };
    'b);

add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self.vector -= right.vector }; );
