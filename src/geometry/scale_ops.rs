use std::ops::{Mul, MulAssign};

use simba::scalar::ClosedMul;

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::U1;
use crate::base::{Const, SVector, Scalar};

use crate::geometry::{Point, Scale};

// Scale × Scale
add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: &'b Scale<T, D>, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(self.vector.component_mul(&right.vector)) };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: Scale<T, D>, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(self.vector.component_mul(&right.vector)) };
    'a);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: &'b Scale<T, D>, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(self.vector.component_mul(&right.vector)) };
    'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: Scale<T, D>, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(self.vector.component_mul(&right.vector)) }; );

// Scale × scalar
add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: T, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(&self.vector * right) };
    'a);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: T, Output = Scale<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Scale::from(self.vector * right) }; );

// Scale × Point
add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.vector.component_mul(&right.coords)) };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.vector.component_mul(&right.coords)) };
    'a);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.vector.component_mul(&right.coords)) };
    'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.vector.component_mul(&right.coords)) }; );

// Scale *= Scale
add_sub_assign_impl!(MulAssign, mul_assign, ClosedMul;
    const D;
    self: Scale<T, D>, right: &'b Scale<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector.component_mul_assign(&right.vector); };
    'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedMul;
    const D;
    self: Scale<T, D>, right: Scale<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector.component_mul_assign(&right.vector); }; );

// Point * Vector
add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Point<T, D>, right: &'b SVector<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.coords.component_mul(&right)) };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Point<T, D>, right: SVector<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.coords.component_mul(&right)) };
    'a);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Point<T, D>, right: &'b SVector<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.coords.component_mul(&right)) };
    'b);

add_sub_impl!(Mul, mul, ClosedMul;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Point<T, D>, right: SVector<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Point::from(self.coords.component_mul(&right)) }; );