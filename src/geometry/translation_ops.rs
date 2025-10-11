use std::ops::{Div, DivAssign, Mul, MulAssign};

use simba::scalar::{ClosedAddAssign, ClosedSubAssign};

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::U1;
use crate::base::{Const, Scalar};

use crate::geometry::{Point, Translation};

// Translation × Translation
/// Composes two translations by adding their displacement vectors.
///
/// Translation composition is commutative: `t1 * t2 = t2 * t1`.
/// The result is a translation by the sum of the two displacement vectors.
///
/// # Example
/// ```
/// # use nalgebra::Translation3;
/// // Translate by (1, 0, 0)
/// let t1 = Translation3::new(1.0, 0.0, 0.0);
/// // Translate by (0, 2, 0)
/// let t2 = Translation3::new(0.0, 2.0, 0.0);
///
/// // Compose: total displacement is (1, 2, 0)
/// let combined = t1 * t2;
/// assert_eq!(combined, Translation3::new(1.0, 2.0, 0.0));
/// ```
///
/// # See Also
/// - [`Translation::div`]: For computing relative translations
add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: &'b Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector + &right.vector) };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector + right.vector) };
    'a);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: &'b Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector + &right.vector) };
    'b);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector + right.vector) }; );

// Translation ÷ Translation
// TODO: instead of calling inverse explicitly, could we just add a `mul_tr` or `mul_inv` method?
add_sub_impl!(Div, div, ClosedSubAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: &'b Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector - &right.vector) };
    'a, 'b);

add_sub_impl!(Div, div, ClosedSubAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(&self.vector - right.vector) };
    'a);

add_sub_impl!(Div, div, ClosedSubAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: &'b Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector - &right.vector) };
    'b);

add_sub_impl!(Div, div, ClosedSubAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: Translation<T, D>, Output = Translation<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { Translation::from(self.vector - right.vector) }; );

// Translation × Point
/// Translates a point by adding the translation's displacement vector.
///
/// This is the fundamental operation for moving points in space. The point's
/// coordinates are shifted by the translation vector.
///
/// # Example
/// ```
/// # use nalgebra::{Translation2, Point2};
/// let trans = Translation2::new(3.0, 4.0);
/// let point = Point2::new(1.0, 2.0);
///
/// let result = trans * point;
/// assert_eq!(result, Point2::new(4.0, 6.0)); // (1+3, 2+4)
/// ```
///
/// # See Also
/// - [`Isometry::transform_point`]: For combined rotation and translation
// TODO: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + &self.vector };
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Translation<T, D>, right: Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + &self.vector };
    'a);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + self.vector };
    'b);

add_sub_impl!(Mul, mul, ClosedAddAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Translation<T, D>, right: Point<T, D>, Output = Point<T, D>;
    #[allow(clippy::suspicious_arithmetic_impl)] { right + self.vector }; );

// Translation *= Translation
add_sub_assign_impl!(MulAssign, mul_assign, ClosedAddAssign;
    const D;
    self: Translation<T, D>, right: &'b Translation<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector += &right.vector };
    'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedAddAssign;
    const D;
    self: Translation<T, D>, right: Translation<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector += right.vector }; );

add_sub_assign_impl!(DivAssign, div_assign, ClosedSubAssign;
    const D;
    self: Translation<T, D>, right: &'b Translation<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector -= &right.vector };
    'b);

add_sub_assign_impl!(DivAssign, div_assign, ClosedSubAssign;
    const D;
    self: Translation<T, D>, right: Translation<T, D>;
    #[allow(clippy::suspicious_op_assign_impl)] { self.vector -= right.vector }; );
