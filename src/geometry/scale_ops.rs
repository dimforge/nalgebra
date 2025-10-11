use std::ops::{Mul, MulAssign};

use simba::scalar::ClosedMulAssign;

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::U1;
use crate::base::{Const, SVector, Scalar};

use crate::geometry::{Point, Scale};

// Scale × Scale
/// Composes two non-uniform scaling transformations.
///
/// The result is a scale transformation where each axis is scaled by the product
/// of the corresponding scale factors. This is component-wise multiplication.
///
/// # Example
/// ```
/// # use nalgebra::Scale3;
/// // Scale x by 2, y by 3, z by 4
/// let scale1 = Scale3::new(2.0, 3.0, 4.0);
/// // Scale x by 1.5, y by 2, z by 0.5
/// let scale2 = Scale3::new(1.5, 2.0, 0.5);
///
/// // Compose: x scaled by 3, y by 6, z by 2
/// let combined = scale1 * scale2;
/// assert_eq!(combined, Scale3::new(3.0, 6.0, 2.0));
/// ```
///
/// # See Also
/// - [`Scale::div`]: For computing inverse scaling transformations
add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: &'b Scale<T, D>, Output = Scale<T, D>;
    Scale::from(self.vector.component_mul(&right.vector));
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: Scale<T, D>, Output = Scale<T, D>;
    Scale::from(self.vector.component_mul(&right.vector));
    'a);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: &'b Scale<T, D>, Output = Scale<T, D>;
    Scale::from(self.vector.component_mul(&right.vector));
    'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: Scale<T, D>, Output = Scale<T, D>;
    Scale::from(self.vector.component_mul(&right.vector)); );

// Scale × scalar
add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: T, Output = Scale<T, D>;
    Scale::from(&self.vector * right);
    'a);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: T, Output = Scale<T, D>;
    Scale::from(self.vector * right); );

// Scale × Point
/// Scales a point using non-uniform scaling.
///
/// Each coordinate of the point is multiplied by the corresponding scale factor.
/// This allows stretching or compressing along different axes independently.
///
/// # Example
/// ```
/// # use nalgebra::{Scale2, Point2};
/// // Scale x by 2, y by 3
/// let scale = Scale2::new(2.0, 3.0);
/// let point = Point2::new(4.0, 5.0);
///
/// let result = scale * point;
/// assert_eq!(result, Point2::new(8.0, 15.0)); // (4*2, 5*3)
/// ```
///
/// # See Also
/// - [`Scale::transform_vector`]: For scaling vectors
add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    Point::from(self.vector.component_mul(&right.coords));
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: Point<T, D>, Output = Point<T, D>;
    Point::from(self.vector.component_mul(&right.coords));
    'a);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: &'b Point<T, D>, Output = Point<T, D>;
    Point::from(self.vector.component_mul(&right.coords));
    'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: Point<T, D>, Output = Point<T, D>;
    Point::from(self.vector.component_mul(&right.coords)); );

// Scale * Vector
/// Scales a vector using non-uniform scaling.
///
/// Each component of the vector is multiplied by the corresponding scale factor.
/// This is useful for applying different scaling factors along different axes.
///
/// # Example
/// ```
/// # use nalgebra::{Scale3, Vector3};
/// // Scale x by 2, y by 3, z by 4
/// let scale = Scale3::new(2.0, 3.0, 4.0);
/// let vec = Vector3::new(1.0, 1.0, 1.0);
///
/// let result = scale * vec;
/// assert_eq!(result, Vector3::new(2.0, 3.0, 4.0));
/// ```
///
/// # See Also
/// - [`Scale::transform_point`]: For scaling points
add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: &'b SVector<T, D>, Output = SVector<T, D>;
    self.vector.component_mul(right);
    'a, 'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Scale<T, D>, right: SVector<T, D>, Output = SVector<T, D>;
    self.vector.component_mul(&right);
    'a);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: &'b SVector<T, D>, Output = SVector<T, D>;
    self.vector.component_mul(right);
    'b);

add_sub_impl!(Mul, mul, ClosedMulAssign;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Scale<T, D>, right: SVector<T, D>, Output = SVector<T, D>;
    self.vector.component_mul(&right); );

// Scale *= Scale
add_sub_assign_impl!(MulAssign, mul_assign, ClosedMulAssign;
    const D;
    self: Scale<T, D>, right: &'b Scale<T, D>;
    self.vector.component_mul_assign(&right.vector);
    'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedMulAssign;
    const D;
    self: Scale<T, D>, right: Scale<T, D>;
    self.vector.component_mul_assign(&right.vector); );

// Scale ×= scalar
add_sub_assign_impl!(MulAssign, mul_assign, ClosedMulAssign;
    const D;
    self: Scale<T, D>, right: T;
    self.vector *= right; );
