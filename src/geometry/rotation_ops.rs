/*
 *
 * This provides the following operator overladings:
 *
 * Index<(usize, usize)>
 *
 * Rotation × Rotation
 * Rotation ÷ Rotation
 * Rotation × Matrix
 * Matrix   × Rotation
 * Matrix   ÷ Rotation
 * Rotation × Point
 * Rotation × Unit<Vector>
 *
 *
 * Rotation ×= Rotation
 * Matrix   ×= Rotation
 */

use num::{One, Zero};
use std::ops::{Div, DivAssign, Index, Mul, MulAssign};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use crate::base::allocator::Allocator;
use crate::base::constraint::{AreMultipliable, ShapeConstraint};
use crate::base::dimension::{Dim, U1};
use crate::base::storage::Storage;
use crate::base::{
    Const, DefaultAllocator, Matrix, OMatrix, SMatrix, SVector, Scalar, Unit, Vector,
};

use crate::geometry::{Point, Rotation};

/// Indexes the rotation matrix by row and column.
///
/// Allows accessing elements of the underlying rotation matrix using `rot[(row, col)]` syntax.
///
/// # Example
/// ```
/// # use nalgebra::Rotation2;
/// # use std::f64::consts::PI;
/// let rot = Rotation2::new(PI / 2.0); // 90° rotation
/// // Access the rotation matrix elements
/// let element = rot[(0, 0)]; // Top-left element
/// ```
impl<T: Scalar, const D: usize> Index<(usize, usize)> for Rotation<T, D> {
    type Output = T;

    #[inline]
    fn index(&self, row_col: (usize, usize)) -> &T {
        self.matrix().index(row_col)
    }
}

// Rotation × Rotation
/// Composes two rotations by multiplying their matrices.
///
/// The multiplication order is: `rot1 * rot2` means "first apply `rot2`, then apply `rot1`".
/// This composes the two rotations into a single rotation.
///
/// # Example
/// ```
/// # use nalgebra::Rotation2;
/// # use std::f64::consts::PI;
/// // Two 45° rotations
/// let rot1 = Rotation2::new(PI / 4.0);
/// let rot2 = Rotation2::new(PI / 4.0);
///
/// // Compose them to get a 90° rotation
/// let combined = rot1 * rot2;
///
/// // Verify the result is approximately a 90° rotation
/// let expected = Rotation2::new(PI / 2.0);
/// assert_relative_eq!(combined.angle(), expected.angle(), epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`Rotation::div`]: For computing relative rotations
md_impl_all!(
    Mul, mul;
    (Const<D>, Const<D>), (Const<D>, Const<D>)
    const D;
    for;
    where;
    self: Rotation<T, D>, right: Rotation<T, D>, Output = Rotation<T, D>;
    [val val] => Rotation::from_matrix_unchecked(self.into_inner() * right.into_inner());
    [ref val] => Rotation::from_matrix_unchecked(self.matrix() * right.into_inner());
    [val ref] => Rotation::from_matrix_unchecked(self.into_inner() * right.matrix());
    [ref ref] => Rotation::from_matrix_unchecked(self.matrix() * right.matrix());
);

// Rotation ÷ Rotation
/// Computes the relative rotation from `right` to `self`.
///
/// This operation returns the rotation that, when applied after `right`, produces `self`.
/// Mathematically: `self / right = self * right.inverse()`.
///
/// # Example
/// ```
/// # use nalgebra::Rotation2;
/// # use std::f64::consts::PI;
/// let rot1 = Rotation2::new(PI / 2.0); // 90° rotation
/// let rot2 = Rotation2::new(PI / 4.0); // 45° rotation
///
/// // Compute the rotation difference
/// let diff = rot1 / rot2;
///
/// // Verify: rot2 * diff should equal rot1
/// assert_relative_eq!((rot2 * diff).angle(), rot1.angle(), epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`Rotation::rotation_to`]: For finding the shortest rotation between two orientations
// TODO: instead of calling inverse explicitly, could we just add a `mul_tr` or `mul_inv` method?
md_impl_all!(
    Div, div;
    (Const<D>, Const<D>), (Const<D>, Const<D>)
    const D;
    for;
    where;
    self: Rotation<T, D>, right: Rotation<T, D>, Output = Rotation<T, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
);

// Rotation × Matrix
md_impl_all!(
    Mul, mul;
    (Const<D1>, Const<D1>), (R2, C2)
    const D1;
    for R2, C2, SB;
    where R2: Dim, C2: Dim, SB: Storage<T, R2, C2>,
          DefaultAllocator: Allocator<Const<D1>, C2>,
          ShapeConstraint: AreMultipliable<Const<D1>, Const<D1>, R2, C2>;
    self: Rotation<T, D1>, right: Matrix<T, R2, C2, SB>, Output = OMatrix<T, Const<D1>, C2>;
    [val val] => self.into_inner() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.into_inner() * right;
    [ref ref] => self.matrix() * right;
);

// Matrix × Rotation
md_impl_all!(
    Mul, mul;
    (R1, C1), (Const<D2>, Const<D2>)
    const D2;
    for R1, C1, SA;
    where R1: Dim, C1: Dim, SA: Storage<T, R1, C1>,
          DefaultAllocator: Allocator<R1, Const<D2>>,
          ShapeConstraint:  AreMultipliable<R1, C1, Const<D2>, Const<D2>>;
    self: Matrix<T, R1, C1, SA>, right: Rotation<T, D2>, Output = OMatrix<T, R1, Const<D2>>;
    [val val] => self * right.into_inner();
    [ref val] => self * right.into_inner();
    [val ref] => self * right.matrix();
    [ref ref] => self * right.matrix();
);

// Matrix ÷ Rotation
md_impl_all!(
    Div, div;
    (R1, C1), (Const<D2>, Const<D2>)
    const D2;
    for R1, C1, SA;
    where R1: Dim, C1: Dim, SA: Storage<T, R1, C1>,
          DefaultAllocator: Allocator<R1, Const<D2>>,
          ShapeConstraint: AreMultipliable<R1, C1, Const<D2>, Const<D2>>;
    self: Matrix<T, R1, C1, SA>, right: Rotation<T, D2>, Output = OMatrix<T, R1, Const<D2>>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * right.inverse() };
);

// Rotation × Point
/// Rotates a point around the origin.
///
/// This operation applies the rotation to a point in space. The rotation is performed
/// around the origin, so points at the origin remain unchanged.
///
/// # Example
/// ```
/// # use nalgebra::{Rotation2, Point2};
/// # use std::f64::consts::PI;
/// // 90° counterclockwise rotation
/// let rot = Rotation2::new(PI / 2.0);
/// let point = Point2::new(1.0, 0.0);
///
/// let rotated = rot * point;
///
/// // Point is rotated from positive x-axis to positive y-axis
/// assert_relative_eq!(rotated, Point2::new(0.0, 1.0), epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`Isometry::transform_point`]: For rotation combined with translation
// TODO: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
md_impl_all!(
    Mul, mul;
    (Const<D>, Const<D>), (Const<D>, U1)
    const D;
    for;
    where ShapeConstraint:  AreMultipliable<Const<D>, Const<D>, Const<D>, U1>;
    self: Rotation<T, D>, right: Point<T, D>, Output = Point<T, D>;
    [val val] => self.into_inner() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.into_inner() * right;
    [ref ref] => self.matrix() * right;
);

// Rotation × Unit<Vector>
md_impl_all!(
    Mul, mul;
    (Const<D>, Const<D>), (Const<D>, U1)
    const D;
    for S;
    where S: Storage<T, Const<D>>,
          ShapeConstraint: AreMultipliable<Const<D>, Const<D>, Const<D>, U1>;
    self: Rotation<T, D>, right: Unit<Vector<T, Const<D>, S>>, Output = Unit<SVector<T, D>>;
    [val val] => Unit::new_unchecked(self.into_inner() * right.into_inner());
    [ref val] => Unit::new_unchecked(self.matrix() * right.into_inner());
    [val ref] => Unit::new_unchecked(self.into_inner() * right.as_ref());
    [ref ref] => Unit::new_unchecked(self.matrix() * right.as_ref());
);

// Rotation ×= Rotation
// TODO: try not to call `inverse()` explicitly.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (Const<D>, Const<D>), (Const<D>, Const<D>)
    const D; for; where;
    self: Rotation<T, D>, right: Rotation<T, D>;
    [val] => self.matrix_mut_unchecked().mul_assign(right.into_inner());
    [ref] => self.matrix_mut_unchecked().mul_assign(right.matrix());
);

md_assign_impl_all!(
    DivAssign, div_assign;
    (Const<D>, Const<D>), (Const<D>, Const<D>)
    const D; for; where;
    self: Rotation<T, D>, right: Rotation<T, D>;
    [val] => self.matrix_mut_unchecked().mul_assign(right.inverse().into_inner());
    [ref] => self.matrix_mut_unchecked().mul_assign(right.inverse().matrix());
);

// Matrix *= Rotation
// TODO: try not to call `inverse()` explicitly.
// TODO: this shares the same limitations as for the current impl. of MulAssign for matrices.
// (In particular the number of matrix column must be equal to the number of rotation columns,
// i.e., equal to the rotation dimension.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (Const<R1>, Const<C1>), (Const<C1>, Const<C1>)
    const R1, C1; for; where;
    self: SMatrix<T, R1, C1>, right: Rotation<T, C1>;
    [val] => self.mul_assign(right.into_inner());
    [ref] => self.mul_assign(right.matrix());
);

md_assign_impl_all!(
    DivAssign, div_assign;
    (Const<R1>, Const<C1>), (Const<C1>, Const<C1>)
    const R1, C1; for; where;
    self: SMatrix<T, R1, C1>, right: Rotation<T, C1>;
    [val] => self.mul_assign(right.inverse().into_inner());
    [ref] => self.mul_assign(right.inverse().matrix());
);
