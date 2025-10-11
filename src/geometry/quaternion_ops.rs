// The macros break if the references are taken out, for some reason.
#![allow(clippy::op_ref)]

/*
 * This file provides:
 * ===================
 *
 *
 * (Quaternion)
 *
 * Index<usize>
 * IndexMut<usize>
 * Quaternion × Quaternion
 * Quaternion + Quaternion
 * Quaternion - Quaternion
 * -Quaternion
 * Quaternion × Scalar
 * Quaternion ÷ Scalar
 * Scalar × Quaternion
 *
 * (Unit Quaternion)
 * UnitQuaternion × UnitQuaternion
 * UnitQuaternion × Rotation       -> UnitQuaternion
 * Rotation       × UnitQuaternion -> UnitQuaternion
 *
 * UnitQuaternion ÷ UnitQuaternion
 * UnitQuaternion ÷ Rotation       -> UnitQuaternion
 * Rotation       ÷ UnitQuaternion -> UnitQuaternion
 *
 *
 * UnitQuaternion × Point
 * UnitQuaternion × Vector
 * UnitQuaternion × Unit<Vector>
 *
 * NOTE: -UnitQuaternion is already provided by `Unit<T>`.
 *
 *
 * (Assignment Operators)
 *
 * Quaternion ×= Scalar
 * Quaternion ×= Quaternion
 * Quaternion += Quaternion
 * Quaternion -= Quaternion
 *
 * UnitQuaternion ×= UnitQuaternion
 * UnitQuaternion ×= Rotation
 *
 * UnitQuaternion ÷= UnitQuaternion
 * UnitQuaternion ÷= Rotation
 *
 * TODO: Rotation ×= UnitQuaternion
 * TODO: Rotation ÷= UnitQuaternion
 *
 */

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::SimdRealField;
use crate::base::dimension::U3;
use crate::base::storage::Storage;
use crate::base::{Const, Scalar, Unit, Vector, Vector3};

use crate::geometry::{Point3, Quaternion, Rotation, UnitQuaternion};

/// Indexes the quaternion by component.
///
/// Allows accessing quaternion components using `q[i]` syntax, where:
/// - `i = 0`: x (i component)
/// - `i = 1`: y (j component)
/// - `i = 2`: z (k component)
/// - `i = 3`: w (scalar component)
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let q = Quaternion::new(4.0, 1.0, 2.0, 3.0);
/// assert_eq!(q[0], 1.0); // x component (i)
/// assert_eq!(q[1], 2.0); // y component (j)
/// assert_eq!(q[2], 3.0); // z component (k)
/// assert_eq!(q[3], 4.0); // w component (scalar)
/// ```
impl<T: Scalar> Index<usize> for Quaternion<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}

/// Mutably indexes the quaternion by component.
///
/// Allows modifying quaternion components using `q[i]` syntax.
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let mut q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// q[0] = 5.0; // Modify x component
/// assert_eq!(q[0], 5.0);
/// ```
impl<T: Scalar> IndexMut<usize> for Quaternion<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.coords[i]
    }
}

macro_rules! quaternion_op_impl(
    ($Op: ident, $op: ident;
     $($Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr_2021; $($lives: tt),*) => {
        impl<$($lives ,)* T: SimdRealField $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where T::Element: SimdRealField {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

// Quaternion + Quaternion
/// Adds two quaternions component-wise.
///
/// This operation is **not** the same as quaternion multiplication. It simply adds
/// the corresponding components together: `(w1+w2, x1+x2, y1+y2, z1+z2)`.
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
/// let result = q1 + q2;
/// assert_eq!(result, Quaternion::new(6.0, 8.0, 10.0, 12.0));
/// ```
///
/// # See Also
/// - [`Quaternion::mul`]: For quaternion multiplication (composition of rotations)
quaternion_op_impl!(
    Add, add;
    ;
    self: &'a Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(&self.coords + &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Add, add;
    ;
    self: &'a Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(&self.coords + rhs.coords);
    'a);

quaternion_op_impl!(
    Add, add;
    ;
    self: Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(self.coords + &rhs.coords);
    'b);

quaternion_op_impl!(
    Add, add;
    ;
    self: Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(self.coords + rhs.coords); );

// Quaternion - Quaternion
/// Subtracts two quaternions component-wise.
///
/// This operation is **not** the same as quaternion division. It simply subtracts
/// the corresponding components: `(w1-w2, x1-x2, y1-y2, z1-z2)`.
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let q1 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
/// let q2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let result = q1 - q2;
/// assert_eq!(result, Quaternion::new(4.0, 4.0, 4.0, 4.0));
/// ```
quaternion_op_impl!(
    Sub, sub;
    ;
    self: &'a Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(&self.coords - &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Sub, sub;
    ;
    self: &'a Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(&self.coords - rhs.coords);
    'a);

quaternion_op_impl!(
    Sub, sub;
    ;
    self: Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(self.coords - &rhs.coords);
    'b);

quaternion_op_impl!(
    Sub, sub;
    ;
    self: Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    Quaternion::from(self.coords - rhs.coords); );

// Quaternion × Quaternion
/// Multiplies two quaternions using the Hamilton product.
///
/// Quaternion multiplication is **not commutative**: `q1 * q2` is generally different from `q2 * q1`.
/// The result represents the composition of the two rotations (if the quaternions are unit quaternions).
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity rotation
/// let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0); // 180° rotation around x-axis
/// let result = q1 * q2;
/// // The result combines both rotations
/// ```
///
/// # See Also
/// - [`UnitQuaternion::mul`]: For composing rotations represented as unit quaternions
quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    Quaternion::new(
        self[3].clone() * rhs[3].clone() - self[0].clone() * rhs[0].clone() - self[1].clone() * rhs[1].clone() - self[2].clone() * rhs[2].clone(),
        self[3].clone() * rhs[0].clone() + self[0].clone() * rhs[3].clone() + self[1].clone() * rhs[2].clone() - self[2].clone() * rhs[1].clone(),
        self[3].clone() * rhs[1].clone() - self[0].clone() * rhs[2].clone() + self[1].clone() * rhs[3].clone() + self[2].clone() * rhs[0].clone(),
        self[3].clone() * rhs[2].clone() + self[0].clone() * rhs[1].clone() - self[1].clone() * rhs[0].clone() + self[2].clone() * rhs[3].clone());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: Quaternion<T>, rhs: &'b Quaternion<T>, Output = Quaternion<T>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: Quaternion<T>, rhs: Quaternion<T>, Output = Quaternion<T>;
    &self * &rhs; );

// UnitQuaternion × UnitQuaternion
/// Composes two 3D rotations by multiplying two unit quaternions.
///
/// This operation combines two rotations into a single rotation. The order matters:
/// `q1 * q2` means "first apply rotation `q2`, then apply rotation `q1`".
///
/// # Example
/// ```
/// # use nalgebra::{UnitQuaternion, Vector3};
/// # use std::f32::consts::PI;
/// // 90° rotation around the z-axis
/// let rot_z = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
/// // 90° rotation around the x-axis
/// let rot_x = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);
///
/// // Compose rotations: first rotate around z, then around x
/// let combined = rot_x * rot_z;
///
/// // This is equivalent to applying rot_z first, then rot_x
/// let point = Vector3::new(1.0, 0.0, 0.0);
/// let result1 = combined * point;
/// let result2 = rot_x * (rot_z * point);
/// assert_eq!(result1, result2);
/// ```
///
/// # See Also
/// - [`UnitQuaternion::div`]: For computing the relative rotation between two orientations
quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>, Output = UnitQuaternion<T>;
    UnitQuaternion::new_unchecked(self.quaternion() * rhs.quaternion());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: UnitQuaternion<T>, Output = UnitQuaternion<T>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>, Output = UnitQuaternion<T>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: UnitQuaternion<T>, Output = UnitQuaternion<T>;
    &self * &rhs; );

// UnitQuaternion ÷ UnitQuaternion
/// Computes the relative rotation from `rhs` to `self`.
///
/// This operation returns the rotation that, when applied after `rhs`, gives `self`.
/// Mathematically: `self / rhs = self * rhs.inverse()`.
///
/// # Example
/// ```
/// # use nalgebra::{UnitQuaternion, Vector3};
/// # use std::f32::consts::PI;
/// let rot1 = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
/// let rot2 = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 4.0);
///
/// // Compute the rotation that takes rot2 to rot1
/// let diff = rot1 / rot2;
///
/// // Verify: rot2 * diff should equal rot1
/// assert_relative_eq!(rot2 * diff, rot1, epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`UnitQuaternion::rotation_to`]: For computing the shortest rotation between two orientations
quaternion_op_impl!(
    Div, div;
    ;
    self: &'a UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>, Output = UnitQuaternion<T>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: &'a UnitQuaternion<T>, rhs: UnitQuaternion<T>, Output = UnitQuaternion<T>;
    self / &rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    ;
    self: UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>, Output = UnitQuaternion<T>;
    &self / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: UnitQuaternion<T>, rhs: UnitQuaternion<T>, Output = UnitQuaternion<T>;
    &self / &rhs; );

// UnitQuaternion × Rotation
quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: &'b Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    // TODO: can we avoid the conversion from a rotation matrix?
    self * UnitQuaternion::<T>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self * UnitQuaternion::<T>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: &'b Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self * UnitQuaternion::<T>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self * UnitQuaternion::<T>::from_rotation_matrix(&rhs); );

// UnitQuaternion ÷ Rotation
quaternion_op_impl!(
    Div, div;
    ;
    self: &'a UnitQuaternion<T>, rhs: &'b Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    // TODO: can we avoid the conversion to a rotation matrix?
    self / UnitQuaternion::<T>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: &'a UnitQuaternion<T>, rhs: Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self / UnitQuaternion::<T>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Div, div;
    ;
    self: UnitQuaternion<T>, rhs: &'b Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self / UnitQuaternion::<T>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: UnitQuaternion<T>, rhs: Rotation<T, 3>,
    Output = UnitQuaternion<T>;
    self / UnitQuaternion::<T>::from_rotation_matrix(&rhs); );

// Rotation × UnitQuaternion
quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a Rotation<T, 3>, rhs: &'b UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    // TODO: can we avoid the conversion from a rotation matrix?
    UnitQuaternion::<T>::from_rotation_matrix(self) * rhs;
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a Rotation<T, 3>, rhs: UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(self) * rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: Rotation<T, 3>, rhs: &'b UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(&self) * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: Rotation<T, 3>, rhs: UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(&self) * rhs; );

// Rotation ÷ UnitQuaternion
quaternion_op_impl!(
    Div, div;
    ;
    self: &'a Rotation<T, 3>, rhs: &'b UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    // TODO: can we avoid the conversion from a rotation matrix?
    UnitQuaternion::<T>::from_rotation_matrix(self) / rhs;
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: &'a Rotation<T, 3>, rhs: UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(self) / rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    ;
    self: Rotation<T, 3>, rhs: &'b UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(&self) / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    ;
    self: Rotation<T, 3>, rhs: UnitQuaternion<T>,
    Output = UnitQuaternion<T>;
    UnitQuaternion::<T>::from_rotation_matrix(&self) / rhs; );

// UnitQuaternion × Vector
/// Rotates a 3D vector using this unit quaternion.
///
/// This operation applies the 3D rotation represented by the unit quaternion to the input vector.
/// It's an efficient way to transform vectors without converting to a rotation matrix.
///
/// # Example
/// ```
/// # use nalgebra::{UnitQuaternion, Vector3};
/// # use std::f32::consts::PI;
/// // 90° rotation around the z-axis
/// let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
///
/// // Rotate the x-axis vector
/// let v = Vector3::new(1.0, 0.0, 0.0);
/// let rotated = rotation * v;
///
/// // The x-axis should now point along the y-axis
/// assert_relative_eq!(rotated, Vector3::new(0.0, 1.0, 0.0), epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`UnitQuaternion::transform_point`]: For rotating points in 3D space
quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: &'a UnitQuaternion<T>, rhs: &'b Vector<T, Const<3>, SB>,
    Output = Vector3<T>;
    {
        let two: T = crate::convert(2.0f64);
        let t = self.as_ref().vector().cross(rhs) * two;
        let cross = self.as_ref().vector().cross(&t);

        t * self.as_ref().scalar() + cross + rhs
    };
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: &'a UnitQuaternion<T>, rhs: Vector<T, U3, SB>,
    Output = Vector3<T>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: UnitQuaternion<T>, rhs: &'b Vector<T, U3, SB>,
    Output = Vector3<T>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: UnitQuaternion<T>, rhs: Vector<T, U3, SB>,
    Output = Vector3<T>;
    &self * &rhs; );

// UnitQuaternion × Point
/// Rotates a 3D point using this unit quaternion.
///
/// This operation applies the 3D rotation to a point in space. Note that quaternions
/// only represent rotations (not translations), so the origin remains fixed.
///
/// # Example
/// ```
/// # use nalgebra::{UnitQuaternion, Point3, Vector3};
/// # use std::f32::consts::PI;
/// // 180° rotation around the z-axis
/// let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI);
///
/// // Rotate a point
/// let point = Point3::new(1.0, 2.0, 3.0);
/// let rotated = rotation * point;
///
/// // x and y are negated, z unchanged
/// assert_relative_eq!(rotated, Point3::new(-1.0, -2.0, 3.0), epsilon = 1.0e-6);
/// ```
///
/// # See Also
/// - [`Isometry::transform_point`]: For combining rotation and translation
quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: &'b Point3<T>,
    Output = Point3<T>;
    Point3::from(self * &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: &'a UnitQuaternion<T>, rhs: Point3<T>,
    Output = Point3<T>;
    Point3::from(self * rhs.coords);
    'a);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: &'b Point3<T>,
    Output = Point3<T>;
    Point3::from(self * &rhs.coords);
    'b);

quaternion_op_impl!(
    Mul, mul;
    ;
    self: UnitQuaternion<T>, rhs: Point3<T>,
    Output = Point3<T>;
    Point3::from(self * rhs.coords); );

// UnitQuaternion × Unit<Vector>
quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: &'a UnitQuaternion<T>, rhs: &'b Unit<Vector<T, U3, SB>>,
    Output = Unit<Vector3<T>>;
    Unit::new_unchecked(self * rhs.as_ref());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: &'a UnitQuaternion<T>, rhs: Unit<Vector<T, U3, SB>>,
    Output = Unit<Vector3<T>>;
    Unit::new_unchecked(self * rhs.into_inner());
    'a);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: UnitQuaternion<T>, rhs: &'b Unit<Vector<T, U3, SB>>,
    Output = Unit<Vector3<T>>;
    Unit::new_unchecked(self * rhs.as_ref());
    'b);

quaternion_op_impl!(
    Mul, mul;
    SB: Storage<T, Const<3>> ;
    self: UnitQuaternion<T>, rhs: Unit<Vector<T, U3, SB>>,
    Output = Unit<Vector3<T>>;
    Unit::new_unchecked(self * rhs.into_inner()); );

macro_rules! scalar_op_impl(
    ($($Op: ident, $op: ident, $OpAssign: ident, $op_assign: ident);* $(;)*) => {$(
        impl<T: SimdRealField> $Op<T> for Quaternion<T>
         where T::Element: SimdRealField {
            type Output = Quaternion<T>;

            #[inline]
            fn $op(self, n: T) -> Self::Output {
                Quaternion::from(self.coords.$op(n))
            }
        }

        impl<'a, T: SimdRealField> $Op<T> for &'a Quaternion<T>
         where T::Element: SimdRealField {
            type Output = Quaternion<T>;

            #[inline]
            fn $op(self, n: T) -> Self::Output {
                Quaternion::from((&self.coords).$op(n))
            }
        }

        impl<T: SimdRealField> $OpAssign<T> for Quaternion<T>
         where T::Element: SimdRealField {

            #[inline]
            fn $op_assign(&mut self, n: T) {
                self.coords.$op_assign(n)
            }
        }
    )*}
);

scalar_op_impl!(
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl Mul<Quaternion<$T>> for $T {
            type Output = Quaternion<$T>;

            #[inline]
            fn mul(self, right: Quaternion<$T>) -> Self::Output {
                Quaternion::from(self * right.coords)
            }
        }

        impl<'b> Mul<&'b Quaternion<$T>> for $T {
            type Output = Quaternion<$T>;

            #[inline]
            fn mul(self, right: &'b Quaternion<$T>) -> Self::Output {
                Quaternion::from(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(f32, f64);

/// Negates all components of the quaternion.
///
/// For a unit quaternion, negation represents the same rotation. This is because
/// both `q` and `-q` represent the same 3D rotation.
///
/// # Example
/// ```
/// # use nalgebra::Quaternion;
/// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let neg_q = -q;
/// assert_eq!(neg_q, Quaternion::new(-1.0, -2.0, -3.0, -4.0));
/// ```
impl<T: SimdRealField> Neg for Quaternion<T>
where
    T::Element: SimdRealField,
{
    type Output = Quaternion<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-self.coords)
    }
}

impl<T: SimdRealField> Neg for &Quaternion<T>
where
    T::Element: SimdRealField,
{
    type Output = Quaternion<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-&self.coords)
    }
}

macro_rules! quaternion_op_impl(
    ($OpAssign: ident, $op_assign: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr_2021; $($lives: tt),*) => {
        impl<$($lives ,)* T: SimdRealField> $OpAssign<$Rhs> for $Lhs
            where T::Element: SimdRealField {

            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action
            }
        }
    }
);

// Quaternion += Quaternion
quaternion_op_impl!(
    AddAssign, add_assign;
    self: Quaternion<T>, rhs: &'b Quaternion<T>;
    self.coords += &rhs.coords;
    'b);

quaternion_op_impl!(
    AddAssign, add_assign;
    self: Quaternion<T>, rhs: Quaternion<T>;
    self.coords += rhs.coords; );

// Quaternion -= Quaternion
quaternion_op_impl!(
    SubAssign, sub_assign;
    self: Quaternion<T>, rhs: &'b Quaternion<T>;
    self.coords -= &rhs.coords;
    'b);

quaternion_op_impl!(
    SubAssign, sub_assign;
    self: Quaternion<T>, rhs: Quaternion<T>;
    self.coords -= rhs.coords; );

// Quaternion ×= Quaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    self: Quaternion<T>, rhs: &'b Quaternion<T>;
    {
        let res = &*self * rhs;
        // TODO: will this be optimized away?
        self.coords.copy_from(&res.coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    self: Quaternion<T>, rhs: Quaternion<T>;
    *self *= &rhs; );

// UnitQuaternion ×= UnitQuaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    self: UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    self: UnitQuaternion<T>, rhs: UnitQuaternion<T>;
    *self *= &rhs; );

// UnitQuaternion ÷= UnitQuaternion
quaternion_op_impl!(
    DivAssign, div_assign;
    self: UnitQuaternion<T>, rhs: &'b UnitQuaternion<T>;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    self: UnitQuaternion<T>, rhs: UnitQuaternion<T>;
    *self /= &rhs; );

// UnitQuaternion ×= Rotation
quaternion_op_impl!(
    MulAssign, mul_assign;
    self: UnitQuaternion<T>, rhs: &'b Rotation<T, 3>;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    self: UnitQuaternion<T>, rhs: Rotation<T, 3>;
    *self *= &rhs; );

// UnitQuaternion ÷= Rotation
quaternion_op_impl!(
    DivAssign, div_assign;
    self: UnitQuaternion<T>, rhs: &'b Rotation<T, 3>;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    self: UnitQuaternion<T>, rhs: Rotation<T, 3>;
    *self /= &rhs; );
