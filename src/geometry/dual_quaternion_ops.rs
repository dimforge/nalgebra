// /*
//  * This file provides:
//  * ===================
//  *
//  *
//  * (DualQuaternion)
//  *
//  * Index<usize>
//  * IndexMut<usize>
//  * DualQuaternion × DualQuaternion
//  * DualQuaternion + DualQuaternion
//  * DualQuaternion - DualQuaternion
//  * -DualQuaternion
//  * DualQuaternion × Scalar
//  * DualQuaternion ÷ Scalar
//  * Scalar × DualQuaternion
//  *
//  * (Unit DualQuaternion)
//  * UnitDualQuaternion × UnitDualQuaternion
//  * UnitDualQuaternion × Rotation       -> UnitDualQuaternion
//  * Rotation       × UnitDualQuaternion -> UnitDualQuaternion
//  *
//  * UnitDualQuaternion ÷ UnitDualQuaternion
//  * UnitDualQuaternion ÷ Rotation       -> UnitDualQuaternion
//  * Rotation       ÷ UnitDualQuaternion -> UnitDualQuaternion
//  *
//  *
//  * UnitDualQuaternion × Point
//  * UnitDualQuaternion × Vector
//  * UnitDualQuaternion × Unit<Vector>
//  *
//  * NOTE: -UnitDualQuaternion is already provided by `Unit<T>`.
//  *
//  *
//  * (Assignment Operators)
//  *
//  * DualQuaternion ×= Scalar
//  * DualQuaternion ×= DualQuaternion
//  * DualQuaternion += DualQuaternion
//  * DualQuaternion -= DualQuaternion
//  *
//  * UnitDualQuaternion ×= UnitDualQuaternion
//  * UnitDualQuaternion ×= Rotation
//  *
//  * UnitDualQuaternion ÷= UnitDualQuaternion
//  * UnitDualQuaternion ÷= Rotation
//  *
//  * FIXME: Rotation ×= UnitDualQuaternion
//  * FIXME: Rotation ÷= UnitDualQuaternion
//  *
//  */

// use std::ops::{
//     Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
// };

// use alga::general::Real;

// use base::allocator::Allocator;
// use base::dimension::{U1, U3, U4};
// use base::storage::Storage;
// use base::{DefaultAllocator, Unit, Vector, Vector3};

// use geometry::{Point3, DualQuaternion, Rotation, UnitDualQuaternion};

// impl<N: Real> Index<usize> for DualQuaternion<N> {
//     type Output = N;

//     #[inline]
//     fn index(&self, i: usize) -> &N {
//         &self.coords[i]
//     }
// }

// impl<N: Real> IndexMut<usize> for DualQuaternion<N> {
//     #[inline]
//     fn index_mut(&mut self, i: usize) -> &mut N {
//         &mut self.coords[i]
//     }
// }

// macro_rules! quaternion_op_impl(
//     ($Op: ident, $op: ident;
//      ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident)
//      $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
//      $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty $(=> $VDimA: ty, $VDimB: ty)*;
//      $action: expr; $($lives: tt),*) => {
//         impl<$($lives ,)* N: Real $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
//             where DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
//                                     Allocator<N, $RhsRDim, $RhsCDim> {
//             type Output = $Result;

//             #[inline]
//             fn $op($lhs, $rhs: $Rhs) -> Self::Output {
//                 $action
//             }
//         }
//     }
// );

// // DualQuaternion + DualQuaternion
// quaternion_op_impl!(
//     Add, add;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(&self.coords + &rhs.coords);
//     'a, 'b);

// quaternion_op_impl!(
//     Add, add;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(&self.coords + rhs.coords);
//     'a);

// quaternion_op_impl!(
//     Add, add;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(self.coords + &rhs.coords);
//     'b);

// quaternion_op_impl!(
//     Add, add;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(self.coords + rhs.coords);
//     );

// // DualQuaternion - DualQuaternion
// quaternion_op_impl!(
//     Sub, sub;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(&self.coords - &rhs.coords);
//     'a, 'b);

// quaternion_op_impl!(
//     Sub, sub;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(&self.coords - rhs.coords);
//     'a);

// quaternion_op_impl!(
//     Sub, sub;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(self.coords - &rhs.coords);
//     'b);

// quaternion_op_impl!(
//     Sub, sub;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::from(self.coords - rhs.coords);
//     );

// // DualQuaternion × DualQuaternion
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     DualQuaternion::new(
//         self[3] * rhs[3] - self[0] * rhs[0] - self[1] * rhs[1] - self[2] * rhs[2],
//         self[3] * rhs[0] + self[0] * rhs[3] + self[1] * rhs[2] - self[2] * rhs[1],
//         self[3] * rhs[1] - self[0] * rhs[2] + self[1] * rhs[3] + self[2] * rhs[0],
//         self[3] * rhs[2] + self[0] * rhs[1] - self[1] * rhs[0] + self[2] * rhs[3]);
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     self * &rhs;
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
//     &self * rhs;
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
//     &self * &rhs;
//     );

// // UnitDualQuaternion × UnitDualQuaternion
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     UnitDualQuaternion::new_unchecked(self.quaternion() * rhs.quaternion());
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     self * &rhs;
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     &self * rhs;
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     &self * &rhs;
//     );

// // UnitDualQuaternion ÷ UnitDualQuaternion
// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U4, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     self * rhs.inverse();
//     'a, 'b);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U4, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     self / &rhs;
//     'a);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     &self / rhs;
//     'b);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
//     &self / &rhs;
//     );

// // UnitDualQuaternion × Rotation
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U3);
//     self: &'a UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     // FIXME: can we avoid the conversion from a rotation matrix?
//     self * UnitDualQuaternion::<N>::from_rotation_matrix(rhs);
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U3);
//     self: &'a UnitDualQuaternion<N>, rhs: Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self * UnitDualQuaternion::<N>::from_rotation_matrix(&rhs);
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self * UnitDualQuaternion::<N>::from_rotation_matrix(rhs);
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self * UnitDualQuaternion::<N>::from_rotation_matrix(&rhs);
//     );

// // UnitDualQuaternion ÷ Rotation
// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U3, U3);
//     self: &'a UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     // FIXME: can we avoid the conversion to a rotation matrix?
//     self / UnitDualQuaternion::<N>::from_rotation_matrix(rhs);
//     'a, 'b);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U3, U3);
//     self: &'a UnitDualQuaternion<N>, rhs: Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self / UnitDualQuaternion::<N>::from_rotation_matrix(&rhs);
//     'a);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self / UnitDualQuaternion::<N>::from_rotation_matrix(rhs);
//     'b);

// quaternion_op_impl!(
//     Div, div;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: Rotation<N, U3>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     self / UnitDualQuaternion::<N>::from_rotation_matrix(&rhs);
//     );

// // Rotation × UnitDualQuaternion
// quaternion_op_impl!(
//     Mul, mul;
//     (U3, U3), (U4, U1);
//     self: &'a Rotation<N, U3>, rhs: &'b UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     // FIXME: can we avoid the conversion from a rotation matrix?
//     UnitDualQuaternion::<N>::from_rotation_matrix(self) * rhs;
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U3, U3), (U4, U1);
//     self: &'a Rotation<N, U3>, rhs: UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(self) * rhs;
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U3, U3), (U4, U1);
//     self: Rotation<N, U3>, rhs: &'b UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(&self) * rhs;
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U3, U3), (U4, U1);
//     self: Rotation<N, U3>, rhs: UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(&self) * rhs;
//     );

// // Rotation ÷ UnitDualQuaternion
// quaternion_op_impl!(
//     Div, div;
//     (U3, U3), (U4, U1);
//     self: &'a Rotation<N, U3>, rhs: &'b UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     // FIXME: can we avoid the conversion from a rotation matrix?
//     UnitDualQuaternion::<N>::from_rotation_matrix(self) / rhs;
//     'a, 'b);

// quaternion_op_impl!(
//     Div, div;
//     (U3, U3), (U4, U1);
//     self: &'a Rotation<N, U3>, rhs: UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(self) / rhs;
//     'a);

// quaternion_op_impl!(
//     Div, div;
//     (U3, U3), (U4, U1);
//     self: Rotation<N, U3>, rhs: &'b UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(&self) / rhs;
//     'b);

// quaternion_op_impl!(
//     Div, div;
//     (U3, U3), (U4, U1);
//     self: Rotation<N, U3>, rhs: UnitDualQuaternion<N>,
//     Output = UnitDualQuaternion<N> => U3, U3;
//     UnitDualQuaternion::<N>::from_rotation_matrix(&self) / rhs;
//     );

// // UnitDualQuaternion × Vector
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: &'a UnitDualQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
//     Output = Vector3<N> => U3, U4;
//     {
//         let two: N = ::convert(2.0f64);
//         let t = self.as_ref().vector().cross(rhs) * two;
//         let cross = self.as_ref().vector().cross(&t);

//         t * self.as_ref().scalar() + cross + rhs
//     };
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: &'a UnitDualQuaternion<N>, rhs: Vector<N, U3, SB>,
//     Output = Vector3<N> => U3, U4;
//     self * &rhs;
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: UnitDualQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
//     Output = Vector3<N> => U3, U4;
//     &self * rhs;
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: UnitDualQuaternion<N>, rhs: Vector<N, U3, SB>,
//     Output = Vector3<N> => U3, U4;
//     &self * &rhs;
//     );

// // UnitDualQuaternion × Point
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: &'b Point3<N>,
//     Output = Point3<N> => U3, U4;
//     Point3::from(self * &rhs.coords);
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1);
//     self: &'a UnitDualQuaternion<N>, rhs: Point3<N>,
//     Output = Point3<N> => U3, U4;
//     Point3::from(self * rhs.coords);
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1);
//     self: UnitDualQuaternion<N>, rhs: &'b Point3<N>,
//     Output = Point3<N> => U3, U4;
//     Point3::from(self * &rhs.coords);
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1);
//     self: UnitDualQuaternion<N>, rhs: Point3<N>,
//     Output = Point3<N> => U3, U4;
//     Point3::from(self * rhs.coords);
//     );

// // UnitDualQuaternion × Unit<Vector>
// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: &'a UnitDualQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
//     Output = Unit<Vector3<N>> => U3, U4;
//     Unit::new_unchecked(self * rhs.as_ref());
//     'a, 'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: &'a UnitDualQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
//     Output = Unit<Vector3<N>> => U3, U4;
//     Unit::new_unchecked(self * rhs.into_inner());
//     'a);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: UnitDualQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
//     Output = Unit<Vector3<N>> => U3, U4;
//     Unit::new_unchecked(self * rhs.as_ref());
//     'b);

// quaternion_op_impl!(
//     Mul, mul;
//     (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
//     self: UnitDualQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
//     Output = Unit<Vector3<N>> => U3, U4;
//     Unit::new_unchecked(self * rhs.into_inner());
//     );

// macro_rules! scalar_op_impl(
//     ($($Op: ident, $op: ident, $OpAssign: ident, $op_assign: ident);* $(;)*) => {$(
//         impl<N: Real> $Op<N> for DualQuaternion<N> {
//             type Output = DualQuaternion<N>;

//             #[inline]
//             fn $op(self, n: N) -> Self::Output {
//                 DualQuaternion::from(self.coords.$op(n))
//             }
//         }

//         impl<'a, N: Real> $Op<N> for &'a DualQuaternion<N> {
//             type Output = DualQuaternion<N>;

//             #[inline]
//             fn $op(self, n: N) -> Self::Output {
//                 DualQuaternion::from((&self.coords).$op(n))
//             }
//         }

//         impl<N: Real> $OpAssign<N> for DualQuaternion<N> {

//             #[inline]
//             fn $op_assign(&mut self, n: N) {
//                 self.coords.$op_assign(n)
//             }
//         }
//     )*}
// );

// scalar_op_impl!(
//     Mul, mul, MulAssign, mul_assign;
//     Div, div, DivAssign, div_assign;
// );

// macro_rules! left_scalar_mul_impl(
//     ($($T: ty),* $(,)*) => {$(
//         impl Mul<DualQuaternion<$T>> for $T {
//             type Output = DualQuaternion<$T>;

//             #[inline]
//             fn mul(self, right: DualQuaternion<$T>) -> Self::Output {
//                 DualQuaternion::from(self * right.coords)
//             }
//         }

//         impl<'b> Mul<&'b DualQuaternion<$T>> for $T {
//             type Output = DualQuaternion<$T>;

//             #[inline]
//             fn mul(self, right: &'b DualQuaternion<$T>) -> Self::Output {
//                 DualQuaternion::from(self * &right.coords)
//             }
//         }
//     )*}
// );

// left_scalar_mul_impl!(f32, f64);

// impl<N: Real> Neg for DualQuaternion<N> {
//     type Output = DualQuaternion<N>;

//     #[inline]
//     fn neg(self) -> Self::Output {
//         DualQuaternion::from(-self.coords)
//     }
// }

// impl<'a, N: Real> Neg for &'a DualQuaternion<N> {
//     type Output = DualQuaternion<N>;

//     #[inline]
//     fn neg(self) -> Self::Output {
//         DualQuaternion::from(-&self.coords)
//     }
// }

// macro_rules! quaternion_op_impl(
//     ($OpAssign: ident, $op_assign: ident;
//      ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident);
//      $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty $(=> $VDimA: ty, $VDimB: ty)*;
//      $action: expr; $($lives: tt),*) => {
//         impl<$($lives ,)* N: Real> $OpAssign<$Rhs> for $Lhs
//             where DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
//                                     Allocator<N, $RhsRDim, $RhsCDim> {

//             #[inline]
//             fn $op_assign(&mut $lhs, $rhs: $Rhs) {
//                 $action
//             }
//         }
//     }
// );

// // DualQuaternion += DualQuaternion
// quaternion_op_impl!(
//     AddAssign, add_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
//     self.coords += &rhs.coords;
//     'b);

// quaternion_op_impl!(
//     AddAssign, add_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>;
//     self.coords += rhs.coords; );

// // DualQuaternion -= DualQuaternion
// quaternion_op_impl!(
//     SubAssign, sub_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
//     self.coords -= &rhs.coords;
//     'b);

// quaternion_op_impl!(
//     SubAssign, sub_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>;
//     self.coords -= rhs.coords; );

// // DualQuaternion ×= DualQuaternion
// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
//     {
//         let res = &*self * rhs;
//         // FIXME: will this be optimized away?
//         self.coords.copy_from(&res.coords);
//     };
//     'b);

// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U4, U1);
//     self: DualQuaternion<N>, rhs: DualQuaternion<N>;
//     *self *= &rhs; );

// // UnitDualQuaternion ×= UnitDualQuaternion
// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>;
//     {
//         let res = &*self * rhs;
//         self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
//     };
//     'b);

// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>;
//     *self *= &rhs; );

// // UnitDualQuaternion ÷= UnitDualQuaternion
// quaternion_op_impl!(
//     DivAssign, div_assign;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>;
//     {
//         let res = &*self / rhs;
//         self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
//     };
//     'b);

// quaternion_op_impl!(
//     DivAssign, div_assign;
//     (U4, U1), (U4, U1);
//     self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>;
//     *self /= &rhs; );

// // UnitDualQuaternion ×= Rotation
// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3> => U3, U3;
//     {
//         let res = &*self * rhs;
//         self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
//     };
//     'b);

// quaternion_op_impl!(
//     MulAssign, mul_assign;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: Rotation<N, U3> => U3, U3;
//     *self *= &rhs; );

// // UnitDualQuaternion ÷= Rotation
// quaternion_op_impl!(
//     DivAssign, div_assign;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: &'b Rotation<N, U3> => U3, U3;
//     {
//         let res = &*self / rhs;
//         self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
//     };
//     'b);

// quaternion_op_impl!(
//     DivAssign, div_assign;
//     (U4, U1), (U3, U3);
//     self: UnitDualQuaternion<N>, rhs: Rotation<N, U3> => U3, U3;
//     *self /= &rhs; );
