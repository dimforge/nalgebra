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
 * UnitQuaternion × RotationBase   -> UnitQuaternion
 * RotationBase   × UnitQuaternion -> UnitQuaternion
 *
 * UnitQuaternion ÷ UnitQuaternion
 * UnitQuaternion ÷ RotationBase   -> UnitQuaternion
 * RotationBase   ÷ UnitQuaternion -> UnitQuaternion
 *
 *
 * UnitQuaternion × PointBase
 * UnitQuaternion × ColumnVector
 * UnitQuaternion × Unit<ColumnVector>
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
 * UnitQuaternion ×= RotationBase
 *
 * UnitQuaternion ÷= UnitQuaternion
 * UnitQuaternion ÷= RotationBase
 *
 * FIXME: RotationBase ×= UnitQuaternion
 * FIXME: RotationBase ÷= UnitQuaternion
 *
 */

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, Mul, MulAssign, Div, DivAssign, Sub, SubAssign};

use alga::general::Real;

use core::{ColumnVector, OwnedColumnVector, Unit};
use core::storage::{Storage, StorageMut};
use core::allocator::Allocator;
use core::dimension::{U1, U3, U4};

use geometry::{QuaternionBase, OwnedQuaternionBase, UnitQuaternionBase, OwnedUnitQuaternionBase,
               PointBase, OwnedPoint, RotationBase};

impl<N, S> Index<usize> for QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &N {
        &self.coords[i]
    }
}

impl<N, S> IndexMut<usize> for QuaternionBase<N, S>
    where N: Real,
          S: StorageMut<N, U4, U1> {

    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut N {
        &mut self.coords[i]
    }
}

macro_rules! quaternion_op_impl(
    ($Op: ident, $op: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N, SA, SB> $Op<$Rhs> for $Lhs
            where N: Real,
                  SA: Storage<N, $LhsRDim, $LhsCDim>,
                  SB: Storage<N, $RhsRDim, $RhsCDim>,
                  $(SA::Alloc: Allocator<N, $VDimA, U1>,
                    SB::Alloc: Allocator<N, $VDimB, U1>)* {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);


// Quaternion + Quaternion
quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(&self.coords + &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SB::Alloc>;
    QuaternionBase::from_vector(&self.coords + rhs.coords);
    'a);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(self.coords + &rhs.coords);
    'b);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(self.coords + rhs.coords);
    );


// Quaternion - Quaternion
quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(&self.coords - &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SB::Alloc>;
    QuaternionBase::from_vector(&self.coords - rhs.coords);
    'a);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(self.coords - &rhs.coords);
    'b);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::from_vector(self.coords - rhs.coords);
    );


// Quaternion × Quaternion
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    QuaternionBase::new(
        self[3] * rhs[3] - self[0] * rhs[0] - self[1] * rhs[1] - self[2] * rhs[2],
        self[3] * rhs[0] + self[0] * rhs[3] + self[1] * rhs[2] - self[2] * rhs[1],
        self[3] * rhs[1] - self[0] * rhs[2] + self[1] * rhs[3] + self[2] * rhs[0],
        self[3] * rhs[2] + self[0] * rhs[1] - self[1] * rhs[0] + self[2] * rhs[3]);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>, Output = OwnedQuaternionBase<N, SA::Alloc>;
    &self * &rhs;
    );

// UnitQuaternion × UnitQuaternion
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    UnitQuaternionBase::new_unchecked(self.quaternion() * rhs.quaternion());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    &self * &rhs;
    );

// UnitQuaternion ÷ UnitQuaternion
quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    self * rhs.inverse();
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    self / &rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    &self / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>, Output = OwnedUnitQuaternionBase<N, SA::Alloc>;
    &self / &rhs;
    );

// UnitQuaternion × RotationBase
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    self * OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self * OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self * OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self * OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(&rhs);
    );

// UnitQuaternion ÷ RotationBase
quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    // FIXME: can we avoid the conversion to a rotation matrix?
    self / OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self / OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self / OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB>,
    Output = OwnedUnitQuaternionBase<N, SA::Alloc> => U3, U3;
    self / OwnedUnitQuaternionBase::<N, SA::Alloc>::from_rotation_matrix(&rhs);
    );

// RotationBase × UnitQuaternion
quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: &'a RotationBase<N, U3, SA>, rhs: &'b UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(self) * rhs;
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: &'a RotationBase<N, U3, SA>, rhs: UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(self) * rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: RotationBase<N, U3, SA>, rhs: &'b UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(&self) * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: RotationBase<N, U3, SA>, rhs: UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(&self) * rhs;
    );

// RotationBase ÷ UnitQuaternion
quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: &'a RotationBase<N, U3, SA>, rhs: &'b UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(self) / rhs;
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: &'a RotationBase<N, U3, SA>, rhs: UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(self) / rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: RotationBase<N, U3, SA>, rhs: &'b UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(&self) / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: RotationBase<N, U3, SA>, rhs: UnitQuaternionBase<N, SB>,
    Output = OwnedUnitQuaternionBase<N, SB::Alloc> => U3, U3;
    OwnedUnitQuaternionBase::<N, SB::Alloc>::from_rotation_matrix(&self) / rhs;
    );

// UnitQuaternion × Vector
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b ColumnVector<N, U3, SB>,
    Output = OwnedColumnVector<N, U3, SA::Alloc> => U3, U4;
    {
        let two: N = ::convert(2.0f64);
        let t = self.as_ref().vector().cross(rhs) * two;
        let cross = self.as_ref().vector().cross(&t);

        t * self.as_ref().scalar() + cross + rhs
    };
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: ColumnVector<N, U3, SB>,
    Output = OwnedColumnVector<N, U3, SA::Alloc> => U3, U4;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b ColumnVector<N, U3, SB>,
    Output = OwnedColumnVector<N, U3, SA::Alloc> => U3, U4;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: ColumnVector<N, U3, SB>,
    Output = OwnedColumnVector<N, U3, SA::Alloc> => U3, U4;
    &self * &rhs;
    );

// UnitQuaternion × PointBase
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b PointBase<N, U3, SB>,
    Output = OwnedPoint<N, U3, SA::Alloc> => U3, U4;
    PointBase::from_coordinates(self * &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: PointBase<N, U3, SB>,
    Output = OwnedPoint<N, U3, SA::Alloc> => U3, U4;
    PointBase::from_coordinates(self * rhs.coords);
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b PointBase<N, U3, SB>,
    Output = OwnedPoint<N, U3, SA::Alloc> => U3, U4;
    PointBase::from_coordinates(self * &rhs.coords);
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: PointBase<N, U3, SB>,
    Output = OwnedPoint<N, U3, SA::Alloc> => U3, U4;
    PointBase::from_coordinates(self * rhs.coords);
    );

// UnitQuaternion × Unit<Vector>
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: &'b Unit<ColumnVector<N, U3, SB>>,
    Output = Unit<OwnedColumnVector<N, U3, SA::Alloc>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternionBase<N, SA>, rhs: Unit<ColumnVector<N, U3, SB>>,
    Output = Unit<OwnedColumnVector<N, U3, SA::Alloc>> => U3, U4;
    Unit::new_unchecked(self * rhs.unwrap());
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b Unit<ColumnVector<N, U3, SB>>,
    Output = Unit<OwnedColumnVector<N, U3, SA::Alloc>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, rhs: Unit<ColumnVector<N, U3, SB>>,
    Output = Unit<OwnedColumnVector<N, U3, SA::Alloc>> => U3, U4;
    Unit::new_unchecked(self * rhs.unwrap());
    );



macro_rules! scalar_op_impl(
    ($($Op: ident, $op: ident, $OpAssign: ident, $op_assign: ident);* $(;)*) => {$(
        impl<N, S> $Op<N> for QuaternionBase<N, S>
            where N: Real,
                  S: Storage<N, U4, U1> {
            type Output = OwnedQuaternionBase<N, S::Alloc>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                QuaternionBase::from_vector(self.coords.$op(n))
            }
        }

        impl<'a, N, S> $Op<N> for &'a QuaternionBase<N, S>
            where N: Real,
                  S: Storage<N, U4, U1> {
            type Output = OwnedQuaternionBase<N, S::Alloc>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                QuaternionBase::from_vector((&self.coords).$op(n))
            }
        }

        impl<N, S> $OpAssign<N> for QuaternionBase<N, S>
            where N: Real,
                  S: StorageMut<N, U4, U1> {

            #[inline]
            fn $op_assign(&mut self, n: N) {
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
        impl<S> Mul<QuaternionBase<$T, S>> for $T
            where S: Storage<$T, U4, U1> {
            type Output = OwnedQuaternionBase<$T, S::Alloc>;

            #[inline]
            fn mul(self, right: QuaternionBase<$T, S>) -> Self::Output {
                QuaternionBase::from_vector(self * right.coords)
            }
        }

        impl<'b, S> Mul<&'b QuaternionBase<$T, S>> for $T
            where S: Storage<$T, U4, U1> {
            type Output = OwnedQuaternionBase<$T, S::Alloc>;

            #[inline]
            fn mul(self, right: &'b QuaternionBase<$T, S>) -> Self::Output {
                QuaternionBase::from_vector(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(f32, f64);

impl<N, S> Neg for QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    type Output = OwnedQuaternionBase<N, S::Alloc>;

    #[inline]
    fn neg(self) -> Self::Output {
        QuaternionBase::from_vector(-self.coords)
    }
}

impl<'a, N, S> Neg for &'a QuaternionBase<N, S>
    where N: Real,
          S: Storage<N, U4, U1> {
    type Output = OwnedQuaternionBase<N, S::Alloc>;

    #[inline]
    fn neg(self) -> Self::Output {
        QuaternionBase::from_vector(-&self.coords)
    }
}

macro_rules! quaternion_op_impl(
    ($OpAssign: ident, $op_assign: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N, SA, SB> $OpAssign<$Rhs> for $Lhs
            where N: Real,
                  SA: StorageMut<N, $LhsRDim, $LhsCDim>,
                  SB: Storage<N, $RhsRDim, $RhsCDim>,
                  $(SA::Alloc: Allocator<N, $VDimA, U1> + Allocator<N, U4, U1>,
                    //                                    ^^^^^^^^^^^^^^^^^^^^
                    //                                    XXX: For some reasons, the compiler needs
                    //                                    this bound to compile UnitQuat *= RotationBase.
                    //                                    Though in theory this bound is already
                    //                                    inherited from `SA: StorageMut`…
                    SB::Alloc: Allocator<N, $VDimB, U1>)* {

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
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>;
    self.coords += &rhs.coords;
    'b);

quaternion_op_impl!(
    AddAssign, add_assign;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>;
    self.coords += rhs.coords; );


// Quaternion -= Quaternion
quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>;
    self.coords -= &rhs.coords;
    'b);

quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>;
    self.coords -= rhs.coords; );

// Quaternion ×= Quaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: &'b QuaternionBase<N, SB>;
    {
        let res = &*self * rhs;
        // FIXME: will this be optimized away?
        self.coords.copy_from(&res.coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: QuaternionBase<N, SA>, rhs: QuaternionBase<N, SB>;
    *self *= &rhs; );

// UnitQuaternion ×= UnitQuaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>;
    *self *= &rhs; );

// UnitQuaternion ÷= UnitQuaternion
quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: &'b UnitQuaternionBase<N, SB>;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternionBase<N, SA>, rhs: UnitQuaternionBase<N, SB>;
    *self /= &rhs; );

// UnitQuaternion ×= RotationBase
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB> => U3, U3;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB> => U3, U3;
    *self *= &rhs; );

// UnitQuaternion ÷= RotationBase
quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: &'b RotationBase<N, U3, SB> => U3, U3;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternionBase<N, SA>, rhs: RotationBase<N, U3, SB> => U3, U3;
    *self /= &rhs; );
