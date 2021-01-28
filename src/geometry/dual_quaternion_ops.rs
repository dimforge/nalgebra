/*
 * This file provides:
 *
 * NOTE: Work in progress https://github.com/dimforge/nalgebra/issues/487
 *
 * (Dual Quaternion)
 *
 * Index<usize>
 * IndexMut<usize>
 *
 * (Assignment Operators)
 *
 * DualQuaternion × Scalar
 * DualQuaternion × DualQuaternion
 * DualQuaternion + DualQuaternion
 * DualQuaternion - DualQuaternion
 *
 * ---
 *
 * References:
 *   Multiplication:
 *   - https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
 */

use crate::base::storage::Storage;
use crate::{
    Allocator, DefaultAllocator, DualQuaternion, Isometry3, Point, Point3, Quaternion,
    SimdRealField, Translation3, Unit, UnitDualQuaternion, UnitQuaternion, Vector, Vector3, U1, U3,
    U4,
};
use std::mem;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

impl<N: SimdRealField> AsRef<[N; 8]> for DualQuaternion<N> {
    #[inline]
    fn as_ref(&self) -> &[N; 8] {
        unsafe { mem::transmute(self) }
    }
}

impl<N: SimdRealField> AsMut<[N; 8]> for DualQuaternion<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [N; 8] {
        unsafe { mem::transmute(self) }
    }
}

impl<N: SimdRealField> Index<usize> for DualQuaternion<N> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.as_ref()[i]
    }
}

impl<N: SimdRealField> IndexMut<usize> for DualQuaternion<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut N {
        &mut self.as_mut()[i]
    }
}

impl<N: SimdRealField> Neg for DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    type Output = DualQuaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        DualQuaternion::from_real_and_dual(-self.real, -self.dual)
    }
}

impl<'a, N: SimdRealField> Neg for &'a DualQuaternion<N>
where
    N::Element: SimdRealField,
{
    type Output = DualQuaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        DualQuaternion::from_real_and_dual(-&self.real, -&self.dual)
    }
}

impl<N: SimdRealField> Neg for UnitDualQuaternion<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitDualQuaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        UnitDualQuaternion::new_unchecked(-self.into_inner())
    }
}

impl<'a, N: SimdRealField> Neg for &'a UnitDualQuaternion<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitDualQuaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        UnitDualQuaternion::new_unchecked(-self.as_ref())
    }
}

macro_rules! dual_quaternion_op_impl(
    ($Op: ident, $op: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident)
     $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: SimdRealField $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where N::Element: SimdRealField,
                  DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
                                    Allocator<N, $RhsRDim, $RhsCDim> {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

// DualQuaternion + DualQuaternion
dual_quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        &self.real + &rhs.real,
        &self.dual + &rhs.dual,
    );
    'a, 'b);

dual_quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        &self.real + rhs.real,
        &self.dual + rhs.dual,
    );
    'a);

dual_quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        self.real + &rhs.real,
        self.dual + &rhs.dual,
    );
    'b);

dual_quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        self.real + rhs.real,
        self.dual + rhs.dual,
    ); );

// DualQuaternion - DualQuaternion
dual_quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        &self.real - &rhs.real,
        &self.dual - &rhs.dual,
    );
    'a, 'b);

dual_quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        &self.real - rhs.real,
        &self.dual - rhs.dual,
    );
    'a);

dual_quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        self.real - &rhs.real,
        self.dual - &rhs.dual,
    );
    'b);

dual_quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        self.real - rhs.real,
        self.dual - rhs.dual,
    ); );

// DualQuaternion × DualQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    DualQuaternion::from_real_and_dual(
        &self.real * &rhs.real,
        &self.real * &rhs.dual + &self.dual * &rhs.real,
    );
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    self * &rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>, Output = DualQuaternion<N>;
    &self * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>, Output = DualQuaternion<N>;
    &self * &rhs; );

// UnitDualQuaternion × UnitDualQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    UnitDualQuaternion::new_unchecked(self.as_ref() * rhs.as_ref());
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    self * &rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    &self * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    &self * &rhs; );

// UnitDualQuaternion ÷ UnitDualQuaternion
dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    self / &rhs;
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    &self / rhs;
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>, Output = UnitDualQuaternion<N>;
    &self / &rhs; );

// UnitDualQuaternion × UnitQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U1, U4;
    self * UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(rhs.into_inner()));
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    self * UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(rhs.into_inner()));
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    self * UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(rhs.into_inner()));
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    self * UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(rhs.into_inner())););

// UnitQuaternion × UnitDualQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U1, U4;
    UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(self.into_inner())) * rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(self.into_inner())) * rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(self.into_inner())) * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    UnitDualQuaternion::<N>::new_unchecked(DualQuaternion::from_real(self.into_inner())) * rhs;);

// UnitDualQuaternion ÷ UnitQuaternion
dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U1, U4;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_rotation(rhs.inverse()) };
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitDualQuaternion<N>, rhs: UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_rotation(rhs.inverse()) };
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_rotation(rhs.inverse()) };
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_rotation(rhs.inverse()) };);

// UnitQuaternion ÷ UnitDualQuaternion
dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U1, U4;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        UnitDualQuaternion::<N>::new_unchecked(
            DualQuaternion::from_real(self.into_inner())
        ) * rhs.inverse()
    }; 'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        UnitDualQuaternion::<N>::new_unchecked(
            DualQuaternion::from_real(self.into_inner())
        ) * rhs.inverse()
    }; 'a);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        UnitDualQuaternion::<N>::new_unchecked(
            DualQuaternion::from_real(self.into_inner())
        ) * rhs.inverse()
    }; 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U3;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        UnitDualQuaternion::<N>::new_unchecked(
            DualQuaternion::from_real(self.into_inner())
        ) * rhs.inverse()
    };);

// UnitDualQuaternion × Translation3
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_parts(rhs.clone(), UnitQuaternion::identity());
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitDualQuaternion<N>, rhs: Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_parts(rhs, UnitQuaternion::identity());
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: &'b Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_parts(rhs.clone(), UnitQuaternion::identity());
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_parts(rhs, UnitQuaternion::identity()); );

// UnitDualQuaternion ÷ Translation3
dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_parts(rhs.inverse(), UnitQuaternion::identity()) };
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitDualQuaternion<N>, rhs: Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_parts(rhs.inverse(), UnitQuaternion::identity()) };
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: &'b Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_parts(rhs.inverse(), UnitQuaternion::identity()) };
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: Translation3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    #[allow(clippy::suspicious_arithmetic_impl)]
    { self * UnitDualQuaternion::<N>::from_parts(rhs.inverse(), UnitQuaternion::identity()) };);

// Translation3 × UnitDualQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: &'b Translation3<N>, rhs: &'a UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self.clone(), UnitQuaternion::identity()) * rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: &'a Translation3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self.clone(), UnitQuaternion::identity()) * rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: Translation3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self, UnitQuaternion::identity()) * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: Translation3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self, UnitQuaternion::identity()) * rhs;);

// Translation3 ÷ UnitDualQuaternion
dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: &'b Translation3<N>, rhs: &'a UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self.clone(), UnitQuaternion::identity()) / rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: &'a Translation3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self.clone(), UnitQuaternion::identity()) / rhs;
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: Translation3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self, UnitQuaternion::identity()) / rhs;
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: Translation3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_parts(self, UnitQuaternion::identity()) / rhs;);

// UnitDualQuaternion × Isometry3
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_isometry(rhs);
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitDualQuaternion<N>, rhs: Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_isometry(&rhs);
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: &'b Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_isometry(rhs);
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self * UnitDualQuaternion::<N>::from_isometry(&rhs); );

// UnitDualQuaternion ÷ Isometry
dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    // TODO: can we avoid the conversion to a rotation matrix?
    self / UnitDualQuaternion::<N>::from_isometry(rhs);
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitDualQuaternion<N>, rhs: Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self / UnitDualQuaternion::<N>::from_isometry(&rhs);
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: &'b Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self / UnitDualQuaternion::<N>::from_isometry(rhs);
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitDualQuaternion<N>, rhs: Isometry3<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    self / UnitDualQuaternion::<N>::from_isometry(&rhs); );

// Isometry × UnitDualQuaternion
dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: &'a Isometry3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(self) * rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: &'a Isometry3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(self) * rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: Isometry3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(&self) * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U3, U1), (U4, U1);
    self: Isometry3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(&self) * rhs; );

// Isometry ÷ UnitDualQuaternion
dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: &'a Isometry3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    // TODO: can we avoid the conversion from a rotation matrix?
    UnitDualQuaternion::<N>::from_isometry(self) / rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: &'a Isometry3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(self) / rhs;
    'a);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: Isometry3<N>, rhs: &'b UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(&self) / rhs;
    'b);

dual_quaternion_op_impl!(
    Div, div;
    (U3, U1), (U4, U1);
    self: Isometry3<N>, rhs: UnitDualQuaternion<N>,
    Output = UnitDualQuaternion<N> => U3, U1;
    UnitDualQuaternion::<N>::from_isometry(&self) / rhs; );

// UnitDualQuaternion × Vector
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitDualQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U1;
    Unit::new_unchecked(self.as_ref().real) * rhs;
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitDualQuaternion<N>, rhs: Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U1;
    self * &rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitDualQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U1;
    &self * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitDualQuaternion<N>, rhs: Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U1;
    &self * &rhs; );

// UnitDualQuaternion × Point
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: &'b Point3<N>,
    Output = Point3<N> => U3, U1;
    {
        let two: N = crate::convert(2.0f64);
        let q_point = Quaternion::from_parts(N::zero(), rhs.coords.clone());
        Point::from(
            ((self.as_ref().real * q_point + self.as_ref().dual * two) * self.as_ref().real.conjugate())
                .vector()
                .into_owned(),
        )
    };
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitDualQuaternion<N>, rhs: Point3<N>,
    Output = Point3<N> => U3, U1;
    self * &rhs;
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: &'b Point3<N>,
    Output = Point3<N> => U3, U1;
    &self * rhs;
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: Point3<N>,
    Output = Point3<N> => U3, U1;
    &self * &rhs; );

// UnitDualQuaternion × Unit<Vector>
dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitDualQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'a, 'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitDualQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.into_inner());
    'a);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitDualQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'b);

dual_quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitDualQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.into_inner()); );

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl Mul<DualQuaternion<$T>> for $T {
            type Output = DualQuaternion<$T>;

            #[inline]
            fn mul(self, right: DualQuaternion<$T>) -> Self::Output {
                DualQuaternion::from_real_and_dual(
                    self * right.real,
                    self * right.dual
                )
            }
        }

        impl<'b> Mul<&'b DualQuaternion<$T>> for $T {
            type Output = DualQuaternion<$T>;

            #[inline]
            fn mul(self, right: &'b DualQuaternion<$T>) -> Self::Output {
                DualQuaternion::from_real_and_dual(
                    self * &right.real,
                    self * &right.dual
                )
            }
        }
    )*}
);

left_scalar_mul_impl!(f32, f64);

macro_rules! dual_quaternion_op_impl(
    ($OpAssign: ident, $op_assign: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: SimdRealField> $OpAssign<$Rhs> for $Lhs
            where N::Element: SimdRealField,
                  DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
                                    Allocator<N, $RhsRDim, $RhsCDim> {

            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action
            }
        }
    }
);

// DualQuaternion += DualQuaternion
dual_quaternion_op_impl!(
    AddAssign, add_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
    {
        self.real += &rhs.real;
        self.dual += &rhs.dual;
    };
    'b);

dual_quaternion_op_impl!(
    AddAssign, add_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>;
    {
        self.real += rhs.real;
        self.dual += rhs.dual;
    };);

// DualQuaternion -= DualQuaternion
dual_quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
    {
        self.real -= &rhs.real;
        self.dual -= &rhs.dual;
    };
    'b);

dual_quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>;
    {
        self.real -= rhs.real;
        self.dual -= rhs.dual;
    };);

// DualQuaternion ×= DualQuaternion
dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: &'b DualQuaternion<N>;
    {
        let res = &*self * rhs;
        self.real.coords.copy_from(&res.real.coords);
        self.dual.coords.copy_from(&res.dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: DualQuaternion<N>, rhs: DualQuaternion<N>;
    *self *= &rhs; );

// UnitDualQuaternion ×= UnitDualQuaternion
dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>;
    *self *= &rhs; );

// UnitDualQuaternion ÷= UnitDualQuaternion
dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitDualQuaternion<N>;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitDualQuaternion<N>;
    *self /= &rhs; );

// UnitDualQuaternion ×= UnitQuaternion
dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitQuaternion<N>;
    {
        let res = &*self * UnitDualQuaternion::from_rotation(rhs);
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };);

dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>;
    *self *= rhs.clone(); 'b);

// UnitDualQuaternion ÷= UnitQuaternion
dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b UnitQuaternion<N>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        let res = &*self * UnitDualQuaternion::from_rotation(rhs.inverse());
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: UnitQuaternion<N>;
    *self /= &rhs; );

// UnitDualQuaternion ×= Translation3
dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: Translation3<N>;
    {
        let res = &*self * UnitDualQuaternion::from_parts(rhs, UnitQuaternion::identity());
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };);

dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b Translation3<N>;
    *self *= rhs.clone(); 'b);

// UnitDualQuaternion ÷= Translation3
dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: &'b Translation3<N>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    {
        let res = &*self * UnitDualQuaternion::from_parts(rhs.inverse(), UnitQuaternion::identity());
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitDualQuaternion<N>, rhs: Translation3<N>;
    *self /= &rhs; );

// UnitDualQuaternion ×= Isometry3
dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: &'b Isometry3<N> => U3, U1;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: Isometry3<N> => U3, U1;
    *self *= &rhs; );

// UnitDualQuaternion ÷= Isometry3
dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: &'b Isometry3<N> => U3, U1;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().real.coords.copy_from(&res.as_ref().real.coords);
        self.as_mut_unchecked().dual.coords.copy_from(&res.as_ref().dual.coords);
    };
    'b);

dual_quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U1);
    self: UnitDualQuaternion<N>, rhs: Isometry3<N> => U3, U1;
    *self /= &rhs; );

macro_rules! scalar_op_impl(
    ($($Op: ident, $op: ident, $OpAssign: ident, $op_assign: ident);* $(;)*) => {$(
        impl<N: SimdRealField> $Op<N> for DualQuaternion<N>
         where N::Element: SimdRealField {
            type Output = DualQuaternion<N>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                DualQuaternion::from_real_and_dual(
                    self.real.$op(n),
                    self.dual.$op(n)
                )
            }
        }

        impl<'a, N: SimdRealField> $Op<N> for &'a DualQuaternion<N>
         where N::Element: SimdRealField {
            type Output = DualQuaternion<N>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                DualQuaternion::from_real_and_dual(
                    self.real.$op(n),
                    self.dual.$op(n)
                )
            }
        }

        impl<N: SimdRealField> $OpAssign<N> for DualQuaternion<N>
         where N::Element: SimdRealField {

            #[inline]
            fn $op_assign(&mut self, n: N) {
                self.real.$op_assign(n);
                self.dual.$op_assign(n);
            }
        }
    )*}
);

scalar_op_impl!(
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
);
