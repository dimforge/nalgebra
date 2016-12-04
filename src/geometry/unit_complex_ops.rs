use std::ops::{Mul, MulAssign, Div, DivAssign};

use alga::general::Real;
use core::{Unit, ColumnVector, OwnedColumnVector};
use core::dimension::{U1, U2};
use core::storage::Storage;
use geometry::{UnitComplex, RotationBase, PointBase, OwnedPoint};

/*
 * This file provides:
 * ===================
 *
 * UnitComplex  × UnitComplex
 * UnitComplex  × RotationBase -> UnitComplex
 * RotationBase × UnitComplex  -> UnitComplex
 *
 * UnitComplex  ÷ UnitComplex
 * UnitComplex  ÷ RotationBase -> UnitComplex
 * RotationBase ÷ UnitComplex  -> UnitComplex
 *
 *
 * UnitComplex × PointBase
 * UnitComplex × ColumnVector
 * UnitComplex × Unit<T>
 *
 * NOTE: -UnitComplex is already provided by `Unit<T>`.
 *
 *
 * (Assignment Operators)
 *
 * UnitComplex  ×= UnitComplex
 * UnitComplex  ×= RotationBase
 *
 * UnitComplex  ÷= UnitComplex
 * UnitComplex  ÷= RotationBase
 *
 * FIXME: RotationBase ×= UnitComplex
 * FIXME: RotationBase ÷= UnitComplex
 *
 */

// UnitComplex × UnitComplex
impl<N: Real> Mul<UnitComplex<N>> for UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.unwrap() * rhs.unwrap())
    }
}

impl<'a, N: Real> Mul<UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.complex() * rhs.unwrap())
    }
}

impl<'b, N: Real> Mul<&'b UnitComplex<N>> for UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.unwrap() * rhs.complex())
    }
}

impl<'a, 'b, N: Real> Mul<&'b UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.complex() * rhs.complex())
    }
}

// UnitComplex ÷ UnitComplex
impl<N: Real> Div<UnitComplex<N>> for UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.unwrap() * rhs.conjugate().unwrap())
    }
}

impl<'a, N: Real> Div<UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.complex() * rhs.conjugate().unwrap())
    }
}

impl<'b, N: Real> Div<&'b UnitComplex<N>> for UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.unwrap() * rhs.conjugate().unwrap())
    }
}

impl<'a, 'b, N: Real> Div<&'b UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> UnitComplex<N> {
        Unit::new_unchecked(self.complex() * rhs.conjugate().unwrap())
    }
}

macro_rules! complex_op_impl(
    ($Op: ident, $op: ident;
     ($RDim: ident, $CDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N, S> $Op<$Rhs> for $Lhs
            where N: Real,
                  S: Storage<N, $RDim, $CDim> {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! complex_op_impl_all(
    ($Op: ident, $op: ident;
     ($RDim: ident, $CDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim);
                     $lhs: $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_val_val; );

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim);
                     $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_ref_val; 'a);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim);
                     $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_val_ref; 'b);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim);
                     $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_ref_ref; 'a, 'b);


    }
);


// UnitComplex × RotationBase
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: UnitComplex<N>, rhs: RotationBase<N, U2, S>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs);
);

// UnitComplex ÷ RotationBase
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: UnitComplex<N>, rhs: RotationBase<N, U2, S>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs).inverse();
);


// RotationBase × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: RotationBase<N, U2, S>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs;
);

// RotationBase ÷ UnitComplex
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: RotationBase<N, U2, S>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs.inverse();
);

// UnitComplex × PointBase
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: PointBase<N, U2, S>, Output = OwnedPoint<N, U2, S::Alloc>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => PointBase::from_coordinates(self * &rhs.coords);
);

// UnitComplex × ColumnVector
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: ColumnVector<N, U2, S>, Output = OwnedColumnVector<N, U2, S::Alloc>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let i = self.as_ref().im;
        let r = self.as_ref().re;
        OwnedColumnVector::<_, U2, S::Alloc>::new(r * rhs[0] - i * rhs[0], i * rhs[1] + r * rhs[1])
    };
);

// UnitComplex × Unit<Vector>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Unit<ColumnVector<N, U2, S>>, Output = Unit<OwnedColumnVector<N, U2, S::Alloc>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Unit::new_unchecked(self * rhs.as_ref());
);

// UnitComplex ×= UnitComplex
impl<N: Real> MulAssign<UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self * rhs
    }
}

impl<'b, N: Real> MulAssign<&'b UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self * rhs
    }
}

// UnitComplex /= UnitComplex
impl<N: Real> DivAssign<UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self / rhs
    }
}

impl<'b, N: Real> DivAssign<&'b UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self / rhs
    }
}


// UnitComplex ×= RotationBase
impl<N: Real, S: Storage<N, U2, U2>> MulAssign<RotationBase<N, U2, S>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: RotationBase<N, U2, S>) {
        *self = &*self * rhs
    }
}

impl<'b, N: Real, S: Storage<N, U2, U2>> MulAssign<&'b RotationBase<N, U2, S>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &'b RotationBase<N, U2, S>) {
        *self = &*self * rhs
    }
}

// UnitComplex ÷= RotationBase
impl<N: Real, S: Storage<N, U2, U2>> DivAssign<RotationBase<N, U2, S>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: RotationBase<N, U2, S>) {
        *self = &*self / rhs
    }
}

impl<'b, N: Real, S: Storage<N, U2, U2>> DivAssign<&'b RotationBase<N, U2, S>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: &'b RotationBase<N, U2, S>) {
        *self = &*self / rhs
    }
}
