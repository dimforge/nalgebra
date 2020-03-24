use num::{One, Zero};
use std::ops::{Div, DivAssign, Mul, MulAssign};

use simba::scalar::{ClosedAdd, ClosedMul};
use simba::simd::SimdRealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, U1, U2, U3, U4};
use crate::base::{DefaultAllocator, Unit, VectorN};
use crate::Scalar;

use crate::geometry::{
    AbstractRotation, Isometry, Point, Rotation, Translation, UnitComplex, UnitQuaternion,
};

// FIXME: there are several cloning of rotations that we could probably get rid of (but we didn't
// yet because that would require to add a bound like `where for<'a, 'b> &'a R: Mul<&'b R, Output = R>`
// which is quite ugly.

/*
 *
 * In this file, we provide:
 * =========================
 *
 *
 * (Operators)
 *
 * Isometry × Isometry
 * Isometry × R
 *
 *
 * Isometry ÷ Isometry
 * Isometry ÷ R
 *
 * Isometry × Point
 * Isometry × Vector
 * Isometry × Unit<Vector>
 *
 *
 * Isometry    × Translation
 * Translation × Isometry
 * Translation × R           -> Isometry<R>
 *
 * NOTE: The following are provided explicitly because we can't have R × Isometry.
 * Rotation       × Isometry<Rotation>
 * UnitQuaternion × Isometry<UnitQuaternion>
 *
 * Rotation       ÷ Isometry<Rotation>
 * UnitQuaternion ÷ Isometry<UnitQuaternion>
 *
 * Rotation       × Translation -> Isometry<Rotation>
 * UnitQuaternion × Translation -> Isometry<UnitQuaternion>
 *
 *
 * (Assignment Operators)
 *
 * Isometry ×= Translation
 *
 * Isometry ×= Isometry
 * Isometry ×= R
 *
 * Isometry ÷= Isometry
 * Isometry ÷= R
 *
 */

macro_rules! isometry_binop_impl(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: SimdRealField, D: DimName, R> $Op<$Rhs> for $Lhs
            where N::Element: SimdRealField,
                  R: AbstractRotation<N, D>,
                  DefaultAllocator: Allocator<N, D> {
            type Output = $Output;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! isometry_binop_impl_all(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {
        isometry_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        isometry_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        isometry_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        isometry_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);

macro_rules! isometry_binop_assign_impl_all(
    ($OpAssign: ident, $op_assign: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty;
     [val] => $action_val: expr;
     [ref] => $action_ref: expr;) => {
        impl<N: SimdRealField, D: DimName, R> $OpAssign<$Rhs> for $Lhs
            where N::Element: SimdRealField,
                  R: AbstractRotation<N, D>,
                  DefaultAllocator: Allocator<N, D> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action_val
            }
        }

        impl<'b, N: SimdRealField, D: DimName, R> $OpAssign<&'b $Rhs> for $Lhs
            where N::Element: SimdRealField,
                  R: AbstractRotation<N, D>,
                  DefaultAllocator: Allocator<N, D> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: &'b $Rhs) {
                $action_ref
            }
        }
    }
);

// Isometry × Isometry
// Isometry ÷ Isometry
isometry_binop_impl_all!(
    Mul, mul;
    self: Isometry<N, D, R>, rhs: Isometry<N, D, R>, Output = Isometry<N, D, R>;
    [val val] => &self * &rhs;
    [ref val] => self * &rhs;
    [val ref] => &self * rhs;
    [ref ref] => {
        let shift = self.rotation.transform_vector(&rhs.translation.vector);

        Isometry::from_parts(Translation::from(&self.translation.vector + shift),
                             self.rotation.clone() * rhs.rotation.clone()) // FIXME: too bad we have to clone.
    };
);

isometry_binop_impl_all!(
    Div, div;
    self: Isometry<N, D, R>, rhs: Isometry<N, D, R>, Output = Isometry<N, D, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Isometry ×= Translation
isometry_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Isometry<N, D, R>, rhs: Translation<N, D>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.rotation.transform_vector(&rhs.vector);
        self.translation.vector += shift;
    };
);

// Isometry ×= Isometry
// Isometry ÷= Isometry
isometry_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Isometry<N, D, R>, rhs: Isometry<N, D, R>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.rotation.transform_vector(&rhs.translation.vector);
        self.translation.vector += shift;
        self.rotation *= rhs.rotation.clone();
    };
);

isometry_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: Isometry<N, D, R>, rhs: Isometry<N, D, R>;
    [val] => *self /= &rhs;
    [ref] => *self *= rhs.inverse();
);

// Isometry ×= R
// Isometry ÷= R
md_assign_impl_all!(
    MulAssign, mul_assign where N: SimdRealField for N::Element: SimdRealField;
    (D, U1), (D, D) for D: DimName;
    self: Isometry<N, D, Rotation<N, D>>, rhs: Rotation<N, D>;
    [val] => self.rotation *= rhs;
    [ref] => self.rotation *= rhs.clone();
);

md_assign_impl_all!(
    DivAssign, div_assign where N: SimdRealField for N::Element: SimdRealField;
    (D, U1), (D, D) for D: DimName;
    self: Isometry<N, D, Rotation<N, D>>, rhs: Rotation<N, D>;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);

md_assign_impl_all!(
    MulAssign, mul_assign where N: SimdRealField for N::Element: SimdRealField;
    (U3, U3), (U3, U3) for;
    self: Isometry<N, U3, UnitQuaternion<N>>, rhs: UnitQuaternion<N>;
    [val] => self.rotation *= rhs;
    [ref] => self.rotation *= rhs.clone();
);

md_assign_impl_all!(
    DivAssign, div_assign where N: SimdRealField for N::Element: SimdRealField;
    (U3, U3), (U3, U3) for;
    self: Isometry<N, U3, UnitQuaternion<N>>, rhs: UnitQuaternion<N>;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);

md_assign_impl_all!(
    MulAssign, mul_assign where N: SimdRealField for N::Element: SimdRealField;
    (U2, U2), (U2, U2) for;
    self: Isometry<N, U2, UnitComplex<N>>, rhs: UnitComplex<N>;
    [val] => self.rotation *= rhs;
    [ref] => self.rotation *= rhs.clone();
);

md_assign_impl_all!(
    DivAssign, div_assign where N: SimdRealField for N::Element: SimdRealField;
    (U2, U2), (U2, U2) for;
    self: Isometry<N, U2, UnitComplex<N>>, rhs: UnitComplex<N>;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);

// Isometry × Point
isometry_binop_impl_all!(
    Mul, mul;
    self: Isometry<N, D, R>, right: Point<N, D>, Output = Point<N, D>;
    [val val] => self.translation  * self.rotation.transform_point(&right);
    [ref val] => &self.translation * self.rotation.transform_point(&right);
    [val ref] => self.translation  * self.rotation.transform_point(right);
    [ref ref] => &self.translation * self.rotation.transform_point(right);
);

// Isometry × Vector
isometry_binop_impl_all!(
    Mul, mul;
    // FIXME: because of `transform_vector`, we cant use a generic storage type for the rhs vector,
    // i.e., right: Vector<N, D, S> where S: Storage<N, D>.
    self: Isometry<N, D, R>, right: VectorN<N, D>, Output = VectorN<N, D>;
    [val val] => self.rotation.transform_vector(&right);
    [ref val] => self.rotation.transform_vector(&right);
    [val ref] => self.rotation.transform_vector(right);
    [ref ref] => self.rotation.transform_vector(right);
);

// Isometry × Unit<Vector>
isometry_binop_impl_all!(
    Mul, mul;
    // FIXME: because of `transform_vector`, we cant use a generic storage type for the rhs vector,
    // i.e., right: Vector<N, D, S> where S: Storage<N, D>.
    self: Isometry<N, D, R>, right: Unit<VectorN<N, D>>, Output = Unit<VectorN<N, D>>;
    [val val] => Unit::new_unchecked(self.rotation.transform_vector(right.as_ref()));
    [ref val] => Unit::new_unchecked(self.rotation.transform_vector(right.as_ref()));
    [val ref] => Unit::new_unchecked(self.rotation.transform_vector(right.as_ref()));
    [ref ref] => Unit::new_unchecked(self.rotation.transform_vector(right.as_ref()));
);

// Isometry × Translation
isometry_binop_impl_all!(
    Mul, mul;
    self: Isometry<N, D, R>, right: Translation<N, D>, Output = Isometry<N, D, R>;
    [val val] => &self * &right;
    [ref val] => self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let new_tr = &self.translation.vector + self.rotation.transform_vector(&right.vector);
        Isometry::from_parts(Translation::from(new_tr), self.rotation.clone())
    };
);

// Translation × Isometry
isometry_binop_impl_all!(
    Mul, mul;
    self: Translation<N, D>, right: Isometry<N, D, R>, Output = Isometry<N, D, R>;
    [val val] => Isometry::from_parts(self * right.translation, right.rotation);
    [ref val] => Isometry::from_parts(self * &right.translation, right.rotation);
    [val ref] => Isometry::from_parts(self * &right.translation, right.rotation.clone());
    [ref ref] => Isometry::from_parts(self * &right.translation, right.rotation.clone());
);

macro_rules! isometry_from_composition_impl(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: SimdRealField $(, $Dims: $DimsBound)*> $Op<$Rhs> for $Lhs
            where N::Element: SimdRealField,
                  DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2> {
            type Output = $Output;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! isometry_from_composition_impl_all(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

        isometry_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        isometry_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        isometry_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        isometry_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);

// Rotation × Translation
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: Rotation<N, D>, right: Translation<N, D>, Output = Isometry<N, D, Rotation<N, D>>;
    [val val] => Isometry::from_parts(Translation::from(&self * right.vector),  self);
    [ref val] => Isometry::from_parts(Translation::from(self * right.vector),   self.clone());
    [val ref] => Isometry::from_parts(Translation::from(&self * &right.vector), self);
    [ref ref] => Isometry::from_parts(Translation::from(self * &right.vector),  self.clone());
);

// UnitQuaternion × Translation
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, right: Translation<N, U3>,
    Output = Isometry<N, U3, UnitQuaternion<N>>;
    [val val] => Isometry::from_parts(Translation::from(&self *  right.vector), self);
    [ref val] => Isometry::from_parts(Translation::from( self *  right.vector), self.clone());
    [val ref] => Isometry::from_parts(Translation::from(&self * &right.vector), self);
    [ref ref] => Isometry::from_parts(Translation::from( self * &right.vector), self.clone());
);

// Isometry × Rotation
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: Isometry<N, D, Rotation<N, D>>, rhs: Rotation<N, D>,
    Output = Isometry<N, D, Rotation<N, D>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation * rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation * rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs.clone());
);

// Rotation × Isometry
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: Rotation<N, D>, right: Isometry<N, D, Rotation<N, D>>,
    Output = Isometry<N, D, Rotation<N, D>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self * &right.translation.vector;
        Isometry::from_parts(Translation::from(shift), self * &right.rotation)
    };
);

// Isometry ÷ Rotation
isometry_from_composition_impl_all!(
    Div, div;
    (D, D), (D, U1) for D: DimName;
    self: Isometry<N, D, Rotation<N, D>>, rhs: Rotation<N, D>,
    Output = Isometry<N, D, Rotation<N, D>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation / rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation / rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs.clone());
);

// Rotation ÷ Isometry
isometry_from_composition_impl_all!(
    Div, div;
    (D, D), (D, U1) for D: DimName;
    self: Rotation<N, D>, right: Isometry<N, D, Rotation<N, D>>,
    Output = Isometry<N, D, Rotation<N, D>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// Isometry × UnitQuaternion
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: Isometry<N, U3, UnitQuaternion<N>>, rhs: UnitQuaternion<N>,
    Output = Isometry<N, U3, UnitQuaternion<N>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation * rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation * rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs.clone());
);

// UnitQuaternion × Isometry
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, right: Isometry<N, U3, UnitQuaternion<N>>,
    Output = Isometry<N, U3, UnitQuaternion<N>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self * &right.translation.vector;
        Isometry::from_parts(Translation::from(shift), self * &right.rotation)
    };
);

// Isometry ÷ UnitQuaternion
isometry_from_composition_impl_all!(
    Div, div;
    (U4, U1), (U3, U1);
    self: Isometry<N, U3, UnitQuaternion<N>>, rhs: UnitQuaternion<N>,
    Output = Isometry<N, U3, UnitQuaternion<N>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation / rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation / rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs.clone());
);

// UnitQuaternion ÷ Isometry
isometry_from_composition_impl_all!(
    Div, div;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, right: Isometry<N, U3, UnitQuaternion<N>>,
    Output = Isometry<N, U3, UnitQuaternion<N>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// Translation × Rotation
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: Translation<N, D>, right: Rotation<N, D>, Output = Isometry<N, D, Rotation<N, D>>;
    [val val] => Isometry::from_parts(self, right);
    [ref val] => Isometry::from_parts(self.clone(), right);
    [val ref] => Isometry::from_parts(self, right.clone());
    [ref ref] => Isometry::from_parts(self.clone(), right.clone());
);

// Translation × UnitQuaternion
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: Translation<N, U3>, right: UnitQuaternion<N>, Output = Isometry<N, U3, UnitQuaternion<N>>;
    [val val] => Isometry::from_parts(self, right);
    [ref val] => Isometry::from_parts(self.clone(), right);
    [val ref] => Isometry::from_parts(self, right.clone());
    [ref ref] => Isometry::from_parts(self.clone(), right.clone());
);

// Isometry × UnitComplex
isometry_from_composition_impl_all!(
    Mul, mul;
    (U2, U1), (U2, U1);
    self: Isometry<N, U2, UnitComplex<N>>, rhs: UnitComplex<N>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation * rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation * rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() * rhs.clone());
);

// Isometry ÷ UnitComplex
isometry_from_composition_impl_all!(
    Div, div;
    (U2, U1), (U2, U1);
    self: Isometry<N, U2, UnitComplex<N>>, rhs: UnitComplex<N>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(self.translation, self.rotation / rhs);
    [ref val] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs); // FIXME: do not clone.
    [val ref] => Isometry::from_parts(self.translation, self.rotation / rhs.clone());
    [ref ref] => Isometry::from_parts(self.translation.clone(), self.rotation.clone() / rhs.clone());
);
