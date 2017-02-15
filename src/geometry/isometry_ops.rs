use std::ops::{Mul, MulAssign, Div, DivAssign};

use alga::general::Real;
use alga::linear::Rotation;

use core::ColumnVector;
use core::dimension::{DimName, U1, U3, U4};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{PointBase, RotationBase, IsometryBase, TranslationBase, UnitQuaternionBase};

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
 * IsometryBase × IsometryBase
 * IsometryBase × R
 *
 *
 * IsometryBase ÷ IsometryBase
 * IsometryBase ÷ R
 *
 * IsometryBase × PointBase
 * IsometryBase × ColumnVector
 *
 *
 * IsometryBase    × TranslationBase
 * TranslationBase × IsometryBase
 * TranslationBase × R           -> IsometryBase<R>
 *
 * NOTE: The following are provided explicitly because we can't have R × IsometryBase.
 * RotationBase   × IsometryBase<RotationBase>
 * UnitQuaternion × IsometryBase<UnitQuaternion>
 *
 * RotationBase   ÷ IsometryBase<RotationBase>
 * UnitQuaternion ÷ IsometryBase<UnitQuaternion>
 *
 * RotationBase   × TranslationBase -> IsometryBase<RotationBase>
 * UnitQuaternion × TranslationBase -> IsometryBase<UnitQuaternion>
 *
 *
 * (Assignment Operators)
 *
 * IsometryBase ×= TranslationBase
 *
 * IsometryBase ×= IsometryBase
 * IsometryBase ×= R
 *
 * IsometryBase ÷= IsometryBase
 * IsometryBase ÷= R
 *
 */


macro_rules! isometry_binop_impl(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N, D: DimName, S, R> $Op<$Rhs> for $Lhs
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
                  R: Rotation<PointBase<N, D, S>>,
                  S::Alloc: OwnedAllocator<N, D, U1, S> {
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
        impl<N, D: DimName, S, R> $OpAssign<$Rhs> for $Lhs
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
                  R: Rotation<PointBase<N, D, S>>,
                  S::Alloc: OwnedAllocator<N, D, U1, S> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action_val
            }
        }

        impl<'b, N, D: DimName, S, R> $OpAssign<&'b $Rhs> for $Lhs
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
                  R: Rotation<PointBase<N, D, S>>,
                  S::Alloc: OwnedAllocator<N, D, U1, S> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: &'b $Rhs) {
                $action_ref
            }
        }
    }
);

// IsometryBase × IsometryBase
// IsometryBase ÷ IsometryBase
isometry_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>, Output = IsometryBase<N, D, S, R>;
    [val val] => &self * &rhs;
    [ref val] => self * &rhs;
    [val ref] => &self * rhs;
    [ref ref] => {
        let shift = self.rotation.transform_vector(&rhs.translation.vector);

        IsometryBase::from_parts(TranslationBase::from_vector(&self.translation.vector + shift),
                             self.rotation.clone() * rhs.rotation.clone()) // FIXME: too bad we have to clone.
    };
);


isometry_binop_impl_all!(
    Div, div;
    self: IsometryBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>, Output = IsometryBase<N, D, S, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);


// IsometryBase ×= TranslationBase
isometry_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: IsometryBase<N, D, S, R>, rhs: TranslationBase<N, D, S>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.rotation.transform_vector(&rhs.vector);
        self.translation.vector += shift;
    };
);

// IsometryBase ×= IsometryBase
// IsometryBase ÷= IsometryBase
isometry_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: IsometryBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.rotation.transform_vector(&rhs.translation.vector);
        self.translation.vector += shift;
        self.rotation *= rhs.rotation.clone();
    };
);

isometry_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: IsometryBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>;
    [val] => *self /= &rhs;
    [ref] => *self *= rhs.inverse();
);

// IsometryBase ×= R
// IsometryBase ÷= R
isometry_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: IsometryBase<N, D, S, R>, rhs: R;
    [val] => self.rotation *= rhs;
    [ref] => self.rotation *= rhs.clone();
);

isometry_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: IsometryBase<N, D, S, R>, rhs: R;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);


// IsometryBase × R
// IsometryBase ÷ R
isometry_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, rhs: R, Output = IsometryBase<N, D, S, R>;
    [val val] => IsometryBase::from_parts(self.translation, self.rotation * rhs);
    [ref val] => IsometryBase::from_parts(self.translation.clone(), self.rotation.clone() * rhs); // FIXME: do not clone.
    [val ref] => IsometryBase::from_parts(self.translation, self.rotation * rhs.clone());
    [ref ref] => IsometryBase::from_parts(self.translation.clone(), self.rotation.clone() * rhs.clone());
);


isometry_binop_impl_all!(
    Div, div;
    self: IsometryBase<N, D, S, R>, rhs: R, Output = IsometryBase<N, D, S, R>;
    [val val] => IsometryBase::from_parts(self.translation, self.rotation / rhs);
    [ref val] => IsometryBase::from_parts(self.translation.clone(), self.rotation.clone() / rhs);
    [val ref] => IsometryBase::from_parts(self.translation, self.rotation / rhs.clone());
    [ref ref] => IsometryBase::from_parts(self.translation.clone(), self.rotation.clone() / rhs.clone());
);


// IsometryBase × PointBase
isometry_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, right: PointBase<N, D, S>, Output = PointBase<N, D, S>;
    [val val] => self.translation * self.rotation.transform_point(&right);
    [ref val] => &self.translation * self.rotation.transform_point(&right);
    [val ref] => self.translation * self.rotation.transform_point(right);
    [ref ref] => &self.translation * self.rotation.transform_point(right);
);


// IsometryBase × Vector
isometry_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, right: ColumnVector<N, D, S>, Output = ColumnVector<N, D, S>;
    [val val] => self.rotation.transform_vector(&right);
    [ref val] => self.rotation.transform_vector(&right);
    [val ref] => self.rotation.transform_vector(right);
    [ref ref] => self.rotation.transform_vector(right);
);


// IsometryBase × TranslationBase
isometry_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, right: TranslationBase<N, D, S>, Output = IsometryBase<N, D, S, R>;
    [val val] => &self * &right;
    [ref val] => self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let new_tr = &self.translation.vector + self.rotation.transform_vector(&right.vector);
        IsometryBase::from_parts(TranslationBase::from_vector(new_tr), self.rotation.clone())
    };
);

// TranslationBase × IsometryBase
isometry_binop_impl_all!(
    Mul, mul;
    self: TranslationBase<N, D, S>, right: IsometryBase<N, D, S, R>, Output = IsometryBase<N, D, S, R>;
    [val val] => IsometryBase::from_parts(self * right.translation, right.rotation);
    [ref val] => IsometryBase::from_parts(self * &right.translation, right.rotation);
    [val ref] => IsometryBase::from_parts(self * &right.translation, right.rotation.clone());
    [ref ref] => IsometryBase::from_parts(self * &right.translation, right.rotation.clone());
);


// TranslationBase × R
isometry_binop_impl_all!(
    Mul, mul;
    self: TranslationBase<N, D, S>, right: R, Output = IsometryBase<N, D, S, R>;
    [val val] => IsometryBase::from_parts(self, right);
    [ref val] => IsometryBase::from_parts(self.clone(), right);
    [val ref] => IsometryBase::from_parts(self, right.clone());
    [ref ref] => IsometryBase::from_parts(self.clone(), right.clone());
);




macro_rules! isometry_from_composition_impl(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N $(, $Dims: $DimsBound)*, SA, SB> $Op<$Rhs> for $Lhs
            where N: Real,
                  SA: OwnedStorage<N, $R1, $C1>,
                  SB: OwnedStorage<N, $R2, $C2, Alloc = SA::Alloc>,
                  SA::Alloc: OwnedAllocator<N, $R1, $C1, SA>,
                  SB::Alloc: OwnedAllocator<N, $R2, $C2, SB> {
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


// RotationBase × TranslationBase
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: RotationBase<N, D, SA>, right: TranslationBase<N, D, SB>, Output = IsometryBase<N, D, SB, RotationBase<N, D, SA>>;
    [val val] => IsometryBase::from_parts(TranslationBase::from_vector(&self * right.vector),  self);
    [ref val] => IsometryBase::from_parts(TranslationBase::from_vector(self * right.vector),   self.clone());
    [val ref] => IsometryBase::from_parts(TranslationBase::from_vector(&self * &right.vector), self);
    [ref ref] => IsometryBase::from_parts(TranslationBase::from_vector(self * &right.vector),  self.clone());
);


// UnitQuaternionBase × TranslationBase
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, right: TranslationBase<N, U3, SB>,
    Output = IsometryBase<N, U3, SB, UnitQuaternionBase<N, SA>>;
    [val val] => IsometryBase::from_parts(TranslationBase::from_vector(&self *  right.vector), self);
    [ref val] => IsometryBase::from_parts(TranslationBase::from_vector( self *  right.vector), self.clone());
    [val ref] => IsometryBase::from_parts(TranslationBase::from_vector(&self * &right.vector), self);
    [ref ref] => IsometryBase::from_parts(TranslationBase::from_vector( self * &right.vector), self.clone());
);

// RotationBase × IsometryBase
isometry_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: RotationBase<N, D, SA>, right: IsometryBase<N, D, SB, RotationBase<N, D, SA>>,
    Output = IsometryBase<N, D, SB, RotationBase<N, D, SA>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self * &right.translation.vector;
        IsometryBase::from_parts(TranslationBase::from_vector(shift), self * &right.rotation)
    };
);

// RotationBase ÷ IsometryBase
isometry_from_composition_impl_all!(
    Div, div;
    (D, D), (D, U1) for D: DimName;
    self: RotationBase<N, D, SA>, right: IsometryBase<N, D, SB, RotationBase<N, D, SA>>,
    Output = IsometryBase<N, D, SB, RotationBase<N, D, SA>>;
    // FIXME: don't call iverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);


// UnitQuaternion × IsometryBase
isometry_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, right: IsometryBase<N, U3, SB, UnitQuaternionBase<N, SA>>,
    Output = IsometryBase<N, U3, SB, UnitQuaternionBase<N, SA>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self * &right.translation.vector;
        IsometryBase::from_parts(TranslationBase::from_vector(shift), self * &right.rotation)
    };
);


// UnitQuaternion ÷ IsometryBase
isometry_from_composition_impl_all!(
    Div, div;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, right: IsometryBase<N, U3, SB, UnitQuaternionBase<N, SA>>,
    Output = IsometryBase<N, U3, SB, UnitQuaternionBase<N, SA>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);
