use std::ops::{Mul, MulAssign, Div, DivAssign};

use alga::general::Real;
use alga::linear::Rotation;

use core::ColumnVector;
use core::dimension::{DimName, U1, U3, U4};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{PointBase, RotationBase, SimilarityBase, TranslationBase, UnitQuaternionBase, IsometryBase};

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
 * SimilarityBase × SimilarityBase
 * SimilarityBase × R
 * SimilarityBase × IsometryBase
 *
 * IsometryBase × SimilarityBase
 * IsometryBase ÷ SimilarityBase
 *
 *
 * SimilarityBase ÷ SimilarityBase
 * SimilarityBase ÷ R
 * SimilarityBase ÷ IsometryBase
 *
 * SimilarityBase × PointBase
 * SimilarityBase × ColumnVector
 *
 *
 * SimilarityBase  × TranslationBase
 * TranslationBase × SimilarityBase
 *
 * NOTE: The following are provided explicitly because we can't have R × SimilarityBase.
 * RotationBase   × SimilarityBase<RotationBase>
 * UnitQuaternion × SimilarityBase<UnitQuaternion>
 *
 * RotationBase   ÷ SimilarityBase<RotationBase>
 * UnitQuaternion ÷ SimilarityBase<UnitQuaternion>
 *
 * (Assignment Operators)
 *
 * SimilarityBase ×= TranslationBase
 *
 * SimilarityBase ×= SimilarityBase
 * SimilarityBase ×= IsometryBase
 * SimilarityBase ×= R
 *
 * SimilarityBase ÷= SimilarityBase
 * SimilarityBase ÷= IsometryBase
 * SimilarityBase ÷= R
 *
 */


// XXX: code duplication: those macros are the same as for the isometry.
macro_rules! similarity_binop_impl(
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

macro_rules! similarity_binop_impl_all(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {
        similarity_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        similarity_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        similarity_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        similarity_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);

macro_rules! similarity_binop_assign_impl_all(
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

// SimilarityBase × SimilarityBase
// SimilarityBase ÷ SimilarityBase
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let mut res = self * &rhs.isometry;
        res.prepend_scaling_mut(rhs.scaling());
        res
    };
);


similarity_binop_impl_all!(
    Div, div;
    self: SimilarityBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);


// SimilarityBase ×= TranslationBase
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: SimilarityBase<N, D, S, R>, rhs: TranslationBase<N, D, S>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.vector) * self.scaling();
        self.isometry.translation.vector += shift;
    };
);


// SimilarityBase ×= SimilarityBase
// SimilarityBase ÷= SimilarityBase
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: SimilarityBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>;
    [val] => *self *= &rhs;
    [ref] => {
        *self *= &rhs.isometry;
        self.prepend_scaling_mut(rhs.scaling());
    };
);


similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: SimilarityBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>;
    [val] => *self /= &rhs;
    // FIXME: don't invert explicitly.
    [ref] => *self *= rhs.inverse();
);


// SimilarityBase ×= IsometryBase
// SimilarityBase ÷= IsometryBase
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: SimilarityBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.translation.vector) * self.scaling();
        self.isometry.translation.vector += shift;
        self.isometry.rotation *= rhs.rotation.clone();
    };
);


similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: SimilarityBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>;
    [val] => *self /= &rhs;
    // FIXME: don't invert explicitly.
    [ref] => *self *= rhs.inverse();
);


// SimilarityBase ×= R
// SimilarityBase ÷= R
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: SimilarityBase<N, D, S, R>, rhs: R;
    [val] => self.isometry.rotation *= rhs;
    [ref] => self.isometry.rotation *= rhs.clone();
);


similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: SimilarityBase<N, D, S, R>, rhs: R;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);


// SimilarityBase × R
// SimilarityBase ÷ R
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, rhs: R, Output = SimilarityBase<N, D, S, R>;
    [val val] => {
        let scaling = self.scaling();
        SimilarityBase::from_isometry(self.isometry * rhs, scaling)
    };
    [ref val] => SimilarityBase::from_isometry(&self.isometry * rhs, self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        SimilarityBase::from_isometry(self.isometry * rhs, scaling)
    };
    [ref ref] => SimilarityBase::from_isometry(&self.isometry * rhs, self.scaling());
);



similarity_binop_impl_all!(
    Div, div;
    self: SimilarityBase<N, D, S, R>, rhs: R, Output = SimilarityBase<N, D, S, R>;
    [val val] => {
        let scaling = self.scaling();
        SimilarityBase::from_isometry(self.isometry / rhs, scaling)
    };
    [ref val] => SimilarityBase::from_isometry(&self.isometry / rhs, self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        SimilarityBase::from_isometry(self.isometry / rhs, scaling)
    };
    [ref ref] => SimilarityBase::from_isometry(&self.isometry / rhs, self.scaling());
);

// SimilarityBase × IsometryBase
// SimilarityBase ÷ IsometryBase
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => &self * &rhs;
    [ref val] => self * &rhs;
    [val ref] => &self * rhs;
    [ref ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.translation.vector) * self.scaling();
        SimilarityBase::from_parts(
            TranslationBase::from_vector(&self.isometry.translation.vector + shift),
            self.isometry.rotation.clone() * rhs.rotation.clone(),
            self.scaling())
    };
);



similarity_binop_impl_all!(
    Div, div;
    self: SimilarityBase<N, D, S, R>, rhs: IsometryBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// IsometryBase × SimilarityBase
// IsometryBase ÷ SimilarityBase
similarity_binop_impl_all!(
    Mul, mul;
    self: IsometryBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => {
        let scaling = rhs.scaling();
        SimilarityBase::from_isometry(self * rhs.isometry, scaling)
    };
    [ref val] => {
        let scaling = rhs.scaling();
        SimilarityBase::from_isometry(self * rhs.isometry, scaling)
    };
    [val ref] => {
        let scaling = rhs.scaling();
        SimilarityBase::from_isometry(self * &rhs.isometry, scaling)
    };
    [ref ref] => {
        let scaling = rhs.scaling();
        SimilarityBase::from_isometry(self * &rhs.isometry, scaling)
    };
);


similarity_binop_impl_all!(
    Div, div;
    self: IsometryBase<N, D, S, R>, rhs: SimilarityBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);


// SimilarityBase × PointBase
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, right: PointBase<N, D, S>, Output = PointBase<N, D, S>;
    [val val] => {
        let scaling = self.scaling();
        self.isometry.translation * (self.isometry.rotation.transform_point(&right) * scaling)
    };
    [ref val] => &self.isometry.translation * (self.isometry.rotation.transform_point(&right) * self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        self.isometry.translation * (self.isometry.rotation.transform_point(right) * scaling)
    };
    [ref ref] => &self.isometry.translation * (self.isometry.rotation.transform_point(right) * self.scaling());
);


// SimilarityBase × Vector
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, right: ColumnVector<N, D, S>, Output = ColumnVector<N, D, S>;
    [val val] => self.isometry.rotation.transform_vector(&right) * self.scaling();
    [ref val] => self.isometry.rotation.transform_vector(&right) * self.scaling();
    [val ref] => self.isometry.rotation.transform_vector(right) * self.scaling();
    [ref ref] => self.isometry.rotation.transform_vector(right) * self.scaling();
);


// SimilarityBase × TranslationBase
similarity_binop_impl_all!(
    Mul, mul;
    self: SimilarityBase<N, D, S, R>, right: TranslationBase<N, D, S>, Output = SimilarityBase<N, D, S, R>;
    [val val] => &self * &right;
    [ref val] => self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self.isometry.rotation.transform_vector(&right.vector) * self.scaling();
        SimilarityBase::from_parts(
            TranslationBase::from_vector(&self.isometry.translation.vector + shift),
            self.isometry.rotation.clone(),
            self.scaling())
    };
);


// TranslationBase × SimilarityBase
similarity_binop_impl_all!(
    Mul, mul;
    self: TranslationBase<N, D, S>, right: SimilarityBase<N, D, S, R>, Output = SimilarityBase<N, D, S, R>;
    [val val] => {
        let scaling = right.scaling();
        SimilarityBase::from_isometry(self * right.isometry, scaling)
    };
    [ref val] => {
        let scaling = right.scaling();
        SimilarityBase::from_isometry(self * right.isometry, scaling)
    };
    [val ref] => SimilarityBase::from_isometry(self * &right.isometry, right.scaling());
    [ref ref] => SimilarityBase::from_isometry(self * &right.isometry, right.scaling());
);


macro_rules! similarity_from_composition_impl(
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

macro_rules! similarity_from_composition_impl_all(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);


// RotationBase × SimilarityBase
similarity_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: RotationBase<N, D, SA>, right: SimilarityBase<N, D, SB, RotationBase<N, D, SA>>,
    Output = SimilarityBase<N, D, SB, RotationBase<N, D, SA>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self *  right;
    [ref ref] => SimilarityBase::from_isometry(self * &right.isometry, right.scaling());
);


// RotationBase ÷ SimilarityBase
similarity_from_composition_impl_all!(
    Div, div;
    (D, D), (D, U1) for D: DimName;
    self: RotationBase<N, D, SA>, right: SimilarityBase<N, D, SB, RotationBase<N, D, SA>>,
    Output = SimilarityBase<N, D, SB, RotationBase<N, D, SA>>;
    // FIXME: don't call iverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);


// UnitQuaternion × SimilarityBase
similarity_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, right: SimilarityBase<N, U3, SB, UnitQuaternionBase<N, SA>>,
    Output = SimilarityBase<N, U3, SB, UnitQuaternionBase<N, SA>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self *  right;
    [ref ref] => SimilarityBase::from_isometry(self * &right.isometry, right.scaling());
);


// UnitQuaternion ÷ SimilarityBase
similarity_from_composition_impl_all!(
    Div, div;
    (U4, U1), (U3, U1);
    self: UnitQuaternionBase<N, SA>, right: SimilarityBase<N, U3, SB, UnitQuaternionBase<N, SA>>,
    Output = SimilarityBase<N, U3, SB, UnitQuaternionBase<N, SA>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);
