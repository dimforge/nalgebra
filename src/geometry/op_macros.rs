#![macro_use]


// FIXME: merge with `md_impl`.
/// Macro for the implementation of multiplication and division.
macro_rules! md_impl(
    (
    // Operator, operator method, and scalar bounds.
     $Op: ident, $op: ident $(where N: $($ScalarBounds: ident),*)*;
     // Storage dimensions, and dimension bounds.
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) for $($Dims: ident: $DimsBound: ident $(<$($BoundParam: ty),*>)*),+
     // [Optional] Extra allocator bounds.
     $(where $ConstraintType: ty: $ConstraintBound: ident<$($ConstraintBoundParams: ty $( = $EqBound: ty )*),*> )*;
     // Argument identifiers and types + output.
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     // Operator actual implementation.
     $action: expr;
     // Lifetime.
     $($lives: tt),*) => {
        impl<$($lives ,)* N $(, $Dims: $DimsBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where N: Scalar + Zero + One + ClosedAdd + ClosedMul $($(+ $ScalarBounds)*)*,
                  DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2> +
                                    Allocator<N, $R1, $C2>,
                  $( $ConstraintType: $ConstraintBound<$( $ConstraintBoundParams $( = $EqBound )*),*> ),*
                   {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

/// Macro for the implementation of multiplication and division.
/// Implements all the argument reference combinations.
macro_rules! md_impl_all(
    (
     // Operator, operator method, and scalar bounds.
     $Op: ident, $op: ident $(where N: $($ScalarBounds: ident),*)*;
     // Storage dimensions, and dimension bounds.
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) for $($Dims: ident: $DimsBound: ident $(<$($BoundParam: ty),*>)*),+
     // [Optional] Extra allocator bounds.
     $(where $ConstraintType: ty: $ConstraintBound: ident<$($ConstraintBoundParams: ty $( = $EqBound: ty )*),*> )*;
     // Argument identifiers and types + output.
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     // Operators actual implementations.
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

        md_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound<$($ConstraintBoundParams $( = $EqBound )*),*>)*;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Result;
            $action_val_val; );

        md_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound<$($ConstraintBoundParams $( = $EqBound )*),*>)*;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Result;
            $action_ref_val; 'a);

        md_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound<$($ConstraintBoundParams $( = $EqBound )*),*>)*;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Result;
            $action_val_ref; 'b);

        md_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound<$($ConstraintBoundParams $( = $EqBound )*),*>)*;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Result;
            $action_ref_ref; 'a, 'b);
    }
);

/// Macro for the implementation of assignment-multiplication and assignment-division.
macro_rules! md_assign_impl(
    (
     // Operator, operator method, and scalar bounds.
     $Op: ident, $op: ident $(where N: $($ScalarBounds: ident),*)*;
     // Storage dimensions, and dimension bounds.
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) for $($Dims: ident: $DimsBound: ident $(<$($BoundParam: ty),*>)*),+
     // [Optional] Extra allocator bounds.
     $(where $ConstraintType: ty: $ConstraintBound: ident $(<$($ConstraintBoundParams: ty $( = $EqBound: ty )*),*>)* )*;
     // Argument identifiers and types.
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty;
     // Actual implementation and lifetimes.
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N $(, $Dims: $DimsBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where N:  Scalar + Zero + One + ClosedAdd + ClosedMul $($(+ $ScalarBounds)*)*,
                  DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2>,
                  $( $ConstraintType: $ConstraintBound $(<$( $ConstraintBoundParams $( = $EqBound )*),*>)* ),*
        {
            #[inline]
            fn $op(&mut $lhs, $rhs: $Rhs) {
                $action
            }
        }
    }
);

/// Macro for the implementation of assignment-multiplication and assignment-division with and
/// without reference to the right-hand-side.
macro_rules! md_assign_impl_all(
    (
     // Operator, operator method, and scalar bounds.
     $Op: ident, $op: ident $(where N: $($ScalarBounds: ident),*)*;
     // Storage dimensions, and dimension bounds.
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) for $($Dims: ident: $DimsBound: ident $(<$($BoundParam: ty),*>)*),+
     // [Optional] Extra allocator bounds.
     $(where $ConstraintType: ty: $ConstraintBound: ident$(<$($ConstraintBoundParams: ty $( = $EqBound: ty )*),*>)* )*;
     // Argument identifiers and types.
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty;
     // Actual implementation and lifetimes.
     [val] => $action_val: expr;
     [ref] => $action_ref: expr;) => {
        md_assign_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound $(<$($ConstraintBoundParams $( = $EqBound )*),*>)*)*;
            $lhs: $Lhs, $rhs: $Rhs;
            $action_val; );

        md_assign_impl!(
            $Op, $op $(where N: $($ScalarBounds),*)*;
            ($R1, $C1),($R2, $C2) for $($Dims: $DimsBound $(<$($BoundParam),*>)*),+
            $(where $ConstraintType: $ConstraintBound $(<$($ConstraintBoundParams $( = $EqBound )*),*>)*)*;
            $lhs: $Lhs, $rhs: &'b $Rhs;
            $action_ref; 'b);
    }
);

// FIXME: merge with `as_impl`.
/// Macro for the implementation of addition and subtraction.
macro_rules! add_sub_impl(
    ($Op: ident, $op: ident, $bound: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(-> ($RRes: ty))* for $($Dims: ident: $DimsBound: ident $(<$($BoundParam: ty),*>)*),+;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N $(, $Dims: $DimsBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where N: Scalar + $bound,
                  DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2> +
                                    SameShapeAllocator<N, $R1, $C1, $R2, $C2>,
                  ShapeConstraint: SameNumberOfRows<$R1, $R2 $(, Representative = $RRes)*> +
                                   SameNumberOfColumns<$C1, $C2> {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

// FIXME: merge with `md_assign_impl`.
/// Macro for the implementation of assignment-addition and assignment-subtraction.
macro_rules! add_sub_assign_impl(
    ($Op: ident, $op: ident, $bound: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) for $($Dims: ident: $DimsBound: ident),+;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N $(, $Dims: $DimsBound)*> $Op<$Rhs> for $Lhs
            where N: Scalar + $bound,
                  DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2>,
                  ShapeConstraint: SameNumberOfRows<$R1, $R2> + SameNumberOfColumns<$C1, $C2> {
            #[inline]
            fn $op(&mut $lhs, $rhs: $Rhs) {
                $action
            }
        }
    }
);
