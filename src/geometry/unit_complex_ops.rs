use std::ops::{Div, DivAssign, Mul, MulAssign};

use alga::general::Real;
use base::allocator::Allocator;
use base::constraint::{DimEq, ShapeConstraint};
use base::dimension::{Dim, U1, U2};
use base::storage::{Storage, StorageMut};
use base::{DefaultAllocator, Matrix, Unit, Vector, Vector2};
use geometry::{Isometry, Point2, Rotation, Similarity, Translation, UnitComplex};

/*
 * This file provides:
 * ===================
 *
 * UnitComplex  × UnitComplex
 * UnitComplex  × Rotation -> UnitComplex
 * Rotation × UnitComplex  -> UnitComplex
 *
 * UnitComplex  ÷ UnitComplex
 * UnitComplex  ÷ Rotation -> UnitComplex
 * Rotation ÷ UnitComplex  -> UnitComplex
 *
 *
 * UnitComplex × Point
 * UnitComplex × Vector
 * UnitComplex × Unit<T>
 *
 * UnitComplex × Isometry<UnitComplex>
 * UnitComplex × Similarity<UnitComplex>
 * UnitComplex × Translation -> Isometry<UnitComplex>
 *
 * NOTE: -UnitComplex is already provided by `Unit<T>`.
 *
 * (Assignment Operators)
 *
 * UnitComplex  ×= UnitComplex
 * UnitComplex  ×= Rotation
 *
 * UnitComplex  ÷= UnitComplex
 * UnitComplex  ÷= Rotation
 *
 * Rotation ×= UnitComplex
 * Rotation ÷= UnitComplex
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
     ($RDim: ident, $CDim: ident) $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: Real $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where DefaultAllocator: Allocator<N, $RDim, $CDim> {
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
     ($RDim: ident, $CDim: ident) $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_val_val; );

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_ref_val; 'a);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_val_ref; 'b);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_ref_ref; 'a, 'b);


    }
);

// UnitComplex × Rotation
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: UnitComplex<N>, rhs: Rotation<N, U2>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs);
);

// UnitComplex ÷ Rotation
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: UnitComplex<N>, rhs: Rotation<N, U2>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs).inverse();
);

// Rotation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: Rotation<N, U2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs;
);

// Rotation ÷ UnitComplex
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: Rotation<N, U2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs.inverse();
);

// UnitComplex × Point
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Point2<N>, Output = Point2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Point2::from_coordinates(self * &rhs.coords);
);

// UnitComplex × Vector
complex_op_impl_all!(
    Mul, mul;
    (U2, U1) for S: Storage<N, U2>;
    self: UnitComplex<N>, rhs: Vector<N, U2, S>, Output = Vector2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let i = self.as_ref().im;
        let r = self.as_ref().re;
        Vector2::new(r * rhs[0] - i * rhs[1], i * rhs[0] + r * rhs[1])
    };
);

// UnitComplex × Unit<Vector>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1) for S: Storage<N, U2>;
    self: UnitComplex<N>, rhs: Unit<Vector<N, U2, S>>, Output = Unit<Vector2<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Unit::new_unchecked(self * rhs.as_ref());
);

// UnitComplex × Isometry<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Isometry<N, U2, UnitComplex<N>>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let shift = self * &rhs.translation.vector;
        Isometry::from_parts(Translation::from_vector(shift), self * &rhs.rotation)
    };
);

// UnitComplex × Similarity<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Similarity<N, U2, UnitComplex<N>>,
    Output = Similarity<N, U2, UnitComplex<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Similarity::from_isometry(self * &rhs.isometry, rhs.scaling());
);

// UnitComplex × Translation
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Translation<N, U2>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(Translation::from_vector(&self *  rhs.vector), self);
    [ref val] => Isometry::from_parts(Translation::from_vector( self *  rhs.vector), self.clone());
    [val ref] => Isometry::from_parts(Translation::from_vector(&self * &rhs.vector), self);
    [ref ref] => Isometry::from_parts(Translation::from_vector( self * &rhs.vector), self.clone());
);

// Translation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: Translation<N, U2>, right: UnitComplex<N>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(self, right);
    [ref val] => Isometry::from_parts(self.clone(), right);
    [val ref] => Isometry::from_parts(self, right.clone());
    [ref ref] => Isometry::from_parts(self.clone(), right.clone());
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

// UnitComplex ×= Rotation
impl<N: Real> MulAssign<Rotation<N, U2>> for UnitComplex<N>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Rotation<N, U2>) {
        *self = &*self * rhs
    }
}

impl<'b, N: Real> MulAssign<&'b Rotation<N, U2>> for UnitComplex<N>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b Rotation<N, U2>) {
        *self = &*self * rhs
    }
}

// UnitComplex ÷= Rotation
impl<N: Real> DivAssign<Rotation<N, U2>> for UnitComplex<N>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Rotation<N, U2>) {
        *self = &*self / rhs
    }
}

impl<'b, N: Real> DivAssign<&'b Rotation<N, U2>> for UnitComplex<N>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b Rotation<N, U2>) {
        *self = &*self / rhs
    }
}

// Rotation ×= UnitComplex
impl<N: Real> MulAssign<UnitComplex<N>> for Rotation<N, U2>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: Real> MulAssign<&'b UnitComplex<N>> for Rotation<N, U2>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

// Rotation ÷= UnitComplex
impl<N: Real> DivAssign<UnitComplex<N>> for Rotation<N, U2>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: Real> DivAssign<&'b UnitComplex<N>> for Rotation<N, U2>
where
    DefaultAllocator: Allocator<N, U2, U2>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}

// Matrix = UnitComplex * Matrix
impl<N: Real> UnitComplex<N> {
    /// Performs the multiplication `rhs = self * rhs` in-place.
    pub fn rotate<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<R2, U2>,
    {
        assert_eq!(
            rhs.nrows(),
            2,
            "Unit complex rotation: the input matrix must have exactly two rows."
        );
        let i = self.as_ref().im;
        let r = self.as_ref().re;

        for j in 0..rhs.ncols() {
            unsafe {
                let a = *rhs.get_unchecked(0, j);
                let b = *rhs.get_unchecked(1, j);

                *rhs.get_unchecked_mut(0, j) = r * a - i * b;
                *rhs.get_unchecked_mut(1, j) = i * a + r * b;
            }
        }
    }

    /// Performs the multiplication `lhs = lhs * self` in-place.
    pub fn rotate_rows<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        lhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<C2, U2>,
    {
        assert_eq!(
            lhs.ncols(),
            2,
            "Unit complex rotation: the input matrix must have exactly two columns."
        );
        let i = self.as_ref().im;
        let r = self.as_ref().re;

        // FIXME: can we optimize that to iterate on one column at a time ?
        for j in 0..lhs.nrows() {
            unsafe {
                let a = *lhs.get_unchecked(j, 0);
                let b = *lhs.get_unchecked(j, 1);

                *lhs.get_unchecked_mut(j, 0) = r * a + i * b;
                *lhs.get_unchecked_mut(j, 1) = -i * a + r * b;
            }
        }
    }
}
