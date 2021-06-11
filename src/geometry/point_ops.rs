use num::{One, Zero};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use simba::scalar::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub};

use crate::base::constraint::{
    AreMultipliable, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::base::dimension::{Dim, U1};
use crate::base::storage::Storage;
use crate::base::{Const, Matrix, SVector, Scalar, Vector};

use crate::geometry::Point;

/*
 *
 * Indexing.
 *
 */
impl<T: Scalar, const D: usize> Index<usize> for Point<T, D> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}

impl<T: Scalar, const D: usize> IndexMut<usize> for Point<T, D> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.coords[i]
    }
}

/*
 *
 * Neg.
 *
 */
impl<T: Scalar + ClosedNeg, const D: usize> Neg for Point<T, D> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-self.coords)
    }
}

impl<'a, T: Scalar + ClosedNeg, const D: usize> Neg for &'a Point<T, D> {
    type Output = Point<T, D>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-&self.coords)
    }
}

/*
 *
 * Subtraction & Addition.
 *
 */

// Point - Point
add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Point<T, D>, right: &'b Point<T, D>, Output = SVector<T, D>;
    &self.coords - &right.coords; 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: &'a Point<T, D>, right: Point<T, D>, Output = SVector<T, D>;
    &self.coords - right.coords; 'a);

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Point<T, D>, right: &'b Point<T, D>, Output = SVector<T, D>;
    self.coords - &right.coords; 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D>, U1), (Const<D>, U1) -> (Const<D>, U1)
    const D; for; where;
    self: Point<T, D>, right: Point<T, D>, Output = SVector<T, D>;
    self.coords - right.coords; );

// Point - Vector
add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: &'a Point<T, D1>, right: &'b Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(&self.coords - right); 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: &'a Point<T, D1>, right: Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(&self.coords - &right); 'a); // TODO: should not be a ref to `right`.

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: Point<T, D1>, right: &'b Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(self.coords - right); 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: Point<T, D1>, right: Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(self.coords - right); );

// Point + Vector
add_sub_impl!(Add, add, ClosedAdd;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: &'a Point<T, D1>, right: &'b Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(&self.coords + right); 'a, 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: &'a Point<T, D1>, right: Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(&self.coords + &right); 'a); // TODO: should not be a ref to `right`.

add_sub_impl!(Add, add, ClosedAdd;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: Point<T, D1>, right: &'b Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(self.coords + right); 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (Const<D1>, U1), (D2, U1) -> (Const<D1>, U1)
    const D1;
    for D2, SB;
    where D2: Dim, SB: Storage<T, D2>;
    self: Point<T, D1>, right: Vector<T, D2, SB>, Output = Point<T, D1>;
    Self::Output::from(self.coords + right); );

// TODO: replace by the shared macro: add_sub_assign_impl?
macro_rules! op_assign_impl(
    ($($TraitAssign: ident, $method_assign: ident, $bound: ident);* $(;)*) => {$(
        impl<'b, T, D2: Dim, SB, const D1: usize> $TraitAssign<&'b Vector<T, D2, SB>> for Point<T, D1>
            where T: Scalar + $bound,
                  SB: Storage<T, D2>,
                  ShapeConstraint: SameNumberOfRows<Const<D1>, D2> {

            #[inline]
            fn $method_assign(&mut self, right: &'b Vector<T, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }

        impl<T, D2: Dim, SB, const D1: usize> $TraitAssign<Vector<T, D2, SB>> for Point<T, D1>
            where T: Scalar + $bound,
                  SB: Storage<T, D2>,
                  ShapeConstraint: SameNumberOfRows<Const<D1>, D2> {

            #[inline]
            fn $method_assign(&mut self, right: Vector<T, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }
    )*}
);

op_assign_impl!(
    AddAssign, add_assign, ClosedAdd;
    SubAssign, sub_assign, ClosedSub;
);

/*
 *
 * Matrix × Point
 *
 */
md_impl_all!(
    Mul, mul;
    (Const<R1>, Const<C1>), (Const<D2>, U1)
    const D2, R1, C1;
    for SA;
    where SA: Storage<T, Const<R1>, Const<C1>>,
          ShapeConstraint: AreMultipliable<Const<R1>, Const<C1>, Const<D2>, U1>;
    self: Matrix<T, Const<R1>, Const<C1>, SA>, right: Point<T, D2>, Output = Point<T, R1>;
    [val val] => Point::from(self * right.coords);
    [ref val] => Point::from(self * right.coords);
    [val ref] => Point::from(self * &right.coords);
    [ref ref] => Point::from(self * &right.coords);
);

/*
 *
 * Point ×/÷ Scalar
 *
 */
macro_rules! componentwise_scalarop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident) => {
        impl<T: Scalar + $bound, const D: usize> $Trait<T> for Point<T, D>
        {
            type Output = Point<T, D>;

            #[inline]
            fn $method(self, right: T) -> Self::Output {
                Point::from(self.coords.$method(right))
            }
        }

        impl<'a, T: Scalar + $bound, const D: usize> $Trait<T> for &'a Point<T, D>
        {
            type Output = Point<T, D>;

            #[inline]
            fn $method(self, right: T) -> Self::Output {
                Point::from((&self.coords).$method(right))
            }
        }

        impl<T: Scalar + $bound, const D: usize> $TraitAssign<T> for Point<T, D>
            /* where DefaultAllocator: Allocator<T, D> */
        {
            #[inline]
            fn $method_assign(&mut self, right: T) {
                self.coords.$method_assign(right)
            }
        }
    }
);

componentwise_scalarop_impl!(Mul, mul, ClosedMul; MulAssign, mul_assign);
componentwise_scalarop_impl!(Div, div, ClosedDiv; DivAssign, div_assign);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl<const D: usize> Mul<Point<$T, D>> for $T
        {
            type Output = Point<$T, D>;

            #[inline]
            fn mul(self, right: Point<$T, D>) -> Self::Output {
                Point::from(self * right.coords)
            }
        }

        impl<'b, const D: usize> Mul<&'b Point<$T, D>> for $T
        {
            type Output = Point<$T, D>;

            #[inline]
            fn mul(self, right: &'b Point<$T, D>) -> Self::Output {
                Point::from(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
