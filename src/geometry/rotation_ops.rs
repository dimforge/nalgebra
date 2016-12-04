/*
 *
 * This provides the following operator overladings:
 *
 * Index<(usize, usize)>
 *
 * RotationBase × RotationBase
 * RotationBase ÷ RotationBase
 * RotationBase × Matrix
 * Matrix       × RotationBase
 * Matrix       ÷ RotationBase
 * RotationBase × PointBase
 *
 *
 * RotationBase ×= RotationBase
 * Matrix       ×= RotationBase
 */


use std::ops::{Mul, MulAssign, Div, DivAssign, Index};
use num::Zero;

use alga::general::{ClosedMul, ClosedAdd};

use core::{Scalar, Matrix, MatrixMul};
use core::dimension::{Dim, DimName, U1};
use core::constraint::{ShapeConstraint, AreMultipliable};
use core::storage::{OwnedStorage, Storage};
use core::allocator::{OwnedAllocator, Allocator};

use geometry::{PointBase, PointMul, RotationBase, OwnedRotation};

impl<N: Scalar, D: DimName, S: Storage<N, D, D>> Index<(usize, usize)> for RotationBase<N, D, S> {
    type Output = N;

    #[inline]
    fn index(&self, row_col: (usize, usize)) -> &N {
        self.matrix().index(row_col)
    }
}

// RotationBase × RotationBase
md_impl_all!(
    Mul, mul;
    (D, D), (D, D) for D: DimName;
    self: RotationBase<N, D, SA>, right: RotationBase<N, D, SB>, Output = OwnedRotation<N, D, SA::Alloc>;
    [val val] => RotationBase::from_matrix_unchecked(self.unwrap() * right.unwrap());
    [ref val] => RotationBase::from_matrix_unchecked(self.matrix() * right.unwrap());
    [val ref] => RotationBase::from_matrix_unchecked(self.unwrap() * right.matrix());
    [ref ref] => RotationBase::from_matrix_unchecked(self.matrix() * right.matrix());
);

// RotationBase ÷ RotationBase
// FIXME: instead of calling inverse explicitely, could we just add a `mul_tr` or `mul_inv` method?
md_impl_all!(
    Div, div;
    (D, D), (D, D) for D: DimName;
    self: RotationBase<N, D, SA>, right: RotationBase<N, D, SB>, Output = OwnedRotation<N, D, SA::Alloc>;
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// RotationBase × Matrix
md_impl_all!(
    Mul, mul;
    (D1, D1), (R2, C2) for D1: DimName, R2: Dim, C2: Dim
    where SA::Alloc: Allocator<N, D1, C2>
    where ShapeConstraint: AreMultipliable<D1, D1, R2, C2>;
    self: RotationBase<N, D1, SA>, right: Matrix<N, R2, C2, SB>, Output = MatrixMul<N, D1, D1, C2 , SA>;
    [val val] => self.unwrap() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.unwrap() * right;
    [ref ref] => self.matrix() * right;
);

// Matrix × RotationBase
md_impl_all!(
    Mul, mul;
    (R1, C1), (D2, D2) for R1: Dim, C1: Dim, D2: DimName
    where SA::Alloc: Allocator<N, R1, D2>
    where ShapeConstraint: AreMultipliable<R1, C1, D2, D2>;
    self: Matrix<N, R1, C1, SA>, right: RotationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
    [val val] => self * right.unwrap();
    [ref val] => self * right.unwrap();
    [val ref] => self * right.matrix();
    [ref ref] => self * right.matrix();
);

// Matrix ÷ RotationBase
md_impl_all!(
    Div, div;
    (R1, C1), (D2, D2) for R1: Dim, C1: Dim, D2: DimName
    where SA::Alloc: Allocator<N, R1, D2>
    where ShapeConstraint: AreMultipliable<R1, C1, D2, D2>;
    self: Matrix<N, R1, C1, SA>, right: RotationBase<N, D2, SB>, Output = MatrixMul<N, R1, C1, D2, SA>;
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);


// RotationBase × PointBase
// FIXME: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
md_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName
    where SA::Alloc: Allocator<N, D, U1>
    where ShapeConstraint: AreMultipliable<D, D, D, U1>;
    self: RotationBase<N, D, SA>, right: PointBase<N, D, SB>, Output = PointMul<N, D, D, SA>;
    [val val] => self.unwrap() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.unwrap() * right;
    [ref ref] => self.matrix() * right;
);


// RotationBase *= RotationBase
// FIXME: try not to call `inverse()` explicitly.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (D, D), (D, D) for D: DimName;
    self: RotationBase<N, D, SA>, right: RotationBase<N, D, SB>;
    [val] => unsafe { self.matrix_mut().mul_assign(right.unwrap()) };
    [ref] => unsafe { self.matrix_mut().mul_assign(right.matrix()) };
);


md_assign_impl_all!(
    DivAssign, div_assign;
    (D, D), (D, D) for D: DimName;
    self: RotationBase<N, D, SA>, right: RotationBase<N, D, SB>;
    [val] => unsafe { self.matrix_mut().mul_assign(right.inverse().unwrap()) };
    [ref] => unsafe { self.matrix_mut().mul_assign(right.inverse().matrix()) };
);

// Matrix *= RotationBase
// FIXME: try not to call `inverse()` explicitly.
// FIXME: this shares the same limitations as for the current impl. of MulAssign for matrices.
// (In particular the number of matrix column must be equal to the number of rotation columns,
// i.e., equal to the rotation dimension.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (R1, C1), (C1, C1) for R1: DimName, C1: DimName;
    self: Matrix<N, R1, C1, SA>, right: RotationBase<N, C1, SB>;
    [val] => self.mul_assign(right.unwrap());
    [ref] => self.mul_assign(right.matrix());
);


md_assign_impl_all!(
    DivAssign, div_assign;
    (R1, C1), (C1, C1) for R1: DimName, C1: DimName;
    self: Matrix<N, R1, C1, SA>, right: RotationBase<N, C1, SB>;
    [val] => self.mul_assign(right.inverse().unwrap());
    [ref] => self.mul_assign(right.inverse().matrix());
);
