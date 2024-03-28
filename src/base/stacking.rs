///! Utilities for stacking matrices horizontally and vertically.
use crate::{
    base::allocator::Allocator,
    constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    Const, DefaultAllocator, Dim, DimAdd, DimDiff, DimSub, DimSum, Matrix, RawStorage,
    RawStorageMut, Scalar, ViewStorageMut,
};
use num_traits::Zero;
use std::marker::PhantomData;

/// A visitor for each folding over each element of a tuple.
pub trait Visitor<A> {
    /// The output type of this step.
    type Output;
    /// Visits an element of a tuple.
    fn visit(self, x: A) -> Self::Output;
}

/// The driver for visiting each element of a tuple.
pub trait VisitTuple<F> {
    /// The output type of the fold.
    type Output;
    /// Visits each element of a tuple.
    fn visit(visitor: F, x: Self) -> Self::Output;
}

macro_rules! impl_visit_tuple {
    ($($is:ident),*) => {
        impl_visit_tuple!(__GENERATE_TAILS, [$($is),*]);
    };
    (__GENERATE_TAILS, [$i:ident]) => {
        impl_visit_tuple!(__GENERATE_CLAUSE, [$i]);
    };
    (__GENERATE_TAILS, [$i:ident, $($is:ident),*]) => {
        impl_visit_tuple!(__GENERATE_CLAUSE, [$i, $($is),*]);
        impl_visit_tuple!(__GENERATE_TAILS, [$($is),*]);
    };
    (__GENERATE_CLAUSE, [$i:ident]) => {
        impl<$i, Func: Visitor<$i>> VisitTuple<Func> for ($i,) {
            type Output = <Func as Visitor<$i>>::Output;
            #[allow(non_snake_case)]
            #[inline(always)]
            fn visit(visitor: Func, ($i,): Self) -> Self::Output {
                visitor.visit($i)
            }
        }
    };
    (__GENERATE_CLAUSE, [$i:ident, $($is:ident),*]) => {
        impl<$i, $($is,)* Func: Visitor<$i>> VisitTuple<Func> for ($i, $($is),*)
        where ($($is,)*): VisitTuple<<Func as Visitor<$i>>::Output>
        {
            type Output = <($($is,)*) as VisitTuple<<Func as Visitor<$i>>::Output>>::Output;
            #[allow(non_snake_case)]
            #[inline(always)]
            fn visit(visitor: Func, ($i, $($is),*): Self) -> Self::Output {
                VisitTuple::visit(visitor.visit($i), ($($is,)*))
            }
        }
    };
}

impl_visit_tuple!(H, G, F, E, D, C, B, A);

/// Source of data that can populate a block of a matrix.
pub trait Block {
    /// The scalar type of the data.
    type T: Scalar;
    /// The number of rows of the block.
    type Rows: Dim;
    /// The number of columns of the block.
    type Cols: Dim;
    /// The shape of the block.
    fn shape(self) -> (Self::Rows, Self::Cols);
}

/// Source of data that can populate a block of a matrix.
/// Separate from Block because it's useful to specify the bound on the storage independently of
/// the other bounds.
pub trait BlockPopulate<S>: Block
where
    S: RawStorageMut<Self::T, Self::Rows, Self::Cols>,
{
    /// Populate a matrix from this block's data.
    fn populate(self, m: &mut Matrix<Self::T, Self::Rows, Self::Cols, S>);
}

impl<'a, T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Block for &'a Matrix<T, R, C, S> {
    type T = T;
    type Rows = R;
    type Cols = C;
    #[inline(always)]
    fn shape(self) -> (Self::Rows, Self::Cols) {
        self.shape_generic()
    }
}
impl<'a, T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>, S2: RawStorageMut<T, R, C>>
    BlockPopulate<S2> for &'a Matrix<T, R, C, S>
{
    #[inline(always)]
    fn populate(self, m: &mut Matrix<T, Self::Rows, Self::Cols, S2>) {
        m.copy_from(self);
    }
}

#[inline]
fn build<B: Copy, T: Scalar + Zero, R: Dim, C: Dim, S>(x: B) -> Matrix<T, R, C, S>
where
    S: RawStorageMut<T, R, C>,
    DefaultAllocator: Allocator<T, R, C, Buffer = S>,
    B: Block<T = T, Rows = R, Cols = C> + BlockPopulate<S>,
{
    let (r, c) = x.shape();
    let mut out = Matrix::zeros_generic(r, c);
    x.populate(&mut out);
    out
}

mod vstack_impl {
    use super::*;
    #[derive(Clone, Copy)]
    pub struct VStackShapeInit;

    #[derive(Clone, Copy)]
    pub struct VStackShape<T, R, C> {
        t: PhantomData<T>,
        r: R,
        c: C,
    }

    impl<B: Block> Visitor<B> for VStackShapeInit {
        type Output = VStackShape<B::T, B::Rows, B::Cols>;
        fn visit(self, x: B) -> Self::Output {
            let (r, c) = x.shape();
            VStackShape {
                t: PhantomData,
                r,
                c,
            }
        }
    }
    impl<B: Block, R1: Dim + DimAdd<B::Rows>, C1: Dim> Visitor<B> for VStackShape<B::T, R1, C1>
    where
        DimSum<R1, B::Rows>: DimSub<R1> + DimSub<B::Rows>,
        ShapeConstraint: SameNumberOfColumns<C1, B::Cols>
            + SameNumberOfRows<DimDiff<DimSum<R1, B::Rows>, R1>, B::Rows>
            + SameNumberOfRows<DimDiff<DimSum<R1, B::Rows>, B::Rows>, R1>,
    {
        type Output = VStackShape<
            B::T,
            DimSum<R1, B::Rows>,
            <ShapeConstraint as DimEq<C1, B::Cols>>::Representative,
        >;
        fn visit(self, x: B) -> Self::Output {
            let (r, c) = x.shape();
            VStackShape {
                t: self.t,
                r: self.r.add(r),
                c: <ShapeConstraint as DimEq<C1, B::Cols>>::Representative::from_usize(c.value()),
            }
        }
    }

    /// Specify vertical stacking as a Block.
    #[derive(Copy, Clone)]
    pub struct VStackLazy<X>(pub X);

    impl<T: Scalar, R: Dim, C: Dim, X> Block for VStackLazy<X>
    where
        X: Copy + VisitTuple<VStackShapeInit, Output = VStackShape<T, R, C>>,
    {
        type T = T;
        type Rows = R;
        type Cols = C;
        fn shape(self) -> (Self::Rows, Self::Cols) {
            let shape = <X as VisitTuple<_>>::visit(VStackShapeInit, self.0);
            (shape.r, shape.c)
        }
    }
    impl<T: Scalar, R: Dim, C: Dim, S: RawStorageMut<T, R, C>, X> BlockPopulate<S> for VStackLazy<X>
    where
        X: Copy
            + VisitTuple<VStackShapeInit, Output = VStackShape<T, R, C>>
            + for<'a> VisitTuple<VStack<'a, T, R, C, S, Const<0>>, Output = VStack<'a, T, R, C, S, R>>,
    {
        fn populate(self, m: &mut Matrix<T, R, C, S>) {
            let vstack_visitor = VStack {
                out: m,
                current_row: Const,
            };
            let _ = <X as VisitTuple<_>>::visit(vstack_visitor, self.0);
        }
    }

    pub struct VStack<'a, T, R, C, S, R2> {
        out: &'a mut Matrix<T, R, C, S>,
        current_row: R2,
    }

    impl<
            'a,
            B: Copy
                + Block
                + for<'b> BlockPopulate<
                    ViewStorageMut<
                        'b,
                        <B as Block>::T,
                        <B as Block>::Rows,
                        <B as Block>::Cols,
                        <S1 as RawStorage<<B as Block>::T, R1, C1>>::RStride,
                        <S1 as RawStorage<<B as Block>::T, R1, C1>>::CStride,
                    >,
                >,
            R1: Dim + DimAdd<B::Rows>,
            C1: Dim,
            S1: RawStorageMut<B::T, R1, C1>,
            R3: Dim + DimAdd<B::Rows>,
        > Visitor<B> for VStack<'a, B::T, R1, C1, S1, R3>
    where
        B::T: Scalar,
        ShapeConstraint: SameNumberOfColumns<C1, B::Cols>,
    {
        type Output = VStack<'a, B::T, R1, C1, S1, DimSum<R3, B::Rows>>;
        fn visit(self, x: B) -> Self::Output {
            let (r2, c2) = x.shape();
            let VStack { out, current_row } = self;
            x.populate(&mut out.generic_view_mut((current_row.value(), 0), (r2, c2)));
            let current_row = current_row.add(r2);
            VStack { out, current_row }
        }
    }

    /// Stack a tuple of references to matrices with equal column counts vertically, yielding a
    /// matrix with every row of the input matrices.
    #[inline]
    pub fn vstack<T: Scalar + Zero, R: Dim, C: Dim, X: Copy>(
        x: X,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        DefaultAllocator: Allocator<T, R, C>,
        VStackLazy<X>: Block<T = T, Rows = R, Cols = C>
            + BlockPopulate<<DefaultAllocator as Allocator<T, R, C>>::Buffer>,
    {
        build(VStackLazy(x))
    }
}
pub use vstack_impl::{vstack, VStackLazy};

mod hstack_impl {
    use super::*;
    #[derive(Clone, Copy)]
    pub struct HStackShapeInit;

    #[derive(Clone, Copy)]
    pub struct HStackShape<T, R, C> {
        t: PhantomData<T>,
        r: R,
        c: C,
    }

    impl<B: Block> Visitor<B> for HStackShapeInit {
        type Output = HStackShape<B::T, B::Rows, B::Cols>;
        fn visit(self, x: B) -> Self::Output {
            let (r, c) = x.shape();
            HStackShape {
                t: PhantomData,
                r,
                c,
            }
        }
    }
    impl<B: Block, R1: Dim, C1: Dim + DimAdd<B::Cols>> Visitor<B> for HStackShape<B::T, R1, C1>
    where
        DimSum<C1, B::Cols>: DimSub<C1> + DimSub<B::Cols>,
        ShapeConstraint: SameNumberOfRows<R1, B::Rows>
            + SameNumberOfColumns<DimDiff<DimSum<C1, B::Cols>, C1>, B::Cols>
            + SameNumberOfColumns<DimDiff<DimSum<C1, B::Cols>, B::Cols>, C1>,
    {
        type Output = HStackShape<
            B::T,
            <ShapeConstraint as DimEq<R1, B::Rows>>::Representative,
            DimSum<C1, B::Cols>,
        >;
        fn visit(self, x: B) -> Self::Output {
            let (r, c) = x.shape();
            HStackShape {
                t: self.t,
                r: <ShapeConstraint as DimEq<R1, B::Rows>>::Representative::from_usize(r.value()),
                c: self.c.add(c),
            }
        }
    }

    /// Specify horizontal stacking as a Block.
    #[derive(Copy, Clone)]
    pub struct HStackLazy<X>(pub X);

    impl<T: Scalar, R: Dim, C: Dim, X> Block for HStackLazy<X>
    where
        X: Copy + VisitTuple<HStackShapeInit, Output = HStackShape<T, R, C>>,
    {
        type T = T;
        type Rows = R;
        type Cols = C;
        fn shape(self) -> (Self::Rows, Self::Cols) {
            let shape = <X as VisitTuple<_>>::visit(HStackShapeInit, self.0);
            (shape.r, shape.c)
        }
    }
    impl<T: Scalar, R: Dim, C: Dim, S: RawStorageMut<T, R, C>, X> BlockPopulate<S> for HStackLazy<X>
    where
        X: Copy
            + VisitTuple<HStackShapeInit, Output = HStackShape<T, R, C>>
            + for<'a> VisitTuple<HStack<'a, T, R, C, S, Const<0>>, Output = HStack<'a, T, R, C, S, C>>,
    {
        fn populate(self, m: &mut Matrix<T, R, C, S>) {
            let hstack_visitor = HStack {
                out: m,
                current_col: Const,
            };
            let _ = <X as VisitTuple<_>>::visit(hstack_visitor, self.0);
        }
    }

    pub struct HStack<'a, T, R, C, S, C2> {
        out: &'a mut Matrix<T, R, C, S>,
        current_col: C2,
    }

    impl<
            'a,
            B: Copy
                + Block
                + for<'b> BlockPopulate<
                    ViewStorageMut<
                        'b,
                        <B as Block>::T,
                        <B as Block>::Rows,
                        <B as Block>::Cols,
                        <S1 as RawStorage<<B as Block>::T, R1, C1>>::RStride,
                        <S1 as RawStorage<<B as Block>::T, R1, C1>>::CStride,
                    >,
                >,
            R1: Dim,
            C1: Dim,
            S1: RawStorageMut<B::T, R1, C1>,
            C3: Dim + DimAdd<B::Cols>,
        > Visitor<B> for HStack<'a, B::T, R1, C1, S1, C3>
    where
        B::T: Scalar,
        ShapeConstraint: SameNumberOfRows<R1, B::Rows>,
    {
        type Output = HStack<'a, B::T, R1, C1, S1, DimSum<C3, B::Cols>>;
        fn visit(self, x: B) -> Self::Output {
            let (r2, c2) = x.shape();
            let HStack { out, current_col } = self;
            x.populate(&mut out.generic_view_mut((0, current_col.value()), (r2, c2)));
            let current_col = current_col.add(c2);
            HStack { out, current_col }
        }
    }

    /// Stack a tuple of references to matrices with equal row counts horizontally, yielding a
    /// matrix with every column of the input matrices.
    #[inline]
    pub fn hstack<T: Scalar + Zero, R: Dim, C: Dim, X: Copy>(
        x: X,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        DefaultAllocator: Allocator<T, R, C>,
        HStackLazy<X>: Block<T = T, Rows = R, Cols = C>
            + BlockPopulate<<DefaultAllocator as Allocator<T, R, C>>::Buffer>,
    {
        build(HStackLazy(x))
    }
}
pub use hstack_impl::{hstack, HStackLazy};
