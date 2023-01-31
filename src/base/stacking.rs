///! Utilities for stacking matrices horizontally and vertically.
use crate::{
    base::allocator::Allocator,
    constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    Const, DefaultAllocator, Dim, DimAdd, DimSum, Dyn, Matrix, RawStorage, RawStorageMut, Scalar,
};
use num_traits::Zero;

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

mod vstack_impl {
    use super::*;
    #[derive(Clone, Copy)]
    pub struct VStackShapeInit;

    #[derive(Clone, Copy)]
    pub struct VStackShape<R, C> {
        r: R,
        c: C,
    }

    impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Visitor<&Matrix<T, R, C, S>>
        for VStackShapeInit
    {
        type Output = VStackShape<R, C>;
        fn visit(self, x: &Matrix<T, R, C, S>) -> Self::Output {
            let (r, c) = x.shape_generic();
            VStackShape { r, c }
        }
    }
    impl<T: Scalar, R1: Dim + DimAdd<R2>, C1: Dim, R2: Dim, C2: Dim, S2: RawStorage<T, R2, C2>>
        Visitor<&Matrix<T, R2, C2, S2>> for VStackShape<R1, C1>
    where
        ShapeConstraint: SameNumberOfColumns<C1, C2>,
    {
        type Output =
            VStackShape<DimSum<R1, R2>, <ShapeConstraint as DimEq<C1, C2>>::Representative>;
        fn visit(self, x: &Matrix<T, R2, C2, S2>) -> Self::Output {
            let (r, c) = x.shape_generic();
            VStackShape {
                r: self.r.add(r),
                c: <ShapeConstraint as DimEq<C1, C2>>::Representative::from_usize(c.value()),
            }
        }
    }

    pub struct VStack<T, R, C, S, R2> {
        out: Matrix<T, R, C, S>,
        current_row: R2,
    }

    impl<
            T: Scalar,
            R1: Dim + DimAdd<Const<R2>>,
            C1: Dim,
            S1: RawStorageMut<T, R1, C1>,
            C2: Dim,
            S2: RawStorage<T, Const<R2>, C2>,
            R3: Dim + DimAdd<Const<R2>>,
            const R2: usize,
        > Visitor<&Matrix<T, Const<R2>, C2, S2>> for VStack<T, R1, C1, S1, R3>
    where
        ShapeConstraint: SameNumberOfColumns<C1, C2>,
    {
        type Output = VStack<T, R1, C1, S1, DimSum<R3, Const<R2>>>;
        fn visit(self, x: &Matrix<T, Const<R2>, C2, S2>) -> Self::Output {
            let (r2, _) = x.shape_generic();
            let VStack {
                mut out,
                current_row,
            } = self;
            out.fixed_rows_mut::<{ R2 }>(current_row.value())
                .copy_from::<Const<R2>, C2, S2>(x);
            let current_row = current_row.add(r2);
            VStack { out, current_row }
        }
    }
    impl<
            T: Scalar,
            R1: Dim + DimAdd<Dyn>,
            C1: Dim,
            S1: RawStorageMut<T, R1, C1>,
            C2: Dim,
            S2: RawStorage<T, Dyn, C2>,
            R3: Dim + DimAdd<Dyn>,
        > Visitor<&Matrix<T, Dyn, C2, S2>> for VStack<T, R1, C1, S1, R3>
    where
        ShapeConstraint: SameNumberOfColumns<C1, C2>,
    {
        type Output = VStack<T, R1, C1, S1, DimSum<R3, Dyn>>;
        fn visit(self, x: &Matrix<T, Dyn, C2, S2>) -> Self::Output {
            let (r2, _) = x.shape_generic();
            let VStack {
                mut out,
                current_row,
            } = self;
            out.rows_mut(current_row.value(), r2.value())
                .copy_from::<Dyn, C2, S2>(x);
            let current_row = current_row.add(r2);
            VStack { out, current_row }
        }
    }

    /// Stack a tuple of references to matrices with equal column counts vertically, yielding a
    /// matrix with every row of the input matrices.
    #[inline]
    pub fn vstack<
        T: Scalar + Zero,
        R: Dim,
        C: Dim,
        X: Copy
            + VisitTuple<VStackShapeInit, Output = VStackShape<R, C>>
            + VisitTuple<
                VStack<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer, Const<0>>,
                Output = VStack<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer, R>,
            >,
    >(
        x: X,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        let vstack_shape = VStackShapeInit;
        let vstack_shape = <X as VisitTuple<_>>::visit(vstack_shape, x);
        let vstack_visitor = VStack {
            out: Matrix::zeros_generic(vstack_shape.r, vstack_shape.c),
            current_row: Const,
        };
        let vstack_visitor = <X as VisitTuple<_>>::visit(vstack_visitor, x);
        vstack_visitor.out
    }
}
pub use vstack_impl::vstack;

mod hstack_impl {
    use super::*;
    #[derive(Clone, Copy)]
    pub struct HStackShapeInit;

    #[derive(Clone, Copy)]
    pub struct HStackShape<R, C> {
        r: R,
        c: C,
    }

    impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Visitor<&Matrix<T, R, C, S>>
        for HStackShapeInit
    {
        type Output = HStackShape<R, C>;
        fn visit(self, x: &Matrix<T, R, C, S>) -> Self::Output {
            let (r, c) = x.shape_generic();
            HStackShape { r, c }
        }
    }
    impl<T: Scalar, R1: Dim, C1: Dim + DimAdd<C2>, R2: Dim, C2: Dim, S2: RawStorage<T, R2, C2>>
        Visitor<&Matrix<T, R2, C2, S2>> for HStackShape<R1, C1>
    where
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        type Output =
            HStackShape<<ShapeConstraint as DimEq<R1, R2>>::Representative, DimSum<C1, C2>>;
        fn visit(self, x: &Matrix<T, R2, C2, S2>) -> Self::Output {
            let (r, c) = x.shape_generic();
            HStackShape {
                r: <ShapeConstraint as DimEq<R1, R2>>::Representative::from_usize(r.value()),
                c: self.c.add(c),
            }
        }
    }

    pub struct HStack<T, R, C, S, C2> {
        out: Matrix<T, R, C, S>,
        current_col: C2,
    }

    impl<
            T: Scalar,
            R1: Dim,
            C1: Dim + DimAdd<Const<C2>>,
            S1: RawStorageMut<T, R1, C1>,
            R2: Dim,
            S2: RawStorage<T, R2, Const<C2>>,
            C3: Dim + DimAdd<Const<C2>>,
            const C2: usize,
        > Visitor<&Matrix<T, R2, Const<C2>, S2>> for HStack<T, R1, C1, S1, C3>
    where
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        type Output = HStack<T, R1, C1, S1, DimSum<C3, Const<C2>>>;
        fn visit(self, x: &Matrix<T, R2, Const<C2>, S2>) -> Self::Output {
            let (_, c2) = x.shape_generic();
            let HStack {
                mut out,
                current_col,
            } = self;
            out.fixed_columns_mut::<{ C2 }>(current_col.value())
                .copy_from::<R2, Const<C2>, S2>(x);
            let current_col = current_col.add(c2);
            HStack { out, current_col }
        }
    }
    impl<
            T: Scalar,
            R1: Dim,
            C1: Dim + DimAdd<Dyn>,
            S1: RawStorageMut<T, R1, C1>,
            R2: Dim,
            S2: RawStorage<T, R2, Dyn>,
            C3: Dim + DimAdd<Dyn>,
        > Visitor<&Matrix<T, R2, Dyn, S2>> for HStack<T, R1, C1, S1, C3>
    where
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        type Output = HStack<T, R1, C1, S1, DimSum<C3, Dyn>>;
        fn visit(self, x: &Matrix<T, R2, Dyn, S2>) -> Self::Output {
            let (_, c2) = x.shape_generic();
            let HStack {
                mut out,
                current_col,
            } = self;
            out.columns_mut(current_col.value(), c2.value())
                .copy_from::<R2, Dyn, S2>(x);
            let current_col = current_col.add(c2);
            HStack { out, current_col }
        }
    }

    /// Stack a tuple of references to matrices with equal row counts horizontally, yielding a
    /// matrix with every column of the input matrices.
    #[inline]
    pub fn hstack<
        T: Scalar + Zero,
        R: Dim,
        C: Dim,
        X: Copy
            + VisitTuple<HStackShapeInit, Output = HStackShape<R, C>>
            + VisitTuple<
                HStack<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer, Const<0>>,
                Output = HStack<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer, C>,
            >,
    >(
        x: X,
    ) -> Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
    where
        DefaultAllocator: Allocator<T, R, C>,
    {
        let hstack_shape = HStackShapeInit;
        let hstack_shape = <X as VisitTuple<_>>::visit(hstack_shape, x);
        let hstack_visitor = HStack {
            out: Matrix::zeros_generic(hstack_shape.r, hstack_shape.c),
            current_col: Const,
        };
        let hstack_visitor = <X as VisitTuple<_>>::visit(hstack_visitor, x);
        hstack_visitor.out
    }
}
pub use hstack_impl::hstack;
