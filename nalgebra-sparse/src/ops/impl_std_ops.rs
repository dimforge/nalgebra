//! Impl blocks for standard ops traits like Add, Mul, Sub, etc.

use super::serial::{scalar::*, spadd::*, spmm::*, spsub::*};
use crate::cs::{
    CompressedColumnStorage, CompressedRowStorage, Compression, CsMatrix, CscMatrix, CsrMatrix,
};
use nalgebra::{Dim, Matrix, RawStorage, RawStorageMut, Scalar};
use num_traits::Zero;
use std::{
    borrow::Borrow,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

// Addition

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Add<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Add<T2>>::Output>;

    fn add(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spadd_csr_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Add<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Into<<T2 as Add<T1>>::Output>,
    T2: Scalar + Into<<T2 as Add<T1>>::Output> + Add<T1>,
    <T2 as Add<T1>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T2 as Add<T1>>::Output>;

    fn add(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spadd_csc_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Add<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CscMatrix<<T1 as Add<T2>>::Output>;

    fn add(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spadd_csc_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Add<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Add<T2>>::Output>;

    fn add(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spadd_csr_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Add<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedColumnStorage>
where
    T2: Scalar + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = Matrix<T2, R, C, S>;

    fn add(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spadd_csc_dense(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Add<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedRowStorage>
where
    T2: Scalar + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = Matrix<T2, R, C, S>;

    fn add(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spadd_csr_dense(self, rhs).unwrap()
    }
}

// Subtraction

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Sub<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T1: Scalar + Into<<T1 as Sub<T2>>::Output> + Sub<T2> + Zero,
    T2: Scalar + Into<<T1 as Sub<T2>>::Output>,
    <T1 as Sub<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Sub<T2>>::Output>;

    fn sub(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spsub_csr_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Sub<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Into<<T1 as Sub<T2>>::Output> + Sub<T2> + Zero,
    T2: Scalar + Into<<T1 as Sub<T2>>::Output>,
    <T1 as Sub<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CscMatrix<<T1 as Sub<T2>>::Output>;

    fn sub(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spsub_csc_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Sub<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Into<<T1 as Sub<T2>>::Output> + Sub<T2> + Zero,
    T2: Scalar + Into<<T1 as Sub<T2>>::Output>,
    <T1 as Sub<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CscMatrix<<T1 as Sub<T2>>::Output>;

    fn sub(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spsub_csc_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Sub<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T1: Scalar + Into<<T1 as Sub<T2>>::Output> + Sub<T2> + Zero,
    T2: Scalar + Into<<T1 as Sub<T2>>::Output>,
    <T1 as Sub<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Sub<T2>>::Output>;

    fn sub(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spsub_csr_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Sub<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedColumnStorage>
where
    T2: Scalar + Neg<Output = T2> + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = Matrix<T2, R, C, S>;

    fn sub(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spsub_csc_dense(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Sub<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedRowStorage>
where
    T2: Scalar + Neg<Output = T2> + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = Matrix<T2, R, C, S>;

    fn sub(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spsub_csr_dense(self, rhs).unwrap()
    }
}

// Multiplication

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Mul<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T1: Scalar + Mul<T2>,
    <T1 as Mul<T2>>::Output: Scalar + AddAssign + Zero,
    T2: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Mul<T2>>::Output>;

    fn mul(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spmm_csr_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Mul<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Mul<T2>,
    <T1 as Mul<T2>>::Output: Scalar + AddAssign + Zero,
    T2: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Mul<T2>>::Output>;

    fn mul(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spmm_csc_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Mul<CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>
where
    T1: Scalar + Mul<T2>,
    <T1 as Mul<T2>>::Output: Scalar + AddAssign + Zero,
    T2: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CsrMatrix<<T1 as Mul<T2>>::Output>;

    fn mul(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>) -> Self::Output {
        spmm_csc_csc(self, rhs).unwrap()
    }
}

impl<T1, T2, MO1, MO2, MI1, MI2, D1, D2> Mul<CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>>
    for CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>
where
    T2: Scalar + Mul<T1>,
    <T2 as Mul<T1>>::Output: Scalar + AddAssign + Zero,
    T1: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    type Output = CscMatrix<<T2 as Mul<T1>>::Output>;

    fn mul(self, rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>) -> Self::Output {
        spmm_csr_csr(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Mul<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedRowStorage>
where
    T2: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C>,
    T1: Scalar + Mul<T2>,
    <T1 as Mul<T2>>::Output: Scalar + Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = CscMatrix<<T1 as Mul<T2>>::Output>;

    fn mul(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spmm_csr_dense(self, rhs).unwrap()
    }
}

impl<T1, T2, R, C, S, MO, MI, D> Mul<Matrix<T2, R, C, S>>
    for CsMatrix<T1, MO, MI, D, CompressedColumnStorage>
where
    T2: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C>,
    T1: Scalar + Mul<T2>,
    <T1 as Mul<T2>>::Output: Scalar + Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    type Output = CsrMatrix<<T1 as Mul<T2>>::Output>;

    fn mul(self, rhs: Matrix<T2, R, C, S>) -> Self::Output {
        spmm_csc_dense(self, rhs).unwrap()
    }
}

// Scalars

macro_rules! impl_sparse_scalar_product_and_div {
    ($($t:ty)*) =>  ($(
        impl<T1, MO, MI, D, C> Mul<$t> for CsMatrix<T1, MO, MI, D, C>
        where
            T1: Scalar + Mul<$t>,
            <T1 as Mul<$t>>::Output: Scalar,
            MO: Borrow<[usize]>,
            MI: Borrow<[usize]>,
            D: Borrow<[T1]>,
            C: Compression,
        {
            type Output =
                CsMatrix<<T1 as Mul<$t>>::Output, MO, MI, Vec<<T1 as Mul<$t>>::Output>, C>;

            fn mul(self, rhs: $t) -> Self::Output {
                sp_cs_scalar_prod(self, rhs)
            }
        }

        impl<T1, MO, MI, D, C> Mul<CsMatrix<T1, MO, MI, D, C>> for $t
        where
            T1: Scalar + Mul<$t>,
            <T1 as Mul<$t>>::Output: Scalar,
            MO: Borrow<[usize]>,
            MI: Borrow<[usize]>,
            D: Borrow<[T1]>,
            C: Compression,
        {
            type Output =
                CsMatrix<<T1 as Mul<$t>>::Output, MO, MI, Vec<<T1 as Mul<$t>>::Output>, C>;

            fn mul(self, rhs: CsMatrix<T1, MO, MI, D, C>) -> Self::Output {
                sp_cs_scalar_prod(rhs, self)
            }
        }

        impl<T1, MO, MI, D, C> Div<$t> for CsMatrix<T1, MO, MI, D, C>
        where
            T1: Scalar + Div<$t>,
            <T1 as Div<$t>>::Output: Scalar,
            MO: Borrow<[usize]>,
            MI: Borrow<[usize]>,
            D: Borrow<[T1]>,
            C: Compression,
        {
            type Output =
                CsMatrix<<T1 as Div<$t>>::Output, MO, MI, Vec<<T1 as Div<$t>>::Output>, C>;

            fn div(self, rhs: $t) -> Self::Output {
                sp_cs_scalar_div(self, rhs)
            }
        }
    )*)
}

impl_sparse_scalar_product_and_div!(isize usize u8 i8 u16 i16 u32 i32 u64 i64 f32 f64);
