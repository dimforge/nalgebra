use alga::general::{ClosedAdd, ClosedMul};
#[cfg(feature = "std")]
use matrixmultiply;
use num::{One, Signed, Zero};
#[cfg(feature = "std")]
use std::mem;

use base::allocator::Allocator;
use base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use base::dimension::{Dim, Dynamic, U1, U2, U3, U4};
use base::storage::{Storage, StorageMut};
use base::{DefaultAllocator, Matrix, Scalar, SquareMatrix, Vector};

impl<N: Scalar + PartialOrd + Signed, D: Dim, S: Storage<N, D>> Vector<N, D, S> {

    /// Computes the index of the vector component with the largest value.
    #[inline]
    pub fn imax(&self) -> usize {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index of the vector component with the largest absolute value.
    #[inline]
    pub fn iamax(&self) -> usize {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index of the vector component with the smallest value.
    #[inline]
    pub fn imin(&self) -> usize {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val < the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index of the vector component with the smallest absolute value.
    #[inline]
    pub fn iamin(&self) -> usize {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val < the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }
}

impl<N: Scalar + PartialOrd + Signed, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the index of the matrix component with the largest absolute value.
    #[inline]
    pub fn iamax_full(&self) -> (usize, usize) {
        assert!(!self.is_empty(), "The input matrix must not be empty.");

        let mut the_max = unsafe { self.get_unchecked(0, 0).abs() };
        let mut the_ij = (0, 0);

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = unsafe { self.get_unchecked(i, j).abs() };

                if val > the_max {
                    the_max = val;
                    the_ij = (i, j);
                }
            }
        }

        the_ij
    }
}

impl<N, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
{
    /// The dot product between two matrices (seen as vectors).
    ///
    /// Note that this is **not** the matrix multiplication as in, e.g., numpy. For matrix
    /// multiplication, use one of: `.gemm`, `mul_to`, `.mul`, `*`.
    #[inline]
    pub fn dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        assert!(
            self.nrows() == rhs.nrows(),
            "Dot product dimensions mismatch."
        );

        // So we do some special cases for common fixed-size vectors of dimension lower than 8
        // because the `for` loop below won't be very efficient on those.
        if (R::is::<U2>() || R2::is::<U2>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = *self.get_unchecked(0, 0) * *rhs.get_unchecked(0, 0);
                let b = *self.get_unchecked(1, 0) * *rhs.get_unchecked(1, 0);

                return a + b;
            }
        }
        if (R::is::<U3>() || R2::is::<U3>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = *self.get_unchecked(0, 0) * *rhs.get_unchecked(0, 0);
                let b = *self.get_unchecked(1, 0) * *rhs.get_unchecked(1, 0);
                let c = *self.get_unchecked(2, 0) * *rhs.get_unchecked(2, 0);

                return a + b + c;
            }
        }
        if (R::is::<U4>() || R2::is::<U4>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let mut a = *self.get_unchecked(0, 0) * *rhs.get_unchecked(0, 0);
                let mut b = *self.get_unchecked(1, 0) * *rhs.get_unchecked(1, 0);
                let c = *self.get_unchecked(2, 0) * *rhs.get_unchecked(2, 0);
                let d = *self.get_unchecked(3, 0) * *rhs.get_unchecked(3, 0);

                a += c;
                b += d;

                return a + b;
            }
        }

        // All this is inspired from the "unrolled version" discussed in:
        // http://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/
        //
        // And this comment from bluss:
        // https://users.rust-lang.org/t/how-to-zip-two-slices-efficiently/2048/12
        let mut res = N::zero();

        // We have to define them outside of the loop (and not inside at first assignment)
        // otherwise vectorization won't kick in for some reason.
        let mut acc0;
        let mut acc1;
        let mut acc2;
        let mut acc3;
        let mut acc4;
        let mut acc5;
        let mut acc6;
        let mut acc7;

        for j in 0..self.ncols() {
            let mut i = 0;

            acc0 = N::zero();
            acc1 = N::zero();
            acc2 = N::zero();
            acc3 = N::zero();
            acc4 = N::zero();
            acc5 = N::zero();
            acc6 = N::zero();
            acc7 = N::zero();

            while self.nrows() - i >= 8 {
                acc0 += unsafe { *self.get_unchecked(i + 0, j) * *rhs.get_unchecked(i + 0, j) };
                acc1 += unsafe { *self.get_unchecked(i + 1, j) * *rhs.get_unchecked(i + 1, j) };
                acc2 += unsafe { *self.get_unchecked(i + 2, j) * *rhs.get_unchecked(i + 2, j) };
                acc3 += unsafe { *self.get_unchecked(i + 3, j) * *rhs.get_unchecked(i + 3, j) };
                acc4 += unsafe { *self.get_unchecked(i + 4, j) * *rhs.get_unchecked(i + 4, j) };
                acc5 += unsafe { *self.get_unchecked(i + 5, j) * *rhs.get_unchecked(i + 5, j) };
                acc6 += unsafe { *self.get_unchecked(i + 6, j) * *rhs.get_unchecked(i + 6, j) };
                acc7 += unsafe { *self.get_unchecked(i + 7, j) * *rhs.get_unchecked(i + 7, j) };
                i += 8;
            }

            res += acc0 + acc4;
            res += acc1 + acc5;
            res += acc2 + acc6;
            res += acc3 + acc7;

            for k in i..self.nrows() {
                res += unsafe { *self.get_unchecked(k, j) * *rhs.get_unchecked(k, j) }
            }
        }

        res
    }

    /// The dot product between the transpose of `self` and `rhs`.
    #[inline]
    pub fn tr_dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == rhs.shape(),
            "Transposed dot product dimension mismatch."
        );

        let mut res = N::zero();

        for j in 0..self.nrows() {
            for i in 0..self.ncols() {
                res += unsafe { *self.get_unchecked(j, i) * *rhs.get_unchecked(i, j) }
            }
        }

        res
    }
}

fn array_axpy<N>(y: &mut [N], a: N, x: &[N], beta: N, stride1: usize, stride2: usize, len: usize)
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
{
    for i in 0..len {
        unsafe {
            let y = y.get_unchecked_mut(i * stride1);
            *y = a * *x.get_unchecked(i * stride2) + beta * *y;
        }
    }
}

fn array_ax<N>(y: &mut [N], a: N, x: &[N], stride1: usize, stride2: usize, len: usize)
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
{
    for i in 0..len {
        unsafe {
            *y.get_unchecked_mut(i * stride1) = a * *x.get_unchecked(i * stride2);
        }
    }
}

impl<N, D: Dim, S> Vector<N, D, S>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
    S: StorageMut<N, D>,
{
    /// Computes `self = a * x + b * self`.
    ///
    /// If be is zero, `self` is never read from.
    #[inline]
    pub fn axpy<D2: Dim, SB>(&mut self, a: N, x: &Vector<N, D2, SB>, b: N)
    where
        SB: Storage<N, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        assert_eq!(self.nrows(), x.nrows(), "Axpy: mismatched vector shapes.");

        let rstride1 = self.strides().0;
        let rstride2 = x.strides().0;

        let y = self.data.as_mut_slice();
        let x = x.data.as_slice();

        if !b.is_zero() {
            array_axpy(y, a, x, b, rstride1, rstride2, x.len());
        } else {
            array_ax(y, a, x, rstride1, rstride2, x.len());
        }
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    #[inline]
    pub fn gemv<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, R2> + AreMultipliable<R2, C2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            ncols2 == dim3 && dim1 == nrows2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            return;
        }

        // FIXME: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy(alpha * val, &col2, beta);

        for j in 1..ncols2 {
            let col2 = a.column(j);
            let val = unsafe { *x.vget_unchecked(j) };

            self.axpy(alpha * val, &col2, N::one());
        }
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a **symmetric** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read. If `self` is read, only its lower-triangular part
    /// (including the diagonal) is actually read.
    #[inline]
    pub fn gemv_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &SquareMatrix<N, D2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        let dim1 = self.nrows();
        let dim2 = a.nrows();
        let dim3 = x.nrows();

        assert!(
            a.is_square(),
            "Syetric gemv: the input matrix must be square."
        );
        assert!(
            dim2 == dim3 && dim1 == dim2,
            "Symmetric gemv: dimensions mismatch."
        );

        if dim2 == 0 {
            return;
        }

        // FIXME: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy(alpha * val, &col2, beta);
        self[0] += alpha * x.rows_range(1..).dot(&a.slice_range(1.., 0));

        for j in 1..dim2 {
            let col2 = a.column(j);
            let dot = x.rows_range(j..).dot(&col2.rows_range(j..));

            let val;
            unsafe {
                val = *x.vget_unchecked(j);
                *self.vget_unchecked_mut(j) += alpha * dot;
            }
            self.rows_range_mut(j + 1..)
                .axpy(alpha * val, &col2.rows_range(j + 1..), N::one());
        }
    }

    /// Computes `self = alpha * a.transpose() * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    #[inline]
    pub fn gemv_tr<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            nrows2 == dim3 && dim1 == ncols2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            return;
        }

        if beta.is_zero() {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha * a.column(j).dot(x)
            }
        } else {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha * a.column(j).dot(x) + beta * *val;
            }
        }
    }
}

impl<N, R1: Dim, C1: Dim, S: StorageMut<N, R1, C1>> Matrix<N, R1, C1, S>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
{
    /// Computes `self = alpha * x * y.transpose() + beta * self`.
    ///
    /// If `beta` is zero, `self` is never read.
    #[inline]
    pub fn ger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let (nrows1, ncols1) = self.shape();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            nrows1 == dim2 && ncols1 == dim3,
            "ger: dimensions mismatch."
        );

        for j in 0..ncols1 {
            // FIXME: avoid bound checks.
            let val = unsafe { *y.vget_unchecked(j) };
            self.column_mut(j).axpy(alpha * val, x, beta);
        }
    }

    /// Computes `self = alpha * a * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    #[inline]
    pub fn gemm<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        b: &Matrix<N, R3, C3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<R2, C2, R3, C3>,
    {
        let ncols1 = self.ncols();

        #[cfg(feature = "std")]
        {
            // matrixmultiply can be used only if the std feature is available.
            let nrows1 = self.nrows();
            let (nrows2, ncols2) = a.shape();
            let (nrows3, ncols3) = b.shape();

            assert_eq!(
                ncols2, nrows3,
                "gemm: dimensions mismatch for multiplication."
            );
            assert_eq!(
                (nrows1, ncols1),
                (nrows2, ncols3),
                "gemm: dimensions mismatch for addition."
            );

            // We assume large matrices will be Dynamic but small matrices static.
            // We could use matrixmultiply for large statically-sized matrices but the performance
            // threshold to activate it would be different from SMALL_DIM because our code optimizes
            // better for statically-sized matrices.
            let is_dynamic = R1::is::<Dynamic>() || C1::is::<Dynamic>() || R2::is::<Dynamic>()
                || C2::is::<Dynamic>() || R3::is::<Dynamic>()
                || C3::is::<Dynamic>();
            // Threshold determined empirically.
            const SMALL_DIM: usize = 5;

            if is_dynamic && nrows1 > SMALL_DIM && ncols1 > SMALL_DIM && nrows2 > SMALL_DIM
                && ncols2 > SMALL_DIM
            {
                if N::is::<f32>() {
                    let (rsa, csa) = a.strides();
                    let (rsb, csb) = b.strides();
                    let (rsc, csc) = self.strides();

                    unsafe {
                        matrixmultiply::sgemm(
                            nrows2,
                            ncols2,
                            ncols3,
                            mem::transmute_copy(&alpha),
                            a.data.ptr() as *const f32,
                            rsa as isize,
                            csa as isize,
                            b.data.ptr() as *const f32,
                            rsb as isize,
                            csb as isize,
                            mem::transmute_copy(&beta),
                            self.data.ptr_mut() as *mut f32,
                            rsc as isize,
                            csc as isize,
                        );
                    }
                    return;
                } else if N::is::<f64>() {
                    let (rsa, csa) = a.strides();
                    let (rsb, csb) = b.strides();
                    let (rsc, csc) = self.strides();

                    unsafe {
                        matrixmultiply::dgemm(
                            nrows2,
                            ncols2,
                            ncols3,
                            mem::transmute_copy(&alpha),
                            a.data.ptr() as *const f64,
                            rsa as isize,
                            csa as isize,
                            b.data.ptr() as *const f64,
                            rsb as isize,
                            csb as isize,
                            mem::transmute_copy(&beta),
                            self.data.ptr_mut() as *mut f64,
                            rsc as isize,
                            csc as isize,
                        );
                    }
                    return;
                }
            }
        }

        for j1 in 0..ncols1 {
            // FIXME: avoid bound checks.
            self.column_mut(j1).gemv(alpha, a, &b.column(j1), beta);
        }
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    #[inline]
    pub fn gemm_tr<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        b: &Matrix<N, R3, C3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // FIXME: avoid bound checks.
            self.column_mut(j1).gemv_tr(alpha, a, &b.column(j1), beta);
        }
    }
}

impl<N, R1: Dim, C1: Dim, S: StorageMut<N, R1, C1>> Matrix<N, R1, C1, S>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
{
    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    #[inline]
    pub fn ger_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let dim1 = self.nrows();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            self.is_square(),
            "Symmetric ger: the input matrix must be square."
        );
        assert!(dim1 == dim2 && dim1 == dim3, "ger: dimensions mismatch.");

        for j in 0..dim1 {
            let val = unsafe { *y.vget_unchecked(j) };
            let subdim = Dynamic::new(dim1 - j);
            // FIXME: avoid bound checks.
            self.generic_slice_mut((j, j), (subdim, U1)).axpy(
                alpha * val,
                &x.rows_range(j..),
                beta,
            );
        }
    }
}

impl<N, D1: Dim, S: StorageMut<N, D1, D1>> SquareMatrix<N, D1, S>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
{
    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    pub fn quadform_tr_with_workspace<D2, S2, R3, C3, S3, D4, S4>(
        &mut self,
        work: &mut Vector<N, D2, S2>,
        alpha: N,
        lhs: &Matrix<N, R3, C3, S3>,
        mid: &SquareMatrix<N, D4, S4>,
        beta: N,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S2: StorageMut<N, D2>,
        S3: Storage<N, R3, C3>,
        S4: Storage<N, D4, D4>,
        ShapeConstraint: DimEq<D1, D2> + DimEq<D1, R3> + DimEq<D2, R3> + DimEq<C3, D4>,
    {
        work.gemv(N::one(), lhs, &mid.column(0), N::zero());
        self.ger(alpha, work, &lhs.column(0), beta);

        for j in 1..mid.ncols() {
            work.gemv(N::one(), lhs, &mid.column(j), N::zero());
            self.ger(alpha, work, &lhs.column(j), N::one());
        }
    }

    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This allocates a workspace vector of dimension D1 for intermediate results.
    /// Use `.quadform_tr_with_workspace(...)` instead to avoid allocations.
    pub fn quadform_tr<R3, C3, S3, D4, S4>(
        &mut self,
        alpha: N,
        lhs: &Matrix<N, R3, C3, S3>,
        mid: &SquareMatrix<N, D4, S4>,
        beta: N,
    ) where
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S3: Storage<N, R3, C3>,
        S4: Storage<N, D4, D4>,
        ShapeConstraint: DimEq<D1, D1> + DimEq<D1, R3> + DimEq<C3, D4>,
        DefaultAllocator: Allocator<N, D1>,
    {
        let mut work = unsafe { Vector::new_uninitialized_generic(self.data.shape().0, U1) };
        self.quadform_tr_with_workspace(&mut work, alpha, lhs, mid, beta)
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    pub fn quadform_with_workspace<D2, S2, D3, S3, R4, C4, S4>(
        &mut self,
        work: &mut Vector<N, D2, S2>,
        alpha: N,
        mid: &SquareMatrix<N, D3, S3>,
        rhs: &Matrix<N, R4, C4, S4>,
        beta: N,
    ) where
        D2: Dim,
        D3: Dim,
        R4: Dim,
        C4: Dim,
        S2: StorageMut<N, D2>,
        S3: Storage<N, D3, D3>,
        S4: Storage<N, R4, C4>,
        ShapeConstraint:
            DimEq<D3, R4> + DimEq<D1, C4> + DimEq<D2, D3> + AreMultipliable<C4, R4, D2, U1>,
    {
        work.gemv(N::one(), mid, &rhs.column(0), N::zero());
        self.column_mut(0).gemv_tr(alpha, &rhs, work, beta);

        for j in 1..rhs.ncols() {
            work.gemv(N::one(), mid, &rhs.column(j), N::zero());
            self.column_mut(j).gemv_tr(alpha, &rhs, work, beta);
        }
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This allocates a workspace vector of dimension D2 for intermediate results.
    /// Use `.quadform_with_workspace(...)` instead to avoid allocations.
    pub fn quadform<D2, S2, R3, C3, S3>(
        &mut self,
        alpha: N,
        mid: &SquareMatrix<N, D2, S2>,
        rhs: &Matrix<N, R3, C3, S3>,
        beta: N,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        S2: Storage<N, D2, D2>,
        S3: Storage<N, R3, C3>,
        ShapeConstraint: DimEq<D2, R3> + DimEq<D1, C3> + AreMultipliable<C3, R3, D2, U1>,
        DefaultAllocator: Allocator<N, D2>,
    {
        let mut work = unsafe { Vector::new_uninitialized_generic(mid.data.shape().0, U1) };
        self.quadform_with_workspace(&mut work, alpha, mid, rhs, beta)
    }
}
