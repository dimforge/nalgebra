use ::{Real, Dim, Matrix, VectorN, RowVectorN, DefaultAllocator, U1, VectorSliceN};
use storage::Storage;
use allocator::Allocator;

impl<N: Real, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    #[inline]
    pub fn compress_rows(&self, f: impl Fn(VectorSliceN<N, R, S::RStride, S::CStride>) -> N) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {

        let ncols = self.data.shape().1;
        let mut res = unsafe { RowVectorN::new_uninitialized_generic(U1, ncols) };

        for i in 0..ncols.value() {
            // FIXME: avoid bound checking of column.
            unsafe { *res.get_unchecked_mut(0, i) = f(self.column(i)); }
        }

        res
    }

    #[inline]
    pub fn compress_rows_tr(&self, f: impl Fn(VectorSliceN<N, R, S::RStride, S::CStride>) -> N) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {

        let ncols = self.data.shape().1;
        let mut res = unsafe { VectorN::new_uninitialized_generic(ncols, U1) };

        for i in 0..ncols.value() {
            // FIXME: avoid bound checking of column.
            unsafe { *res.vget_unchecked_mut(i) = f(self.column(i)); }
        }

        res
    }

    #[inline]
    pub fn compress_columns(&self, init: VectorN<N, R>, f: impl Fn(&mut VectorN<N, R>, VectorSliceN<N, R, S::RStride, S::CStride>)) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let mut res = init;

        for i in 0..self.ncols() {
            f(&mut res, self.column(i))
        }

        res
    }
}

impl<N: Real, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /*
     *
     * Sum computation.
     *
     */
    #[inline]
    pub fn sum(&self) -> N {
        self.iter().cloned().fold(N::zero(), |a, b| a + b)
    }

    #[inline]
    pub fn row_sum(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.sum())
    }

    #[inline]
    pub fn row_sum_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.sum())
    }

    #[inline]
    pub fn column_sum(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let nrows = self.data.shape().0;
        self.compress_columns(VectorN::zeros_generic(nrows, U1), |out, col| {
            out.axpy(N::one(), &col, N::one())
        })
    }

    /*
     *
     * Variance computation.
     *
     */
    #[inline]
    pub fn variance(&self) -> N {
        if self.len() == 0 {
            N::zero()
        } else {
            let val = self.iter().cloned().fold((N::zero(), N::zero()), |a, b| (a.0 + b * b, a.1 + b));
            let denom = N::one() / ::convert::<_, N>(self.len() as f64);
            val.0 * denom - (val.1 * denom) * (val.1 * denom)
        }
    }

    #[inline]
    pub fn row_variance(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.variance())
    }

    #[inline]
    pub fn row_variance_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.variance())
    }

    #[inline]
    pub fn column_variance(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let (nrows, ncols) = self.data.shape();

        let mut mean = self.column_mean();
        mean.apply(|e| -(e * e));

        let denom = N::one() / ::convert::<_, N>(ncols.value() as f64);
        self.compress_columns(mean, |out, col| {
            for i in 0..nrows.value() {
                unsafe {
                    let val = col.vget_unchecked(i);
                    *out.vget_unchecked_mut(i) += denom * *val * *val
                }
            }
        })
    }

    /*
     *
     * Mean computation.
     *
     */
    #[inline]
    pub fn mean(&self) -> N {
        if self.len() == 0 {
            N::zero()
        } else {
            self.sum() / ::convert(self.len() as f64)
        }
    }

    #[inline]
    pub fn row_mean(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.mean())
    }

    #[inline]
    pub fn row_mean_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.mean())
    }

    #[inline]
    pub fn column_mean(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let (nrows, ncols) = self.data.shape();
        let denom = N::one() / ::convert::<_, N>(ncols.value() as f64);
        self.compress_columns(VectorN::zeros_generic(nrows, U1), |out, col| {
            out.axpy(denom, &col, N::one())
        })
    }
}