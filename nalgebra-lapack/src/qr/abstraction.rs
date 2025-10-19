use crate::{qr::QrReal, qr_util, sealed::Sealed};
use na::{
    DefaultAllocator, Dim, DimMin, DimMinimum, IsContiguous, Matrix, OMatrix, OVector,
    RawStorageMut, RealField, Scalar, Storage, allocator::Allocator,
};
use num::Zero;
use qr_util::Error;

/// common functionality for the QR decomposition of a matrix `A` with or
/// without column-pivoting.
pub trait QrDecomposition<T, R, C>: Sealed
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<C>,
    R: DimMin<C>,
    C: Dim,
    T: Scalar + RealField + QrReal,
{
    #[doc(hidden)]
    /// get a reference to the internal represenation of the QR decomposition
    /// with the output of the lapack QR decomposition
    fn __lapack_qr_ref(&self) -> &OMatrix<T, R, C>;

    #[doc(hidden)]
    /// get a reference of the householder coefficients vector as computed by
    /// lapack
    fn __lapack_tau_ref(&self) -> &OVector<T, DimMinimum<R, C>>;

    #[inline]
    /// the number of rows of the original matrix `A`
    fn nrows(&self) -> usize {
        self.__lapack_qr_ref().nrows()
    }

    #[inline]
    /// the number of columns of the original matrix `A`
    fn ncols(&self) -> usize {
        self.__lapack_qr_ref().ncols()
    }

    #[inline]
    /// shape of the original matrix `A`
    fn shape_generic(&self) -> (R, C) {
        self.__lapack_qr_ref().shape_generic()
    }

    /// Solve the overdetermined linear system with the given right hand side
    /// in a least squares sense, see the comments on [Self::solve_mut].
    fn solve<C2: Dim, S>(&self, rhs: Matrix<T, R, C2, S>) -> Result<OMatrix<T, C, C2>, Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous + Storage<T, R, C2>,
        T: Zero,
        DefaultAllocator: Allocator<C, C2> + Allocator<R, C2>,
    {
        let (_, c2) = rhs.shape_generic();
        let (_, c) = self.shape_generic();
        let mut x = OMatrix::zeros_generic(c, c2);
        self.solve_mut(&mut x, rhs)?;
        Ok(x)
    }

    /// Solve the square or overdetermined system in `A X = B`, where `X ∈ R^(n ⨯ k)`,
    /// `B ∈ R^(m ⨯ k)`in a least-squares sense, such that `|| A X -B||^2`
    /// is minimized. The solution is placed into the matrix `X ∈ R^(m ⨯ k)`.
    ///
    /// Note that QR decomposition _does not_ typically give the minimum norm solution
    /// for `X`, only the residual is minimized which is typically what we want.
    ///
    /// This function might perform a small allocation.
    fn solve_mut<C2: Dim, S, S2>(
        &self,
        x: &mut Matrix<T, C, C2, S2>,
        b: Matrix<T, R, C2, S>,
    ) -> Result<(), Error>
    where
        S: RawStorageMut<T, R, C2> + IsContiguous,
        S2: RawStorageMut<T, C, C2> + IsContiguous,
        T: Zero;

    /// Efficiently calculate the matrix product `Q B` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    fn q_mul_mut<C2, S>(&self, b: &mut Matrix<T, R, C2, S>) -> Result<(), Error>
    where
        C2: Dim,
        S: RawStorageMut<T, R, C2> + IsContiguous,
    {
        qr_util::q_mul_mut(self.__lapack_qr_ref(), self.__lapack_tau_ref(), b)?;
        Ok(())
    }
    /// Efficiently calculate the matrix product `Q^T B` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    fn q_tr_mul_mut<C2, S>(&self, b: &mut Matrix<T, R, C2, S>) -> Result<(), Error>
    where
        C2: Dim,
        S: RawStorageMut<T, R, C2> + IsContiguous,
    {
        qr_util::q_tr_mul_mut(self.__lapack_qr_ref(), self.__lapack_tau_ref(), b)?;
        Ok(())
    }

    /// Efficiently calculate the matrix product `B Q` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    fn mul_q_mut<R2, S>(&self, b: &mut Matrix<T, R2, R, S>) -> Result<(), Error>
    where
        R2: Dim,
        S: RawStorageMut<T, R2, R> + IsContiguous,
    {
        qr_util::mul_q_mut(self.__lapack_qr_ref(), self.__lapack_tau_ref(), b)?;
        Ok(())
    }

    /// Efficiently calculate the matrix product `B Q^T` of the factor `Q` with a
    /// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
    /// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
    /// must only be considered valid when the function returns without error.
    fn mul_q_tr_mut<R2, S>(&self, b: &mut Matrix<T, R2, R, S>) -> Result<(), Error>
    where
        R2: Dim,
        S: RawStorageMut<T, R2, R> + IsContiguous,
    {
        qr_util::mul_q_tr_mut(self.__lapack_qr_ref(), self.__lapack_tau_ref(), b)?;
        Ok(())
    }

    /// Computes the orthonormal matrix `Q ∈ R^(m ⨯ n)` of this decomposition.
    /// Note that this matrix has _economy_ dimensions, which means it is not
    /// square unless `A` is square. It satisfies `Q^T Q = I`. Note further
    /// that is is typically not necessary to compute `Q` explicitly. Rather,
    /// check if some of the provided multiplication functions can help to
    /// calculate the matrix products `Q B`, `B Q`, `Q^T B`, `B Q^T` more efficiently.
    ///
    /// This function allocates.
    fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, <R as DimMin<C>>::Output>,
        T: Zero,
    {
        let (nrows, ncols) = self.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return OMatrix::from_element_generic(nrows, min_nrows_ncols, T::zero());
        }

        let mut q = self
            .__lapack_qr_ref()
            .generic_view((0, 0), (nrows, min_nrows_ncols))
            .into_owned();

        let nrows = nrows.value() as i32;

        let lwork = unsafe {
            T::xorgqr_work_size(
                nrows,
                min_nrows_ncols.value() as i32,
                self.__lapack_tau_ref().len() as i32,
                q.as_mut_slice(),
                nrows,
                self.__lapack_tau_ref().as_slice(),
            )
            .expect("unexpected error in lapack backend")
        };

        let mut work = vec![T::zero(); lwork as usize];

        unsafe {
            T::xorgqr(
                nrows,
                min_nrows_ncols.value() as i32,
                self.__lapack_tau_ref().len() as i32,
                q.as_mut_slice(),
                nrows,
                self.__lapack_tau_ref().as_slice(),
                &mut work,
                lwork,
            )
            .expect("unexpected error in lapack backend")
        };
        q
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    /// Note that it's typically not necessary to construct this matrix directly
    /// and check if any of the provided multiplication functions can be used
    /// instead.
    ///
    /// This function allocates.
    #[inline]
    #[must_use]
    fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<R, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.shape_generic();
        self.__lapack_qr_ref()
            .rows_generic(0, nrows.min(ncols))
            .upper_triangle()
    }
}
