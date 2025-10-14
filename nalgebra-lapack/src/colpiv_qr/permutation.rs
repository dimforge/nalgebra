use super::{ColPivQrScalar, Error};
use na::{
    DefaultAllocator, Dim, IsContiguous, Matrix, OVector, RawStorageMut, allocator::Allocator,
};

#[cfg(test)]
mod test;

#[derive(Debug, Clone, PartialEq, Eq)]
/// describes an orthogonal permutation matrix `P` that can be used to permute
/// the rows or columns of an appropriately shaped matrix. Due to LAPACK
/// internals, an instance must be mutably borrowed when applying the
/// permutation, but the permutation itself will remain unchanged on success.
///
/// **Note**: the associated functions take `Self` by mutable reference due
/// to the LAPACK internal implementation of the permutation logic.
pub struct Permutation<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    jpvt: OVector<i32, D>,
}

impl<'a, D> Permutation<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Apply the column permutation to a matrix `A`, equivalent to `P A`.
    #[inline]
    pub fn permute_cols_mut<T, R, S>(&mut self, mat: &mut Matrix<T, R, D, S>) -> Result<(), Error>
    where
        R: Dim,
        S: RawStorageMut<T, R, D> + IsContiguous,
        T: ColPivQrScalar,
    {
        //@note(geo-ant) due to the LAPACK internal logic in the jpvt vector, we have
        // to invert the forward/backward argument here
        self.apply_cols_mut(false, mat)
    }

    /// Apply the inverse column permutation to a matrix `A`, equivalent to `P^T A`.
    #[inline]
    pub fn inv_permute_cols_mut<T, R, S>(
        &mut self,
        mat: &mut Matrix<T, R, D, S>,
    ) -> Result<(), Error>
    where
        R: Dim,
        S: RawStorageMut<T, R, D> + IsContiguous,
        T: ColPivQrScalar,
    {
        self.apply_cols_mut(true, mat)
    }

    /// Apply the row permutation to a matrix `A`, equivalent to `P A`.
    #[inline]
    pub fn permute_rows_mut<T, C, S>(&mut self, mat: &mut Matrix<T, D, C, S>) -> Result<(), Error>
    where
        C: Dim,
        S: RawStorageMut<T, D, C> + IsContiguous,
        T: ColPivQrScalar,
    {
        self.apply_rows_mut(false, mat)
    }

    /// Apply the inverse row permutation to a matrix `A`, equivalent to `P^T A`.
    #[inline]
    pub fn inv_permute_rows_mut<T, C, S>(
        &mut self,
        mat: &mut Matrix<T, D, C, S>,
    ) -> Result<(), Error>
    where
        C: Dim,
        S: RawStorageMut<T, D, C> + IsContiguous,
        T: ColPivQrScalar,
    {
        self.apply_rows_mut(true, mat)
    }

    #[inline]
    /// a thin wrapper around lapacks xLAPMR
    fn apply_rows_mut<T, C, S>(
        &mut self,
        forward: bool,
        mat: &mut Matrix<T, D, C, S>,
    ) -> Result<(), Error>
    where
        C: Dim,
        S: RawStorageMut<T, D, C> + IsContiguous,
        T: ColPivQrScalar,
    {
        if mat.nrows() != self.jpvt.len() {
            return Err(Error::Dimensions);
        }

        let m = mat
            .nrows()
            .try_into()
            .expect("matrix dimensions out of bounds");
        let n = mat
            .ncols()
            .try_into()
            .expect("matrix dimensions out of bounds");

        // SAFETY: inputs according to spec, see
        // https://www.netlib.org/lapack/explore-html/d3/d10/group__lapmr.html
        unsafe {
            T::xlapmr(
                forward,
                m,
                n,
                mat.as_mut_slice(),
                m,
                self.jpvt.as_mut_slice(),
            )?;
        }

        Ok(())
    }

    #[inline]
    /// a thin wrapper around LAPACKS xLAMPT
    fn apply_cols_mut<T, R, S>(
        &mut self,
        forward: bool,
        mat: &mut Matrix<T, R, D, S>,
    ) -> Result<(), Error>
    where
        R: Dim,
        S: RawStorageMut<T, R, D> + IsContiguous,
        T: ColPivQrScalar,
    {
        if mat.ncols() != self.jpvt.len() {
            return Err(Error::Dimensions);
        }
        let m = mat
            .nrows()
            .try_into()
            .expect("matrix dimensions out of bounds");
        let n = mat
            .ncols()
            .try_into()
            .expect("matrix dimensions out of bounds");

        // SAFETY: arguments provided according to spec
        // https://www.netlib.org/lapack/explore-html/d0/dcb/group__lapmt.html
        unsafe {
            T::xlapmt(
                forward,
                m,
                n,
                mat.as_mut_slice(),
                m,
                self.jpvt.as_mut_slice(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn new(jpvt: OVector<i32, D>) -> Self {
        Self { jpvt }
    }
}
