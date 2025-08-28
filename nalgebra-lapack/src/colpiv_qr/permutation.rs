use super::{ColPivQrScalar, Error};
use na::{
    DefaultAllocator, Dim, IsContiguous, Matrix, OVector, RawStorageMut, allocator::Allocator,
};

#[cfg(test)]
mod test;

#[derive(Debug, Clone, PartialEq, Eq)]
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
    ///
    //@todo(geo-ant) comment
    #[inline]
    pub fn permute_cols_mut<T, R, S>(&mut self, mat: &mut Matrix<T, R, D, S>) -> Result<(), Error>
    where
        R: Dim,
        S: RawStorageMut<T, R, D> + IsContiguous,
        T: ColPivQrScalar,
    {
        self.apply_cols_mut(true, mat)
    }

    ///
    //@todo(geo-ant) comment
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
        self.apply_cols_mut(false, mat)
    }
    ///
    //@todo(geo-ant) comment
    #[inline]
    pub fn permute_rows_mut<T, C, S>(&mut self, mat: &mut Matrix<T, D, C, S>) -> Result<(), Error>
    where
        C: Dim,
        S: RawStorageMut<T, D, C> + IsContiguous,
        T: ColPivQrScalar,
    {
        self.apply_rows_mut(true, mat)
    }

    ///
    //@todo(geo-ant) comment
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
        self.apply_rows_mut(false, mat)
    }

    #[inline]
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
            return Err(Error::Dimension);
        }

        let m = mat
            .nrows()
            .try_into()
            .expect("matrix dimensions out of bounds");
        let n = mat
            .ncols()
            .try_into()
            .expect("matrix dimensions out of bounds");

        T::xlapmr(
            forward,
            m,
            n,
            mat.as_mut_slice(),
            m,
            self.jpvt.as_mut_slice(),
        )
        .unwrap();
        Ok(())
    }

    #[inline]
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
            return Err(Error::Dimension);
        }
        let m = mat
            .nrows()
            .try_into()
            .expect("matrix dimensions out of bounds");
        let n = mat
            .ncols()
            .try_into()
            .expect("matrix dimensions out of bounds");

        T::xlapmt(
            forward,
            m,
            n,
            mat.as_mut_slice(),
            m,
            self.jpvt.as_mut_slice(),
        )
        .unwrap();
        Ok(())
    }

    pub(crate) fn new(jpvt: OVector<i32, D>) -> Self {
        Self { jpvt }
    }
}
