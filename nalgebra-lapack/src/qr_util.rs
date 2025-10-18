use crate::{
    Side, Transposition,
    colpiv_qr::{ColPivQrReal, error::LapackErrorCode},
};
use na::{
    Dim, DimMin, DimMinimum, IsContiguous, Matrix, RawStorage, RawStorageMut, RealField, Vector,
};
use num::Zero;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("incorrect matrix dimensions")]
    Dimensions,
    #[error("Lapack returned with error: {0}")]
    Lapack(#[from] LapackErrorCode),
}

/// Thin wrapper around certain invokation of `multiply_q_mut`, where:
/// * `qr`: contains the lapack-style qr decomposition of a matrix A
/// * `tau`: scalar factors of the elementary reflectors
/// * `b`: matrix B described below
///
/// Efficiently calculate the matrix product `Q B` of the factor `Q` with a
/// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
/// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
/// must only be considered valid when the function returns without error.
pub(crate) fn q_mul_mut<T, R1, C1, S1, C2, S2, S3>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S3>,
    b: &mut Matrix<T, R1, C2, S2>,
) -> Result<(), Error>
where
    T: ColPivQrReal + Zero + RealField,
    R1: DimMin<C1>,
    C1: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    C2: Dim,
    S2: RawStorageMut<T, R1, C2> + IsContiguous,
    S3: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    if b.nrows() != qr.nrows() {
        return Err(Error::Dimensions);
    }
    if qr.ncols().min(qr.nrows()) != tau.len() {
        return Err(Error::Dimensions);
    }
    // SAFETY: matrix has the correct dimensions for operation Q*B
    unsafe { multiply_q_mut(qr, tau, b, Side::Left, Transposition::No)? };
    Ok(())
}

/// Thin wrapper around certain invokation of `multiply_q_mut`, where:
/// * `qr`: contains the lapack-style qr decomposition of a matrix A
/// * `tau`: scalar factors of the elementary reflectors
/// * `b`: matrix B described below
///
/// Efficiently calculate the matrix product `Q^T B` of the factor `Q` with a
/// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
/// we require `B ∈ R^(m ⨯ k)`. The product is calculated in place and
/// must only be considered valid when the function returns without error.
pub(crate) fn q_tr_mul_mut<T, R1, C1, S1, C2, S2, S3>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S3>,
    b: &mut Matrix<T, R1, C2, S2>,
) -> Result<(), Error>
where
    T: ColPivQrReal + Zero + RealField,
    R1: DimMin<C1>,
    C1: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    C2: Dim,
    C2: Dim,
    S2: RawStorageMut<T, R1, C2> + IsContiguous,
    S3: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    if b.nrows() != qr.nrows() {
        return Err(Error::Dimensions);
    }
    if qr.ncols().min(qr.nrows()) != tau.len() {
        return Err(Error::Dimensions);
    }
    // SAFETY: matrix has the correct dimensions for operation Q^T*B
    unsafe { multiply_q_mut(qr, tau, b, Side::Left, Transposition::Transpose)? };
    Ok(())
}

/// Thin wrapper around certain invokation of `multiply_q_mut`, where:
/// * `qr`: contains the lapack-style qr decomposition of a matrix A
/// * `tau`: scalar factors of the elementary reflectors
/// * `b`: matrix B described below
///
/// Efficiently calculate the matrix product `B Q` of the factor `Q` with a
/// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
/// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
/// must only be considered valid when the function returns without error.
pub(crate) fn mul_q_mut<T, R1, C1, S1, R2, S2, S3>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S3>,
    b: &mut Matrix<T, R2, R1, S2>,
) -> Result<(), Error>
where
    T: ColPivQrReal + Zero + RealField,
    R1: DimMin<C1>,
    C1: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    R2: Dim,
    S2: RawStorageMut<T, R2, R1> + IsContiguous,
    S3: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    if b.ncols() != qr.nrows() {
        return Err(Error::Dimensions);
    }
    if qr.ncols().min(qr.nrows()) != tau.len() {
        return Err(Error::Dimensions);
    }
    // SAFETY: matrix has the correct dimensions for operation B*Q
    unsafe { multiply_q_mut(qr, tau, b, Side::Right, Transposition::No)? };
    Ok(())
}

/// Thin wrapper around certain invokation of `multiply_q_mut`, where:
/// * `qr`: contains the lapack-style qr decomposition of a matrix A
/// * `tau`: scalar factors of the elementary reflectors
/// * `b`: matrix B described below
///
/// Efficiently calculate the matrix product `B Q^T` of the factor `Q` with a
/// given matrix `B`. `Q` acts as if it is a matrix of dimension `m ⨯ m`, so
/// we require `B ∈ R^(k ⨯ m)`. The product is calculated in place and
/// must only be considered valid when the function returns without error.
pub(crate) fn mul_q_tr_mut<T, R1, C1, S1, R2, S2, S3>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S3>,
    b: &mut Matrix<T, R2, R1, S2>,
) -> Result<(), Error>
where
    T: ColPivQrReal + Zero + RealField,
    R1: DimMin<C1>,
    C1: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    R2: Dim,
    S2: RawStorageMut<T, R2, R1> + IsContiguous,
    S3: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    if b.ncols() != qr.nrows() {
        return Err(Error::Dimensions);
    }
    if qr.ncols().min(qr.nrows()) != tau.len() {
        return Err(Error::Dimensions);
    }
    // SAFETY: matrix has the correct dimensions for operation B Q^T
    unsafe { multiply_q_mut(qr, tau, b, Side::Right, Transposition::Transpose)? }
    Ok(())
}

/// Thin-ish wrapper around the LAPACK function
/// [?ormqr](https://www.netlib.org/lapack/explore-html/d7/d50/group__unmqr.html),
/// which allows us to calculate either Q*B, Q^T*B, B*Q, B*Q^T for appropriately
/// shaped matrices B, without having to explicitly form Q. In this calculation
/// Q is constructed as if it were a square matrix of appropriate dimension.
///
/// # Safety
///
/// The dimensions of the matrices must be correct such that the multiplication
/// can be performed.
#[inline]
unsafe fn multiply_q_mut<T, R1, C1, S1, R2, C2, S2, S3>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S3>,
    mat: &mut Matrix<T, R2, C2, S2>,
    side: Side,
    transpose: Transposition,
) -> Result<(), LapackErrorCode>
where
    T: ColPivQrReal,
    R1: DimMin<C1>,
    C1: Dim,
    S2: RawStorageMut<T, R2, C2> + IsContiguous,
    R2: Dim,
    C2: Dim,
    S1: IsContiguous + RawStorage<T, R1, C1>,
    S3: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    let a = qr.as_slice();
    let lda = qr
        .nrows()
        .try_into()
        .expect("integer dimension out of range");
    let m = mat
        .nrows()
        .try_into()
        .expect("integer dimension out of range");
    let n = mat
        .ncols()
        .try_into()
        .expect("integer dimension out of range");
    let k = tau
        .len()
        .try_into()
        .expect("integer dimension out of range");
    let ldc = mat
        .nrows()
        .try_into()
        .expect("integer dimension out of range");
    let c = mat.as_mut_slice();
    let trans = transpose;
    let tau = tau.as_slice();

    // SAFETY: the containing function is unsafe and requires the correct
    // matrix dimensions as input
    let lwork = unsafe { T::xormqr_work_size(side, transpose, m, n, k, a, lda, tau, c, ldc)? };
    let mut work = vec![T::zero(); lwork as usize];

    // SAFETY: the containing function is unsafe and requires the correct
    // matrix dimensions as input
    unsafe {
        T::xormqr(side, trans, m, n, k, a, lda, tau, c, ldc, &mut work, lwork)?;
    }
    Ok(())
}
