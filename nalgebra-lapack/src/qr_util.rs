use crate::{DiagonalKind, LapackErrorCode, Side, Transposition, TriangularStructure, qr::QrReal};
use na::{
    Dim, DimMin, DimMinimum, IsContiguous, Matrix, RawStorage, RawStorageMut, RealField, Vector,
};
use num::{ConstOne, Zero};

#[derive(Debug, PartialEq, thiserror::Error)]
/// error type
pub enum Error {
    #[error("incorrect matrix dimensions")]
    /// incorrect matrix dims
    Dimensions,
    #[error("Lapack returned with error: {0}")]
    /// lapack backend returned error
    Lapack(#[from] LapackErrorCode),
    #[error("QR decomposition for underdetermined systems not supported")]
    /// underdetermined system
    Underdetermined,
    #[error("Matrix has rank zero")]
    /// matrix has zero rank
    ZeroRank,
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
    T: QrReal + Zero + RealField,
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
    T: QrReal + Zero + RealField,
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
    T: QrReal + Zero + RealField,
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
    T: QrReal + Zero + RealField,
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

/// this factors out solving a the A X = B in a least squares sense, given a
/// lapack qr decomposition of matrix A (in qr, tau). This also needs an explicit
/// rank for the matrix, which should be set to full rank for unpivoted QR.
///
/// This solver does not do the final row permutation necessary for col-pivoted
/// qr. For unpivoted QR, no extra permutation is necessary anyways.
pub(crate) fn qr_solve_mut_with_rank_unpermuted<T, R1, C1, S1, C2: Dim, S3, S2, S4>(
    qr: &Matrix<T, R1, C1, S1>,
    tau: &Vector<T, DimMinimum<R1, C1>, S4>,
    rank: u16,
    x: &mut Matrix<T, C1, C2, S2>,
    mut b: Matrix<T, R1, C2, S3>,
) -> Result<(), Error>
where
    T: QrReal + Zero + RealField,
    R1: DimMin<C1>,
    C1: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    S3: RawStorageMut<T, R1, C2> + IsContiguous,
    S2: RawStorageMut<T, C1, C2> + IsContiguous,
    S4: RawStorage<T, <R1 as DimMin<C1>>::Output> + IsContiguous,
{
    if b.nrows() != qr.nrows() {
        return Err(Error::Dimensions);
    }

    if qr.nrows() < qr.ncols() || qr.nrows() == 0 || qr.ncols() == 0 {
        return Err(Error::Underdetermined);
    }

    if x.ncols() != b.ncols() || x.nrows() != qr.ncols() {
        return Err(Error::Dimensions);
    }

    q_tr_mul_mut(qr, tau, &mut b)?;

    if rank == 0 {
        return Err(Error::ZeroRank);
    }

    debug_assert!(rank as usize <= qr.ncols().min(qr.nrows()));

    if (rank as usize) < qr.ncols() {
        x.view_mut((rank as usize, 0), (x.nrows() - rank as usize, x.ncols()))
            .iter_mut()
            .for_each(|val| val.set_zero());
    }

    let x_cols = x.ncols();
    x.view_mut((0, 0), (rank as usize, x_cols))
        .copy_from(&b.view((0, 0), (rank as usize, x_cols)));

    let ldb: i32 = x
        .nrows()
        .try_into()
        .expect("integer dimensions out of bounds");

    // SAFETY: input and dimensions according to lapack spec, see
    // https://www.netlib.org/lapack/explore-html/d4/dc1/group__trtrs_gab0b6a7438a7eb98fe2ab28e6c4d84b21.html#gab0b6a7438a7eb98fe2ab28e6c4d84b21
    unsafe {
        T::xtrtrs(
            TriangularStructure::Upper,
            Transposition::No,
            DiagonalKind::NonUnit,
            rank.try_into().expect("rank out of bounds"),
            x.ncols()
                .try_into()
                .expect("integer dimensions out of bounds"),
            qr.as_slice(),
            qr.nrows()
                .try_into()
                .expect("integer dimensions out of bounds"),
            x.as_mut_slice(),
            ldb,
        )?;
    }

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
    T: QrReal,
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

/// multiply R*B and place the result in B, where R is the upper triangular matrix
/// in a qr decomposition as computed by lapack.
pub fn r_mul_mut<T, R1, C1, S1, C2, S2>(
    qr: &Matrix<T, R1, C1, S1>,
    b: &mut Matrix<T, C1, C2, S2>,
) -> Result<(), Error>
where
    T: QrReal + ConstOne,
    R1: Dim,
    C1: Dim,
    C2: Dim,
    S1: RawStorage<T, R1, C1> + IsContiguous,
    S2: RawStorageMut<T, C1, C2> + IsContiguous,
{
    if qr.ncols() != b.nrows() {
        return Err(Error::Dimensions);
    }

    unsafe { left_multiply_r_mut(qr, Transposition::No, b)? };
    Ok(())
}

/// thin-ish wrapper around the lapack function [?TRMM](https://www.netlib.org/lapack/explore-html/dd/dab/group__trmm.html)
/// for multiplying the upper triangular part R or a QR decomposition with another
/// matrix.
///
/// The ?TRMM functions also allow scaling with a factor alpha, which
/// we always set to 1 and they allow the matrix to be upper or lower triangular,
/// we always use upper triangular. They also allow to multiply from right or
/// left, but the dimension of R in a QR decomposition only allows multiplication
/// from the left, I think.
///
/// # Safety
///
/// The dimensions of the matrices must be correct such that the multiplication
/// can be performed.
#[inline]
unsafe fn left_multiply_r_mut<T, R1, C1, S1, R2, C2, S2>(
    qr: &Matrix<T, R1, C1, S1>,
    transpose: Transposition,
    mat: &mut Matrix<T, R2, C2, S2>,
) -> Result<(), LapackErrorCode>
where
    T: QrReal + ConstOne,
    R1: Dim,
    C1: Dim,
    S2: RawStorageMut<T, R2, C2> + IsContiguous,
    R2: Dim,
    C2: Dim,
    S1: IsContiguous + RawStorage<T, R1, C1>,
{
    let m = mat
        .nrows()
        .try_into()
        .expect("integer dimensions out of bounds");
    let n = mat
        .ncols()
        .try_into()
        .expect("integer dimensions out of bounds");
    let lda = qr
        .nrows()
        .try_into()
        .expect("integer dimensions out of bounds");
    let ldb = mat
        .nrows()
        .try_into()
        .expect("integer dimensions out of bounds");

    unsafe {
        T::xtrmm(
            Side::Left,
            TriangularStructure::Upper,
            transpose,
            DiagonalKind::NonUnit,
            m,
            n,
            T::ONE,
            qr.as_slice(),
            lda,
            mat.as_mut_slice(),
            ldb,
        );
    }
    Ok(())
}
