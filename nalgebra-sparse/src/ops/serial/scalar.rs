//! Module holding the various sparse-matrix scalar operation functions.

use crate::cs::{Compression, CsMatrix};
use std::{borrow::Borrow, ops::Mul};

/// Scalar product for sparse matrices.
///
/// This does not perform any checks to ensure that the Scalar is non-zero. This means that if zero
/// (or close to zero) is passed in, the resulting sparse matrix will store the final values as
/// explicit zeros.
pub fn sp_cs_scalar_prod<T1, T2, MO, MI, D, C>(
    cs: CsMatrix<T1, MO, MI, D, C>,
    scalar: T2,
) -> CsMatrix<<T1 as Mul<T2>>::Output, MO, MI, Vec<<T1 as Mul<T2>>::Output>, C>
where
    T1: Clone + Mul<T2>,
    T2: Clone,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
    C: Compression,
{
    let (rows, columns) = cs.shape();
    let (offsets, indices, data) = cs.disassemble();

    let data = data
        .borrow()
        .iter()
        .map(|x| x.clone() * scalar.clone())
        .collect();

    unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) }
}
