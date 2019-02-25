use base::allocator::Allocator;
use base::default_allocator::DefaultAllocator;
use base::dimension::{Dim, DimAdd, DimDiff, DimMax, DimMaximum, DimSub, DimSum};
use std::cmp;
use storage::Storage;
use {zero, Real, Vector, VectorN, U1};

/// Returns the convolution of the target vector and a kernel
///
/// # Arguments
///
/// * `vector` - A Vector with size > 0
/// * `kernel` - A Vector with size > 0
///
/// # Errors
/// Inputs must statisfy `vector.len() >= kernel.len() > 0`.
///
pub fn convolve_full<N, D1, D2, S1, S2>(
    vector: Vector<N, D1, S1>,
    kernel: Vector<N, D2, S2>,
) -> VectorN<N, DimDiff<DimSum<D1, D2>, U1>>
where
    N: Real,
    D1: DimAdd<D2>,
    D2: DimAdd<D1, Output = DimSum<D1, D2>>,
    DimSum<D1, D2>: DimSub<U1>,
    S1: Storage<N, D1>,
    S2: Storage<N, D2>,
    DefaultAllocator: Allocator<N, DimDiff<DimSum<D1, D2>, U1>>,
{
    let vec = vector.len();
    let ker = kernel.len();

    if ker == 0 || ker > vec {
        panic!("convolve_full expects `vector.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
    }

    let result_len = vector.data.shape().0.add(kernel.data.shape().0).sub(U1);
    let mut conv = VectorN::zeros_generic(result_len, U1);

    for i in 0..(vec + ker - 1) {
        let u_i = if i > vec { i - ker } else { 0 };
        let u_f = cmp::min(i, vec - 1);

        if u_i == u_f {
            conv[i] += vector[u_i] * kernel[(i - u_i)];
        } else {
            for u in u_i..(u_f + 1) {
                if i - u < ker {
                    conv[i] += vector[u] * kernel[(i - u)];
                }
            }
        }
    }
    conv
}

/// Returns the convolution of the vector and a kernel
/// The output convolution consists only of those elements that do not rely on the zero-padding.
/// # Arguments
///
/// * `vector` - A Vector with size > 0
/// * `kernel` - A Vector with size > 0
///
///
/// # Errors
/// Inputs must statisfy `vector.len() >= kernel.len() > 0`.
///
pub fn convolve_valid<N, D1, D2, S1, S2>(
    vector: Vector<N, D1, S1>,
    kernel: Vector<N, D2, S2>,
) -> VectorN<N, DimDiff<DimSum<D1, U1>, D2>>
where
    N: Real,
    D1: DimAdd<U1>,
    D2: Dim,
    DimSum<D1, U1>: DimSub<D2>,
    S1: Storage<N, D1>,
    S2: Storage<N, D2>,
    DefaultAllocator: Allocator<N, DimDiff<DimSum<D1, U1>, D2>>,
{
    let vec = vector.len();
    let ker = kernel.len();

    if ker == 0 || ker > vec {
        panic!("convolve_valid expects `vector.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
    }

    let result_len = vector.data.shape().0.add(U1).sub(kernel.data.shape().0);
    let mut conv = VectorN::zeros_generic(result_len, U1);

    for i in 0..(vec - ker + 1) {
        for j in 0..ker {
            conv[i] += vector[i + j] * kernel[ker - j - 1];
        }
    }
    conv
}

/// Returns the convolution of the vector and a kernel
/// The output convolution is the same size as vector, centered with respect to the ‘full’ output.
/// # Arguments
///
/// * `vector` - A Vector with size > 0
/// * `kernel` - A Vector with size > 0
///
/// # Errors
/// Inputs must statisfy `vector.len() >= kernel.len() > 0`.
pub fn convolve_same<N, D1, D2, S1, S2>(
    vector: Vector<N, D1, S1>,
    kernel: Vector<N, D2, S2>,
) -> VectorN<N, DimMaximum<D1, D2>>
where
    N: Real,
    D1: DimMax<D2>,
    D2: DimMax<D1, Output = DimMaximum<D1, D2>>,
    S1: Storage<N, D1>,
    S2: Storage<N, D2>,
    DefaultAllocator: Allocator<N, DimMaximum<D1, D2>>,
{
    let vec = vector.len();
    let ker = kernel.len();

    if ker == 0 || ker > vec {
        panic!("convolve_same expects `vector.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
    }

    let result_len = vector.data.shape().0.max(kernel.data.shape().0);
    let mut conv = VectorN::zeros_generic(result_len, U1);

    for i in 0..vec {
        for j in 0..ker {
            let val = if i + j < 1 || i + j >= vec + 1 {
                zero::<N>()
            } else {
                vector[i + j - 1]
            };
            conv[i] += val * kernel[ker - j - 1];
        }
    }
    conv
}
