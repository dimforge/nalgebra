use base::allocator::Allocator;
use base::default_allocator::DefaultAllocator;
use base::dimension::{DimAdd, DimDiff, DimMax, DimMaximum, DimName, DimSub, DimSum,Dim};
use std::cmp;
use storage::Storage;
use {zero, Real, Vector, VectorN, U1};

/// Returns the convolution of the vector and a kernel
///
/// # Arguments
///
/// * `vector` - A Vector with size > 0
/// * `kernel` - A Vector with size > 0
///
///  This function is commutative. If kernel > vector,
///  they will swap their roles as in
///  (self, kernel) = (kernel,self)
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
    DimSum<D1, D2>: Dim,
    DefaultAllocator: Allocator<N, DimDiff<DimSum<D1, D2>, U1>>,
{
    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_full(kernel, vector);
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
/// This function is commutative. If kernel > vector,
/// they will swap their roles as in
/// (self, kernel) = (kernel,self)
///
pub fn convolve_valid<N, D1, D2, S1, S2>(
    vector: Vector<N, D1, S1>,
    kernel: Vector<N, D2, S2>,
) -> VectorN<N, DimSum<DimDiff<D1, D2>, U1>>
where
    N: Real,
    D1: DimSub<D2>,
    D2: DimSub<D1, Output = DimDiff<D1, D2>>,
    DimDiff<D1, D2>: DimAdd<U1>,
    S1: Storage<N, D1>,
    S2: Storage<N, D2>,
    DimDiff<D1, D2>: DimName,
    DefaultAllocator: Allocator<N, DimSum<DimDiff<D1, D2>, U1>>
{

    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_valid(kernel, vector);
    }
    let result_len = vector.data.shape().0.sub(kernel.data.shape().0).add(U1);
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
/// This function is commutative. If kernel > vector,
/// they will swap their roles as in
/// (self, kernel) = (kernel,self)
///
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
    DimMaximum<D1, D2>: Dim,
    DefaultAllocator: Allocator<N, DimMaximum<D1, D2>>,
{
    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_same(kernel, vector);
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
