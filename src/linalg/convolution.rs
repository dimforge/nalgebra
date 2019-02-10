use storage::Storage;
use {zero, DVector, Dim, Dynamic, Matrix, Real, VecStorage, Vector, U1};
use std::cmp;

///
/// The output is the full discrete linear convolution of the inputs
/// 
pub fn convolve_full<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_full(kernel, vector);
    }

    let newlen = vec + ker - 1;

    let mut conv = DVector::<R>::zeros(newlen);

    for i in 0..newlen {
        let u_i = if i > ker { i - ker } else { 0 };
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

///
/// The output convolution consists only of those elements that do not rely on the zero-padding. 
/// 
pub fn convolve_valid<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_valid(kernel, vector);
    }

    let newlen = vec - ker + 1;

    let mut conv = DVector::<R>::zeros(newlen);

    for i in 0..newlen {
        for j in 0..ker {
            conv[i] += vector[i + j] * kernel[ker - j - 1];
        }
    }
    conv
}

///
/// The output convolution is the same size as vector, centered with respect to the ‘full’ output.
/// 
pub fn convolve_same<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();

    if vec == 0 || ker == 0 {
        panic!("Convolve's inputs must not be 0-sized. ");
    }

    if ker > vec {
        return convolve_same(kernel, vector);
    }

    let mut conv = DVector::<R>::zeros(vec);

    for i in 0..vec {
        for j in 0..ker {
            let val = if i + j < 1 || i + j >= vec + 1 {
                zero::<R>()
            } else {
                vector[i + j - 1]
            };
            conv[i] += val * kernel[ker - j - 1];
        }
    }
    conv
}