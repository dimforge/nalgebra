use storage::Storage;
use {zero, DVector, Dim, Dynamic, Matrix, Real, VecStorage, Vector, U1, Add};
use std::cmp;

impl<N: Real, D1: Dim, S1: Storage<N,D1>> Vector<N,D1,S1>{

    /// Returns the convolution of the vector and a kernel
    ///
    /// # Arguments
    ///
    /// * `self` - A DVector with size D > 0
    /// * `kernel` - A DVector with size D > 0
    /// 
    /// # Note:
    ///     This function is commutative. If D_kernel > D_vector, 
    ///     they will swap their roles as in 
    ///     (self, kernel) = (kernel,self)
    ///
    /// # Example
    ///
    /// ```
    /// 
    /// ```
    pub fn convolve_full<D2: Dim, S2: Storage<N, D2>>(&self, kernel: Vector<N, D2, S2>) -> Vector<N,Add<D1,D2>,Add<S1,S2>>
    {
        let vec = self.len();
        let ker = kernel.len();

        // if vec == 0 || ker == 0 {
        //     panic!("Convolve's inputs must not be 0-sized. ");
        // }

        // if ker > vec {
        //     return kernel::convolve_full(vector);
        // }

        let newlen = vec + ker - 1;
        let mut conv = DVector::<N>::zeros(newlen);

        for i in 0..newlen {
            let u_i = if i > ker { i - ker } else { 0 };
            let u_f = cmp::min(i, vec - 1);

            if u_i == u_f {
                conv[i] += self[u_i] * kernel[(i - u_i)];
            } else {
                for u in u_i..(u_f + 1) {
                    if i - u < ker {
                        conv[i] += self[u] * kernel[(i - u)];
                    }
                }
            }
        }
        // conv
    }
}
///
/// The output is the full discrete linear convolution of the inputs
/// 


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

