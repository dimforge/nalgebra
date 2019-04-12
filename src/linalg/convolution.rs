use {
    std::cmp,
    crate::{
        base::{
            allocator::Allocator,
            default_allocator::DefaultAllocator,
            dimension::{Dim, DimAdd, DimDiff, DimSub, DimSum}
        },
        storage::Storage,
        zero, RealField, Vector, VectorN, U1
    }
};

impl<N: RealField, D1: Dim, S1: Storage<N, D1>> Vector<N, D1, S1> {
    /// Returns the convolution of the target vector and a kernel.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A Vector with size > 0
    ///
    /// # Errors
    /// Inputs must satisfy `vector.len() >= kernel.len() > 0`.
    ///
    pub fn convolve_full<D2, S2>(
        &self,
        kernel: Vector<N, D2, S2>,
    ) -> VectorN<N, DimDiff<DimSum<D1, D2>, U1>>
    where
        D1: DimAdd<D2>,
        D2: DimAdd<D1, Output = DimSum<D1, D2>>,
        DimSum<D1, D2>: DimSub<U1>,
        S2: Storage<N, D2>,
        DefaultAllocator: Allocator<N, DimDiff<DimSum<D1, D2>, U1>>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!("convolve_full expects `self.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
        }

        let result_len = self.data.shape().0.add(kernel.data.shape().0).sub(U1);
        let mut conv = VectorN::zeros_generic(result_len, U1);

        for i in 0..(vec + ker - 1) {
            let u_i = if i > vec { i - ker } else { 0 };
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
        conv
    }
    /// Returns the convolution of the target vector and a kernel.
    ///
    /// The output convolution consists only of those elements that do not rely on the zero-padding.
    /// # Arguments
    ///
    /// * `kernel` - A Vector with size > 0
    ///
    ///
    /// # Errors
    /// Inputs must satisfy `self.len() >= kernel.len() > 0`.
    ///
    pub fn convolve_valid<D2, S2>(&self, kernel: Vector<N, D2, S2>) -> VectorN<N, DimDiff<DimSum<D1, U1>, D2>>
    where
        D1: DimAdd<U1>,
        D2: Dim,
        DimSum<D1, U1>: DimSub<D2>,
        S2: Storage<N, D2>,
        DefaultAllocator: Allocator<N, DimDiff<DimSum<D1, U1>, D2>>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!("convolve_valid expects `self.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
        }

        let result_len = self.data.shape().0.add(U1).sub(kernel.data.shape().0);
        let mut conv = VectorN::zeros_generic(result_len, U1);

        for i in 0..(vec - ker + 1) {
            for j in 0..ker {
                conv[i] += self[i + j] * kernel[ker - j - 1];
            }
        }
        conv
    }

    /// Returns the convolution of the target vector and a kernel.
    ///
    /// The output convolution is the same size as vector, centered with respect to the ‘full’ output.
    /// # Arguments
    ///
    /// * `kernel` - A Vector with size > 0
    ///
    /// # Errors
    /// Inputs must satisfy `self.len() >= kernel.len() > 0`.
    pub fn convolve_same<D2, S2>(&self, kernel: Vector<N, D2, S2>) -> VectorN<N, D1>
    where
        D2: Dim,
        S2: Storage<N, D2>,
        DefaultAllocator: Allocator<N, D1>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!("convolve_same expects `self.len() >= kernel.len() > 0`, received {} and {} respectively.",vec,ker);
        }

        let mut conv = VectorN::zeros_generic(self.data.shape().0, U1);

        for i in 0..vec {
            for j in 0..ker {
                let val = if i + j < 1 || i + j >= vec + 1 {
                    zero::<N>()
                } else {
                    self[i + j - 1]
                };
                conv[i] += val * kernel[ker - j - 1];
            }
        }

        conv
    }
}
