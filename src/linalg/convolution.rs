use std::cmp;

use crate::base::allocator::Allocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Const, Dim, DimAdd, DimDiff, DimSub, DimSum};
use crate::storage::Storage;
use crate::{OVector, RealField, U1, Vector, zero};

impl<T: RealField, D1: Dim, S1: Storage<T, D1>> Vector<T, D1, S1> {
    /// Returns the full convolution of the target vector and a kernel.
    ///
    /// # What is Convolution?
    ///
    /// Convolution is a mathematical operation that combines two sequences to produce a third sequence.
    /// Think of it as "sliding" the kernel over the input vector, multiplying overlapping elements,
    /// and summing the results at each position.
    ///
    /// In signal processing, convolution is used to apply filters to signals (e.g., smoothing, noise reduction).
    /// In image processing, it's used for operations like blurring, edge detection, and sharpening.
    ///
    /// # Full Convolution
    ///
    /// The "full" convolution returns all positions where the kernel and vector overlap, even partially.
    /// This produces an output of length `vector.len() + kernel.len() - 1`.
    ///
    /// The output is larger than the input because it includes positions where the kernel only
    /// partially overlaps with the vector (edges are implicitly zero-padded).
    ///
    /// # Arguments
    ///
    /// * `kernel` - A vector representing the convolution kernel or filter. Must have length > 0
    ///   and length <= the input vector length.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() >= kernel.len() > 0` is not satisfied.
    ///
    /// # Examples
    ///
    /// ## Basic Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    ///
    /// let signal = Vector3::new(1.0, 2.0, 3.0);
    /// let kernel = Vector3::new(0.5, 1.0, 0.5);
    ///
    /// let result = signal.convolve_full(kernel);
    ///
    /// // Full convolution produces a vector of length 3 + 3 - 1 = 5
    /// assert_eq!(result.len(), 5);
    /// // Result: [0.5, 2.0, 4.0, 4.0, 1.5]
    /// ```
    ///
    /// ## Signal Smoothing (Moving Average)
    ///
    /// ```
    /// use nalgebra::Vector5;
    ///
    /// // A noisy signal
    /// let signal = Vector5::new(1.0, 5.0, 2.0, 8.0, 3.0);
    ///
    /// // A simple smoothing kernel (3-point moving average)
    /// let smooth_kernel = Vector5::from_element(1.0 / 3.0)
    ///     .rows_generic(0, nalgebra::Const::<3>)
    ///     .into_owned();
    ///
    /// let smoothed = signal.convolve_full(smooth_kernel);
    ///
    /// // The smoothed signal has reduced high-frequency noise
    /// assert_eq!(smoothed.len(), 7);
    /// ```
    ///
    /// ## Edge Detection (Discrete Derivative)
    ///
    /// ```
    /// use nalgebra::{Vector4, Vector2};
    ///
    /// // A step signal (like an edge in an image)
    /// let signal = Vector4::new(1.0, 1.0, 5.0, 5.0);
    ///
    /// // A simple difference kernel for edge detection
    /// let edge_kernel = Vector2::new(-1.0, 1.0);
    ///
    /// let edges = signal.convolve_full(edge_kernel);
    ///
    /// // The convolution highlights where the signal changes
    /// assert_eq!(edges.len(), 5);
    /// // The result will show a peak where the edge occurs
    /// ```
    ///
    /// # See Also
    ///
    /// * [`convolve_valid`](Self::convolve_valid) - Returns only the portion where kernel fully overlaps with the vector
    /// * [`convolve_same`](Self::convolve_same) - Returns convolution with the same size as input vector
    ///
    pub fn convolve_full<D2, S2>(
        &self,
        kernel: Vector<T, D2, S2>,
    ) -> OVector<T, DimDiff<DimSum<D1, D2>, U1>>
    where
        D1: DimAdd<D2>,
        D2: DimAdd<D1, Output = DimSum<D1, D2>>,
        DimSum<D1, D2>: DimSub<U1>,
        S2: Storage<T, D2>,
        DefaultAllocator: Allocator<DimDiff<DimSum<D1, D2>, U1>>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!(
                "convolve_full expects `self.len() >= kernel.len() > 0`, received {vec} and {ker} respectively."
            );
        }

        let result_len = self
            .data
            .shape()
            .0
            .add(kernel.shape_generic().0)
            .sub(Const::<1>);
        let mut conv = OVector::zeros_generic(result_len, Const::<1>);

        for i in 0..(vec + ker - 1) {
            let u_i = if i > vec { i - ker } else { 0 };
            let u_f = cmp::min(i, vec - 1);

            if u_i == u_f {
                conv[i] += self[u_i].clone() * kernel[i - u_i].clone();
            } else {
                for u in u_i..(u_f + 1) {
                    if i - u < ker {
                        conv[i] += self[u].clone() * kernel[i - u].clone();
                    }
                }
            }
        }
        conv
    }
    /// Returns the valid convolution of the target vector and a kernel.
    ///
    /// # What is Convolution?
    ///
    /// Convolution is a mathematical operation that combines two sequences to produce a third sequence.
    /// Think of it as "sliding" the kernel over the input vector, multiplying overlapping elements,
    /// and summing the results at each position.
    ///
    /// In signal processing, convolution is used to apply filters to signals (e.g., smoothing, noise reduction).
    /// In image processing, it's used for operations like blurring, edge detection, and sharpening.
    ///
    /// # Valid Convolution
    ///
    /// The "valid" convolution returns only positions where the kernel completely overlaps with the vector,
    /// without requiring any zero-padding. This produces an output of length `vector.len() - kernel.len() + 1`.
    ///
    /// This mode is useful when you want to avoid edge effects and only get results based on
    /// actual data points (no implicit zeros). The output is smaller than the input.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A vector representing the convolution kernel or filter. Must have length > 0
    ///   and length <= the input vector length.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() >= kernel.len() > 0` is not satisfied.
    ///
    /// # Examples
    ///
    /// ## Basic Example
    ///
    /// ```
    /// use nalgebra::{Vector5, Vector3};
    ///
    /// let signal = Vector5::new(1.0, 2.0, 3.0, 4.0, 5.0);
    /// let kernel = Vector3::new(0.25, 0.5, 0.25);
    ///
    /// let result = signal.convolve_valid(kernel);
    ///
    /// // Valid convolution produces a vector of length 5 - 3 + 1 = 3
    /// assert_eq!(result.len(), 3);
    /// // Each output point uses only actual data from the signal
    /// ```
    ///
    /// ## Signal Smoothing Without Edge Effects
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // Original signal
    /// let signal = Vector6::new(10.0, 15.0, 13.0, 17.0, 14.0, 12.0);
    ///
    /// // Smoothing kernel (weighted average)
    /// let kernel = Vector3::new(0.25, 0.5, 0.25);
    ///
    /// let smoothed = signal.convolve_valid(kernel);
    ///
    /// // Result has length 6 - 3 + 1 = 4
    /// // Only includes smoothed values with full kernel support
    /// assert_eq!(smoothed.len(), 4);
    /// ```
    ///
    /// ## Pattern Matching in Signals
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // A signal containing a pattern
    /// let signal = Vector6::new(0.0, 1.0, 2.0, 1.0, 0.0, 1.0);
    ///
    /// // Looking for a peak pattern
    /// let pattern = Vector3::new(1.0, 2.0, 1.0);
    ///
    /// let correlation = signal.convolve_valid(pattern);
    ///
    /// // The result shows where the pattern matches best
    /// assert_eq!(correlation.len(), 4);
    /// // Will have highest value where pattern occurs (position 1)
    /// ```
    ///
    /// ## Feature Detection
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // A signal with features (like pixel values in an image row)
    /// let signal = Vector6::new(1.0, 1.0, 5.0, 5.0, 1.0, 1.0);
    ///
    /// // Gradient kernel to detect rising edges
    /// let gradient_kernel = Vector3::new(-1.0, 0.0, 1.0);
    ///
    /// let gradients = signal.convolve_valid(gradient_kernel);
    ///
    /// // Detects transitions in the signal
    /// assert_eq!(gradients.len(), 4);
    /// // Large positive values indicate rising edges
    /// // Large negative values indicate falling edges
    /// ```
    ///
    /// # See Also
    ///
    /// * [`convolve_full`](Self::convolve_full) - Returns all positions where kernel overlaps (includes zero-padding)
    /// * [`convolve_same`](Self::convolve_same) - Returns convolution with the same size as input vector
    ///
    pub fn convolve_valid<D2, S2>(
        &self,
        kernel: Vector<T, D2, S2>,
    ) -> OVector<T, DimDiff<DimSum<D1, U1>, D2>>
    where
        D1: DimAdd<U1>,
        D2: Dim,
        DimSum<D1, U1>: DimSub<D2>,
        S2: Storage<T, D2>,
        DefaultAllocator: Allocator<DimDiff<DimSum<D1, U1>, D2>>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!(
                "convolve_valid expects `self.len() >= kernel.len() > 0`, received {vec} and {ker} respectively."
            );
        }

        let result_len = self
            .data
            .shape()
            .0
            .add(Const::<1>)
            .sub(kernel.shape_generic().0);
        let mut conv = OVector::zeros_generic(result_len, Const::<1>);

        for i in 0..(vec - ker + 1) {
            for j in 0..ker {
                conv[i] += self[i + j].clone() * kernel[ker - j - 1].clone();
            }
        }
        conv
    }

    /// Returns the same-size convolution of the target vector and a kernel.
    ///
    /// # What is Convolution?
    ///
    /// Convolution is a mathematical operation that combines two sequences to produce a third sequence.
    /// Think of it as "sliding" the kernel over the input vector, multiplying overlapping elements,
    /// and summing the results at each position.
    ///
    /// In signal processing, convolution is used to apply filters to signals (e.g., smoothing, noise reduction).
    /// In image processing, it's used for operations like blurring, edge detection, and sharpening.
    ///
    /// # Same Convolution
    ///
    /// The "same" convolution returns an output with the same length as the input vector.
    /// It's computed by taking the central portion of the full convolution, centered with
    /// respect to the input.
    ///
    /// This mode is most convenient when you want to apply a filter while preserving the
    /// original signal dimensions. Edge positions use zero-padding where the kernel extends
    /// beyond the input.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A vector representing the convolution kernel or filter. Must have length > 0
    ///   and length <= the input vector length.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() >= kernel.len() > 0` is not satisfied.
    ///
    /// # Examples
    ///
    /// ## Basic Example
    ///
    /// ```
    /// use nalgebra::{Vector5, Vector3};
    ///
    /// let signal = Vector5::new(1.0, 2.0, 3.0, 4.0, 5.0);
    /// let kernel = Vector3::new(0.25, 0.5, 0.25);
    ///
    /// let result = signal.convolve_same(kernel);
    ///
    /// // Same convolution produces output with same length as input
    /// assert_eq!(result.len(), signal.len());
    /// assert_eq!(result.len(), 5);
    /// ```
    ///
    /// ## Signal Smoothing (Low-Pass Filter)
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // A noisy signal
    /// let noisy = Vector6::new(1.0, 10.0, 2.0, 9.0, 3.0, 8.0);
    ///
    /// // Gaussian-like smoothing kernel
    /// let smooth = Vector3::new(0.25, 0.5, 0.25);
    ///
    /// let filtered = noisy.convolve_same(smooth);
    ///
    /// // Output has same length, with noise reduced
    /// assert_eq!(filtered.len(), 6);
    /// // The filtered signal has smoother transitions
    /// ```
    ///
    /// ## Moving Average Filter
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // Time series data (e.g., daily temperatures)
    /// let data = Vector6::new(20.0, 22.0, 25.0, 23.0, 21.0, 24.0);
    ///
    /// // 3-day moving average
    /// let avg_kernel = Vector3::from_element(1.0 / 3.0);
    ///
    /// let trend = data.convolve_same(avg_kernel);
    ///
    /// // Same length as input, shows smoothed trend
    /// assert_eq!(trend.len(), 6);
    /// ```
    ///
    /// ## Sharpening Filter
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // A blurred signal (like blurred image pixels)
    /// let blurred = Vector6::new(2.0, 2.5, 3.0, 3.5, 4.0, 4.5);
    ///
    /// // Sharpening kernel (enhances edges)
    /// // Center weight > 1, neighbors negative
    /// let sharpen = Vector3::new(-0.5, 2.0, -0.5);
    ///
    /// let sharpened = blurred.convolve_same(sharpen);
    ///
    /// // Same size output with enhanced transitions
    /// assert_eq!(sharpened.len(), 6);
    /// ```
    ///
    /// ## High-Pass Filter (Edge Enhancement)
    ///
    /// ```
    /// use nalgebra::{Vector6, Vector3};
    ///
    /// // A signal with both slow and fast changes
    /// let signal = Vector6::new(1.0, 1.1, 1.2, 3.0, 3.1, 3.2);
    ///
    /// // High-pass kernel to detect rapid changes
    /// let highpass = Vector3::new(-1.0, 2.0, -1.0);
    ///
    /// let edges = signal.convolve_same(highpass);
    ///
    /// // Highlights rapid transitions while suppressing slow trends
    /// assert_eq!(edges.len(), 6);
    /// // Will show a peak at the step change (index 3)
    /// ```
    ///
    /// ## Image Row Processing
    ///
    /// ```
    /// use nalgebra::{Vector5, Vector3};
    ///
    /// // A row of pixels from an image (grayscale values 0-255)
    /// let pixel_row = Vector5::new(100.0, 120.0, 110.0, 130.0, 125.0);
    ///
    /// // Gaussian blur kernel
    /// let blur = Vector3::new(0.27, 0.46, 0.27);
    ///
    /// let blurred_row = pixel_row.convolve_same(blur);
    ///
    /// // Blurred pixels, same dimensions for easy replacement
    /// assert_eq!(blurred_row.len(), 5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`convolve_full`](Self::convolve_full) - Returns all positions where kernel overlaps (larger output)
    /// * [`convolve_valid`](Self::convolve_valid) - Returns only positions with full kernel overlap (smaller output)
    ///
    #[must_use]
    pub fn convolve_same<D2, S2>(&self, kernel: Vector<T, D2, S2>) -> OVector<T, D1>
    where
        D2: Dim,
        S2: Storage<T, D2>,
        DefaultAllocator: Allocator<D1>,
    {
        let vec = self.len();
        let ker = kernel.len();

        if ker == 0 || ker > vec {
            panic!(
                "convolve_same expects `self.len() >= kernel.len() > 0`, received {vec} and {ker} respectively."
            );
        }

        let mut conv = OVector::zeros_generic(self.shape_generic().0, Const::<1>);

        for i in 0..vec {
            for j in 0..ker {
                let val = if i + j < 1 || i + j >= vec + 1 {
                    zero::<T>()
                } else {
                    self[i + j - 1].clone()
                };
                conv[i] += val * kernel[ker - j - 1].clone();
            }
        }

        conv
    }
}
