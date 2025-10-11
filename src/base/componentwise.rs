// Non-conventional component-wise operators.

use num::{Signed, Zero};
use std::ops::{Add, Mul};

use simba::scalar::{ClosedDivAssign, ClosedMulAssign};
use simba::simd::SimdPartialOrd;

use crate::ClosedAddAssign;
use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixSum, OMatrix, Scalar};

/// The type of the result of a matrix component-wise operation.
pub type MatrixComponentOp<T, R1, C1, R2, C2> = MatrixSum<T, R1, C1, R2, C2>;

impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Computes the component-wise absolute value.
    ///
    /// This method applies the absolute value operation to each element of the matrix or vector
    /// independently, returning a new matrix/vector with all values converted to their
    /// non-negative equivalents. Negative values become positive, and positive values remain
    /// unchanged.
    ///
    /// # Mathematical Definition
    ///
    /// For each element `a[i,j]` in the matrix, the result contains `|a[i,j]|`.
    ///
    /// # Examples
    ///
    /// ## Basic usage with a matrix
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(0.0, 1.0,
    ///                      -2.0, -3.0);
    /// assert_eq!(a.abs(), Matrix2::new(0.0, 1.0, 2.0, 3.0))
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.0, 2.0, -3.5);
    /// let result = v.abs();
    /// assert_eq!(result, Vector3::new(1.0, 2.0, 3.5));
    /// ```
    ///
    /// ## Edge case: All negative values
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(-1.0, -2.0, -3.0, -4.0);
    /// let result = v.abs();
    /// assert_eq!(result, Vector4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    ///
    /// ## Practical use case: Distance calculation
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Calculate element-wise differences between two points
    /// let point_a = Vector3::new(1.0, 2.0, 3.0);
    /// let point_b = Vector3::new(4.0, 1.0, 5.0);
    /// let differences = point_a - point_b;
    /// let absolute_differences = differences.abs();
    /// assert_eq!(absolute_differences, Vector3::new(3.0, 1.0, 2.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`component_mul`](Self::component_mul): Component-wise multiplication
    /// - [`inf`](Self::inf): Component-wise minimum between two matrices
    /// - [`sup`](Self::sup): Component-wise maximum between two matrices
    #[inline]
    #[must_use]
    pub fn abs(&self) -> OMatrix<T, R, C>
    where
        T: Signed,
        DefaultAllocator: Allocator<R, C>,
    {
        let mut res = self.clone_owned();

        for e in res.iter_mut() {
            *e = e.abs();
        }

        res
    }

    // TODO: add other operators like component_ln, component_pow, etc. ?
}

macro_rules! component_binop_impl(
    ($($binop: ident, $binop_mut: ident, $binop_assign: ident, $cmpy: ident, $Trait: ident . $op: ident . $op_assign: ident, $desc:expr_2021, $desc_cmpy:expr_2021, $desc_mut:expr_2021);* $(;)*) => {$(
        #[doc = $desc]
        #[inline]
        #[must_use]
        pub fn $binop<R2, C2, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> MatrixComponentOp<T, R1, C1, R2, C2>
            where T: $Trait,
                  R2: Dim, C2: Dim,
                  SB: Storage<T, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
                  ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

            assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
            let mut res = self.clone_owned_sum();

            for j in 0 .. res.ncols() {
                for i in 0 .. res.nrows() {
                    unsafe {
                        res.get_unchecked_mut((i, j)).$op_assign(rhs.get_unchecked((i, j)).clone());
                    }
                }
            }

            res
        }

        // componentwise binop plus Y.
        #[doc = $desc_cmpy]
        #[inline]
        pub fn $cmpy<R2, C2, SB, R3, C3, SC>(&mut self, alpha: T, a: &Matrix<T, R2, C2, SB>, b: &Matrix<T, R3, C3, SC>, beta: T)
            where T: $Trait + Zero + Mul<T, Output = T> + Add<T, Output = T>,
                  R2: Dim, C2: Dim,
                  R3: Dim, C3: Dim,
                  SA: StorageMut<T, R1, C1>,
                  SB: Storage<T, R2, C2>,
                  SC: Storage<T, R3, C3>,
                  ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> +
                                   SameNumberOfRows<R1, R3> + SameNumberOfColumns<C1, C3> {
            assert_eq!(self.shape(), a.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
            assert_eq!(self.shape(), b.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

            if beta.is_zero() {
                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe {
                            let res = alpha.clone() * a.get_unchecked((i, j)).clone().$op(b.get_unchecked((i, j)).clone());
                            *self.get_unchecked_mut((i, j)) = res;
                        }
                    }
                }
            }
            else {
                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe {
                            let res = alpha.clone() * a.get_unchecked((i, j)).clone().$op(b.get_unchecked((i, j)).clone());
                            *self.get_unchecked_mut((i, j)) = beta.clone() * self.get_unchecked((i, j)).clone() + res;
                        }
                    }
                }
            }
        }

        #[doc = $desc_mut]
        #[inline]
        pub fn $binop_assign<R2, C2, SB>(&mut self, rhs: &Matrix<T, R2, C2, SB>)
            where T: $Trait,
                  R2: Dim,
                  C2: Dim,
                  SA: StorageMut<T, R1, C1>,
                  SB: Storage<T, R2, C2>,
                  ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

            assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

            for j in 0 .. self.ncols() {
                for i in 0 .. self.nrows() {
                    unsafe {
                        self.get_unchecked_mut((i, j)).$op_assign(rhs.get_unchecked((i, j)).clone());
                    }
                }
            }
        }

        #[doc = $desc_mut]
        #[inline]
        #[deprecated(note = "This is renamed using the `_assign` suffix instead of the `_mut` suffix.")]
        pub fn $binop_mut<R2, C2, SB>(&mut self, rhs: &Matrix<T, R2, C2, SB>)
            where T: $Trait,
                  R2: Dim,
                  C2: Dim,
                  SA: StorageMut<T, R1, C1>,
                  SB: Storage<T, R2, C2>,
                  ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
            self.$binop_assign(rhs)
        }
    )*}
);

/// # Componentwise operations
impl<T: Scalar, R1: Dim, C1: Dim, SA: Storage<T, R1, C1>> Matrix<T, R1, C1, SA> {
    component_binop_impl!(
        component_mul, component_mul_mut, component_mul_assign, cmpy, ClosedMulAssign.mul.mul_assign,
        r"
        Componentwise matrix or vector multiplication.

        This operation multiplies each element of `self` with the corresponding element
        of `rhs`, returning a new matrix or vector. This is also known as the Hadamard
        product or element-wise multiplication. Note that this is different from matrix
        multiplication, which involves dot products of rows and columns.

        # Arguments

        * `rhs` - The right-hand side matrix or vector. Must have the same dimensions as `self`.

        # Panics

        Panics if the matrices have different dimensions.

        # Examples

        ## Basic matrix multiplication

        ```
        # use nalgebra::Matrix2;
        let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = Matrix2::new(0.0, 5.0, 12.0, 21.0);

        assert_eq!(a.component_mul(&b), expected);
        ```

        ## Vector element-wise multiplication

        ```
        # use nalgebra::Vector3;
        let u = Vector3::new(1.0, 2.0, 3.0);
        let v = Vector3::new(4.0, 5.0, 6.0);
        let result = u.component_mul(&v);
        assert_eq!(result, Vector3::new(4.0, 10.0, 18.0));
        ```

        ## Practical use case: Scaling different dimensions

        ```
        # use nalgebra::Vector3;
        // Scale an object differently along each axis
        let position = Vector3::new(10.0, 20.0, 30.0);
        let scale_factors = Vector3::new(2.0, 0.5, 1.5);
        let scaled_position = position.component_mul(&scale_factors);
        assert_eq!(scaled_position, Vector3::new(20.0, 10.0, 45.0));
        ```

        ## Masking values

        ```
        # use nalgebra::Vector4;
        // Use 0s and 1s to selectively keep values
        let values = Vector4::new(10.0, 20.0, 30.0, 40.0);
        let mask = Vector4::new(1.0, 0.0, 1.0, 0.0);
        let masked = values.component_mul(&mask);
        assert_eq!(masked, Vector4::new(10.0, 0.0, 30.0, 0.0));
        ```

        # See Also

        - [`component_mul_assign`](Self::component_mul_assign): In-place version of this operation
        - [`cmpy`](Self::cmpy): Fused multiply-add operation: `self = alpha * a * b + beta * self`
        - [`component_div`](Self::component_div): Component-wise division
        ",
        r"
        Computes componentwise `self[i] = alpha * a[i] * b[i] + beta * self[i]`.

        This is a fused multiply-add operation that combines component-wise multiplication
        with scaling and accumulation. It's more efficient than performing separate
        operations and is particularly useful in numerical computations and linear algebra
        algorithms.

        The operation is performed in-place, modifying `self`.

        # Arguments

        * `alpha` - Scalar multiplier for the component-wise product of `a` and `b`
        * `a` - First matrix/vector to multiply component-wise
        * `b` - Second matrix/vector to multiply component-wise
        * `beta` - Scalar multiplier for the current value of `self`

        # Mathematical Definition

        For each element index `i`: `self[i] = alpha * a[i] * b[i] + beta * self[i]`

        When `beta` is zero, the operation simplifies to: `self[i] = alpha * a[i] * b[i]`
        (without reading the current value of `self[i]`, which can be more efficient).

        # Panics

        Panics if `self`, `a`, and `b` don't have the same dimensions.

        # Examples

        ## Basic usage

        ```
        # use nalgebra::Matrix2;
        let mut m = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = (a.component_mul(&b) * 5.0) + m * 10.0;

        m.cmpy(5.0, &a, &b, 10.0);
        assert_eq!(m, expected);
        ```

        ## With vectors

        ```
        # use nalgebra::Vector3;
        let mut result = Vector3::new(1.0, 2.0, 3.0);
        let a = Vector3::new(2.0, 3.0, 4.0);
        let b = Vector3::new(5.0, 6.0, 7.0);

        // result = 2.0 * (a .* b) + 3.0 * result
        result.cmpy(2.0, &a, &b, 3.0);
        assert_eq!(result, Vector3::new(23.0, 42.0, 65.0));
        ```

        ## Beta is zero (doesn't use current self values)

        ```
        # use nalgebra::Vector3;
        let mut result = Vector3::new(100.0, 200.0, 300.0);  // Initial values don't matter
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);

        // When beta = 0, self is overwritten
        result.cmpy(2.0, &a, &b, 0.0);
        assert_eq!(result, Vector3::new(8.0, 20.0, 36.0));
        ```

        ## Accumulating weighted products

        ```
        # use nalgebra::Vector2;
        let mut accumulator = Vector2::new(0.0, 0.0);
        let weights = Vector2::new(0.5, 0.8);
        let values = Vector2::new(10.0, 20.0);

        // Accumulate: accumulator = 1.0 * (weights .* values) + 1.0 * accumulator
        accumulator.cmpy(1.0, &weights, &values, 1.0);
        assert_eq!(accumulator, Vector2::new(5.0, 16.0));

        // Accumulate again
        accumulator.cmpy(1.0, &weights, &values, 1.0);
        assert_eq!(accumulator, Vector2::new(10.0, 32.0));
        ```

        # See Also

        - [`component_mul`](Self::component_mul): Component-wise multiplication
        - [`component_mul_assign`](Self::component_mul_assign): In-place component-wise multiplication
        - [`cdpy`](Self::cdpy): Similar operation but with component-wise division
        ",
        r"
        In-place componentwise matrix or vector multiplication.

        This operation multiplies each element of `self` with the corresponding element
        of `rhs`, modifying `self` in place. This is the in-place version of
        [`component_mul`](Self::component_mul), which is also known as the Hadamard
        product or element-wise multiplication.

        This method is more memory-efficient than `component_mul` as it doesn't allocate
        a new matrix/vector for the result.

        # Arguments

        * `rhs` - The right-hand side matrix or vector. Must have the same dimensions as `self`.

        # Panics

        Panics if the matrices have different dimensions.

        # Examples

        ## Basic matrix multiplication

        ```
        # use nalgebra::Matrix2;
        let mut a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = Matrix2::new(0.0, 5.0, 12.0, 21.0);

        a.component_mul_assign(&b);

        assert_eq!(a, expected);
        ```

        ## Vector element-wise multiplication

        ```
        # use nalgebra::Vector3;
        let mut u = Vector3::new(1.0, 2.0, 3.0);
        let v = Vector3::new(4.0, 5.0, 6.0);
        u.component_mul_assign(&v);
        assert_eq!(u, Vector3::new(4.0, 10.0, 18.0));
        ```

        ## Repeatedly scaling by the same factor

        ```
        # use nalgebra::Vector2;
        let mut position = Vector2::new(10.0, 20.0);
        let decay = Vector2::new(0.9, 0.95);

        // Apply decay multiple times
        position.component_mul_assign(&decay);
        assert_eq!(position, Vector2::new(9.0, 19.0));

        position.component_mul_assign(&decay);
        assert_eq!(position, Vector2::new(8.1, 18.05));
        ```

        ## Practical use case: Applying a mask

        ```
        # use nalgebra::Vector4;
        let mut data = Vector4::new(1.5, 2.5, 3.5, 4.5);
        let enable_mask = Vector4::new(1.0, 1.0, 0.0, 1.0);

        // Disable third element
        data.component_mul_assign(&enable_mask);
        assert_eq!(data, Vector4::new(1.5, 2.5, 0.0, 4.5));
        ```

        # See Also

        - [`component_mul`](Self::component_mul): Non-mutating version that returns a new matrix/vector
        - [`cmpy`](Self::cmpy): Fused multiply-add: `self = alpha * a * b + beta * self`
        - [`component_div_assign`](Self::component_div_assign): In-place component-wise division
        ";
        component_div, component_div_mut, component_div_assign, cdpy, ClosedDivAssign.div.div_assign,
        r"
        Componentwise matrix or vector division.

        This operation divides each element of `self` by the corresponding element of `rhs`,
        returning a new matrix or vector. This is element-wise division, where each element
        is divided independently.

        # Arguments

        * `rhs` - The right-hand side matrix or vector (divisor). Must have the same dimensions as `self`.

        # Panics

        Panics if the matrices have different dimensions.

        # Warning

        Division by zero will produce infinity or NaN values depending on the numerator.
        This method does not check for division by zero.

        # Examples

        ## Basic matrix division

        ```
        # use nalgebra::Matrix2;
        let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = Matrix2::new(0.0, 1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0);

        assert_eq!(a.component_div(&b), expected);
        ```

        ## Vector element-wise division

        ```
        # use nalgebra::Vector3;
        let numerator = Vector3::new(10.0, 20.0, 30.0);
        let denominator = Vector3::new(2.0, 4.0, 5.0);
        let result = numerator.component_div(&denominator);
        assert_eq!(result, Vector3::new(5.0, 5.0, 6.0));
        ```

        ## Practical use case: Computing ratios

        ```
        # use nalgebra::Vector3;
        let actual = Vector3::new(80.0, 90.0, 95.0);
        let target = Vector3::new(100.0, 100.0, 100.0);
        let ratios = actual.component_div(&target);
        assert_eq!(ratios, Vector3::new(0.8, 0.9, 0.95));
        ```

        ## Converting to relative values

        ```
        # use nalgebra::Vector4;
        let values = Vector4::new(25.0, 50.0, 75.0, 100.0);
        let total = Vector4::new(200.0, 200.0, 200.0, 200.0);
        let percentages = values.component_div(&total);
        assert_eq!(percentages, Vector4::new(0.125, 0.25, 0.375, 0.5));
        ```

        ## Normalizing by different factors per dimension

        ```
        # use nalgebra::Vector3;
        let raw_data = Vector3::new(100.0, 50.0, 25.0);
        let scale_factors = Vector3::new(10.0, 5.0, 2.5);
        let normalized = raw_data.component_div(&scale_factors);
        assert_eq!(normalized, Vector3::new(10.0, 10.0, 10.0));
        ```

        # See Also

        - [`component_div_assign`](Self::component_div_assign): In-place version of this operation
        - [`cdpy`](Self::cdpy): Fused divide-add operation: `self = alpha * a / b + beta * self`
        - [`component_mul`](Self::component_mul): Component-wise multiplication
        ",
        r"
        Computes componentwise `self[i] = alpha * a[i] / b[i] + beta * self[i]`.

        This is a fused divide-add operation that combines component-wise division
        with scaling and accumulation. It's more efficient than performing separate
        operations and is useful in numerical computations where you need to combine
        division results with existing values.

        The operation is performed in-place, modifying `self`.

        # Arguments

        * `alpha` - Scalar multiplier for the component-wise quotient of `a` and `b`
        * `a` - Numerator matrix/vector for component-wise division
        * `b` - Denominator matrix/vector for component-wise division
        * `beta` - Scalar multiplier for the current value of `self`

        # Mathematical Definition

        For each element index `i`: `self[i] = alpha * a[i] / b[i] + beta * self[i]`

        When `beta` is zero, the operation simplifies to: `self[i] = alpha * a[i] / b[i]`
        (without reading the current value of `self[i]`, which can be more efficient).

        # Panics

        Panics if `self`, `a`, and `b` don't have the same dimensions.

        # Warning

        Division by zero in `b` will produce infinity or NaN values.
        This method does not check for division by zero.

        # Examples

        ## Basic usage

        ```
        # use nalgebra::Matrix2;
        let mut m = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let a = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = (a.component_div(&b) * 5.0) + m * 10.0;

        m.cdpy(5.0, &a, &b, 10.0);
        assert_eq!(m, expected);
        ```

        ## With vectors

        ```
        # use nalgebra::Vector3;
        let mut result = Vector3::new(1.0, 2.0, 3.0);
        let numerator = Vector3::new(10.0, 20.0, 30.0);
        let denominator = Vector3::new(2.0, 4.0, 5.0);

        // result = 2.0 * (numerator / denominator) + 3.0 * result
        result.cdpy(2.0, &numerator, &denominator, 3.0);
        assert_eq!(result, Vector3::new(13.0, 16.0, 21.0));
        ```

        ## Beta is zero (doesn't use current self values)

        ```
        # use nalgebra::Vector3;
        let mut result = Vector3::new(100.0, 200.0, 300.0);  // Initial values don't matter
        let a = Vector3::new(10.0, 20.0, 30.0);
        let b = Vector3::new(2.0, 4.0, 5.0);

        // When beta = 0, self is overwritten
        result.cdpy(1.0, &a, &b, 0.0);
        assert_eq!(result, Vector3::new(5.0, 5.0, 6.0));
        ```

        ## Computing weighted ratios with accumulation

        ```
        # use nalgebra::Vector2;
        let mut accumulator = Vector2::new(10.0, 20.0);
        let achieved = Vector2::new(80.0, 90.0);
        let target = Vector2::new(100.0, 100.0);

        // accumulator = 0.5 * (achieved / target) + 1.0 * accumulator
        accumulator.cdpy(0.5, &achieved, &target, 1.0);
        assert_eq!(accumulator, Vector2::new(10.4, 20.45));
        ```

        ## Normalizing and scaling in one operation

        ```
        # use nalgebra::Vector3;
        let mut output = Vector3::new(0.0, 0.0, 0.0);
        let raw_values = Vector3::new(50.0, 100.0, 150.0);
        let max_values = Vector3::new(200.0, 400.0, 300.0);

        // Scale to [0, 1] range then multiply by 10
        output.cdpy(10.0, &raw_values, &max_values, 0.0);
        assert_eq!(output, Vector3::new(2.5, 2.5, 5.0));
        ```

        # See Also

        - [`component_div`](Self::component_div): Component-wise division
        - [`component_div_assign`](Self::component_div_assign): In-place component-wise division
        - [`cmpy`](Self::cmpy): Similar operation but with component-wise multiplication
        ",
        r"
        In-place componentwise matrix or vector division.

        This operation divides each element of `self` by the corresponding element of `rhs`,
        modifying `self` in place. This is the in-place version of
        [`component_div`](Self::component_div), performing element-wise division.

        This method is more memory-efficient than `component_div` as it doesn't allocate
        a new matrix/vector for the result.

        # Arguments

        * `rhs` - The right-hand side matrix or vector (divisor). Must have the same dimensions as `self`.

        # Panics

        Panics if the matrices have different dimensions.

        # Warning

        Division by zero will produce infinity or NaN values depending on the numerator.
        This method does not check for division by zero.

        # Examples

        ## Basic matrix division

        ```
        # use nalgebra::Matrix2;
        let mut a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
        let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
        let expected = Matrix2::new(0.0, 1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0);

        a.component_div_assign(&b);

        assert_eq!(a, expected);
        ```

        ## Vector element-wise division

        ```
        # use nalgebra::Vector3;
        let mut numerator = Vector3::new(10.0, 20.0, 30.0);
        let denominator = Vector3::new(2.0, 4.0, 5.0);
        numerator.component_div_assign(&denominator);
        assert_eq!(numerator, Vector3::new(5.0, 5.0, 6.0));
        ```

        ## Normalizing values

        ```
        # use nalgebra::Vector3;
        let mut values = Vector3::new(50.0, 150.0, 200.0);
        let max_values = Vector3::new(100.0, 300.0, 400.0);
        values.component_div_assign(&max_values);
        assert_eq!(values, Vector3::new(0.5, 0.5, 0.5));
        ```

        ## Applying decay factors

        ```
        # use nalgebra::Vector2;
        let mut value = Vector2::new(100.0, 200.0);
        let divisor = Vector2::new(2.0, 4.0);

        // Repeatedly divide
        value.component_div_assign(&divisor);
        assert_eq!(value, Vector2::new(50.0, 50.0));

        value.component_div_assign(&divisor);
        assert_eq!(value, Vector2::new(25.0, 12.5));
        ```

        ## Converting absolute values to percentages

        ```
        # use nalgebra::Vector4;
        let mut parts = Vector4::new(10.0, 20.0, 30.0, 40.0);
        let totals = Vector4::new(100.0, 100.0, 100.0, 100.0);
        parts.component_div_assign(&totals);
        assert_eq!(parts, Vector4::new(0.1, 0.2, 0.3, 0.4));
        ```

        # See Also

        - [`component_div`](Self::component_div): Non-mutating version that returns a new matrix/vector
        - [`cdpy`](Self::cdpy): Fused divide-add: `self = alpha * a / b + beta * self`
        - [`component_mul_assign`](Self::component_mul_assign): In-place component-wise multiplication
        ";
        // TODO: add other operators like bitshift, etc. ?
    );

    /// Computes the infimum (aka. componentwise min) of two matrices/vectors.
    ///
    /// This method compares each element of `self` with the corresponding element of `other`
    /// and returns a new matrix/vector containing the minimum value at each position.
    /// This is also known as the element-wise minimum or componentwise minimum.
    ///
    /// The infimum is a term from mathematics meaning the greatest lower bound. In the context
    /// of component-wise operations, it simply means taking the minimum of each pair of elements.
    ///
    /// # Arguments
    ///
    /// * `other` - Another matrix or vector with the same dimensions as `self`.
    ///
    /// # Examples
    ///
    /// ## Basic usage with matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let u = Matrix2::new(4.0, 2.0, 1.0, -2.0);
    /// let v = Matrix2::new(2.0, 4.0, -2.0, 1.0);
    /// let expected = Matrix2::new(2.0, 2.0, -2.0, -2.0);
    /// assert_eq!(u.inf(&v), expected)
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let a = Vector3::new(1.0, 5.0, 3.0);
    /// let b = Vector3::new(2.0, 4.0, 6.0);
    /// let min_values = a.inf(&b);
    /// assert_eq!(min_values, Vector3::new(1.0, 4.0, 3.0));
    /// ```
    ///
    /// ## Practical use case: Clamping to maximum bounds
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let position = Vector3::new(15.0, 25.0, 35.0);
    /// let max_bounds = Vector3::new(10.0, 30.0, 40.0);
    /// // Ensure position doesn't exceed maximum bounds
    /// let clamped = position.inf(&max_bounds);
    /// assert_eq!(clamped, Vector3::new(10.0, 25.0, 35.0));
    /// ```
    ///
    /// ## Finding element-wise minimum across datasets
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let readings_day1 = Vector4::new(20.0, 22.0, 19.0, 21.0);
    /// let readings_day2 = Vector4::new(19.0, 23.0, 18.0, 22.0);
    /// let minimums = readings_day1.inf(&readings_day2);
    /// assert_eq!(minimums, Vector4::new(19.0, 22.0, 18.0, 21.0));
    /// ```
    ///
    /// ## Working with negative numbers
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let a = Vector3::new(-5.0, 0.0, 5.0);
    /// let b = Vector3::new(-3.0, 0.0, 3.0);
    /// let result = a.inf(&b);
    /// assert_eq!(result, Vector3::new(-5.0, 0.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`sup`](Self::sup): Component-wise maximum (supremum)
    /// - [`inf_sup`](Self::inf_sup): Compute both minimum and maximum simultaneously
    /// - [`abs`](Self::abs): Component-wise absolute value
    #[inline]
    #[must_use]
    pub fn inf(&self, other: &Self) -> OMatrix<T, R1, C1>
    where
        T: SimdPartialOrd,
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.zip_map(other, |a, b| a.simd_min(b))
    }

    /// Computes the supremum (aka. componentwise max) of two matrices/vectors.
    ///
    /// This method compares each element of `self` with the corresponding element of `other`
    /// and returns a new matrix/vector containing the maximum value at each position.
    /// This is also known as the element-wise maximum or componentwise maximum.
    ///
    /// The supremum is a term from mathematics meaning the least upper bound. In the context
    /// of component-wise operations, it simply means taking the maximum of each pair of elements.
    ///
    /// # Arguments
    ///
    /// * `other` - Another matrix or vector with the same dimensions as `self`.
    ///
    /// # Examples
    ///
    /// ## Basic usage with matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let u = Matrix2::new(4.0, 2.0, 1.0, -2.0);
    /// let v = Matrix2::new(2.0, 4.0, -2.0, 1.0);
    /// let expected = Matrix2::new(4.0, 4.0, 1.0, 1.0);
    /// assert_eq!(u.sup(&v), expected)
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let a = Vector3::new(1.0, 5.0, 3.0);
    /// let b = Vector3::new(2.0, 4.0, 6.0);
    /// let max_values = a.sup(&b);
    /// assert_eq!(max_values, Vector3::new(2.0, 5.0, 6.0));
    /// ```
    ///
    /// ## Practical use case: Clamping to minimum bounds
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let position = Vector3::new(5.0, 15.0, 25.0);
    /// let min_bounds = Vector3::new(10.0, 10.0, 10.0);
    /// // Ensure position meets minimum bounds
    /// let clamped = position.sup(&min_bounds);
    /// assert_eq!(clamped, Vector3::new(10.0, 15.0, 25.0));
    /// ```
    ///
    /// ## Finding element-wise maximum across datasets
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let readings_day1 = Vector4::new(20.0, 22.0, 19.0, 21.0);
    /// let readings_day2 = Vector4::new(19.0, 23.0, 18.0, 22.0);
    /// let maximums = readings_day1.sup(&readings_day2);
    /// assert_eq!(maximums, Vector4::new(20.0, 23.0, 19.0, 22.0));
    /// ```
    ///
    /// ## Accumulating maximum values over time
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut max_seen = Vector3::new(10.0, 20.0, 30.0);
    /// let new_values = Vector3::new(15.0, 18.0, 35.0);
    /// max_seen = max_seen.sup(&new_values);
    /// assert_eq!(max_seen, Vector3::new(15.0, 20.0, 35.0));
    /// ```
    ///
    /// ## Working with negative numbers
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let a = Vector3::new(-5.0, 0.0, 5.0);
    /// let b = Vector3::new(-3.0, 0.0, 3.0);
    /// let result = a.sup(&b);
    /// assert_eq!(result, Vector3::new(-3.0, 0.0, 5.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inf`](Self::inf): Component-wise minimum (infimum)
    /// - [`inf_sup`](Self::inf_sup): Compute both minimum and maximum simultaneously
    /// - [`abs`](Self::abs): Component-wise absolute value
    #[inline]
    #[must_use]
    pub fn sup(&self, other: &Self) -> OMatrix<T, R1, C1>
    where
        T: SimdPartialOrd,
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.zip_map(other, |a, b| a.simd_max(b))
    }

    /// Computes the (infimum, supremum) of two matrices/vectors.
    ///
    /// This is a convenience method that computes both the component-wise minimum and maximum
    /// in a single call, returning them as a tuple. This is equivalent to calling both
    /// [`inf`](Self::inf) and [`sup`](Self::sup), but may be more convenient when you need
    /// both results.
    ///
    /// The first element of the tuple is the infimum (component-wise minimum), and the
    /// second element is the supremum (component-wise maximum).
    ///
    /// # Arguments
    ///
    /// * `other` - Another matrix or vector with the same dimensions as `self`.
    ///
    /// # Returns
    ///
    /// A tuple `(min, max)` where:
    /// - `min` contains the component-wise minimum values
    /// - `max` contains the component-wise maximum values
    ///
    /// # Examples
    ///
    /// ## Basic usage with matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let u = Matrix2::new(4.0, 2.0, 1.0, -2.0);
    /// let v = Matrix2::new(2.0, 4.0, -2.0, 1.0);
    /// let expected = (Matrix2::new(2.0, 2.0, -2.0, -2.0), Matrix2::new(4.0, 4.0, 1.0, 1.0));
    /// assert_eq!(u.inf_sup(&v), expected)
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let a = Vector3::new(1.0, 5.0, 3.0);
    /// let b = Vector3::new(2.0, 4.0, 6.0);
    /// let (min, max) = a.inf_sup(&b);
    /// assert_eq!(min, Vector3::new(1.0, 4.0, 3.0));
    /// assert_eq!(max, Vector3::new(2.0, 5.0, 6.0));
    /// ```
    ///
    /// ## Practical use case: Finding value ranges
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let sensor_reading1 = Vector4::new(20.5, 21.0, 19.5, 22.0);
    /// let sensor_reading2 = Vector4::new(19.5, 22.0, 20.0, 21.5);
    /// let (minimums, maximums) = sensor_reading1.inf_sup(&sensor_reading2);
    ///
    /// // Now we have the range of values for each sensor
    /// assert_eq!(minimums, Vector4::new(19.5, 21.0, 19.5, 21.5));
    /// assert_eq!(maximums, Vector4::new(20.5, 22.0, 20.0, 22.0));
    /// ```
    ///
    /// ## Computing bounds for clamping
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let bounds1 = Vector3::new(0.0, 5.0, 10.0);
    /// let bounds2 = Vector3::new(10.0, 15.0, 20.0);
    /// let (lower_bounds, upper_bounds) = bounds1.inf_sup(&bounds2);
    ///
    /// // Get the valid range
    /// assert_eq!(lower_bounds, Vector3::new(0.0, 5.0, 10.0));
    /// assert_eq!(upper_bounds, Vector3::new(10.0, 15.0, 20.0));
    /// ```
    ///
    /// ## Analyzing data ranges
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let dataset_a = Vector2::new(15.0, 25.0);
    /// let dataset_b = Vector2::new(20.0, 22.0);
    /// let (mins, maxs) = dataset_a.inf_sup(&dataset_b);
    ///
    /// // Calculate ranges for each dimension
    /// let ranges = maxs - mins;
    /// assert_eq!(ranges, Vector2::new(5.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inf`](Self::inf): Component-wise minimum only
    /// - [`sup`](Self::sup): Component-wise maximum only
    /// - [`abs`](Self::abs): Component-wise absolute value
    #[inline]
    #[must_use]
    pub fn inf_sup(&self, other: &Self) -> (OMatrix<T, R1, C1>, OMatrix<T, R1, C1>)
    where
        T: SimdPartialOrd,
        DefaultAllocator: Allocator<R1, C1>,
    {
        // TODO: can this be optimized?
        (self.inf(other), self.sup(other))
    }

    /// Adds a scalar to each element of the matrix or vector.
    ///
    /// This method creates a new matrix/vector where the given scalar value is added to
    /// every element. This is useful for shifting all values by a constant amount.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to add to each element.
    ///
    /// # Returns
    ///
    /// A new matrix/vector with the scalar added to each element.
    ///
    /// # Examples
    ///
    /// ## Basic usage with matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let u = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    /// let s = 10.0;
    /// let expected = Matrix2::new(11.0, 12.0, 13.0, 14.0);
    /// assert_eq!(u.add_scalar(s), expected)
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let shifted = v.add_scalar(5.0);
    /// assert_eq!(shifted, Vector3::new(6.0, 7.0, 8.0));
    /// ```
    ///
    /// ## Practical use case: Offsetting coordinates
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// // Move all points by an offset
    /// let position = Vector2::new(10.0, 20.0);
    /// let offset = 5.0;
    /// let new_position = position.add_scalar(offset);
    /// assert_eq!(new_position, Vector2::new(15.0, 25.0));
    /// ```
    ///
    /// ## Adjusting brightness values
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// // Increase brightness of RGBA color
    /// let color = Vector4::new(0.5, 0.3, 0.2, 1.0);
    /// let brighter = color.add_scalar(0.1);
    /// assert_eq!(brighter, Vector4::new(0.6, 0.4, 0.3, 1.1));
    /// ```
    ///
    /// ## Adding negative values (subtraction)
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(10.0, 20.0, 30.0);
    /// let decreased = v.add_scalar(-5.0);
    /// assert_eq!(decreased, Vector3::new(5.0, 15.0, 25.0));
    /// ```
    ///
    /// ## Converting from zero-based to one-based indexing
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let indices = Vector4::new(0.0, 1.0, 2.0, 3.0);
    /// let one_based = indices.add_scalar(1.0);
    /// assert_eq!(one_based, Vector4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`add_scalar_mut`](Self::add_scalar_mut): In-place version of this operation
    /// - [`component_mul`](Self::component_mul): Multiply each element by corresponding elements
    /// - [`abs`](Self::abs): Component-wise absolute value
    #[inline]
    #[must_use = "Did you mean to use add_scalar_mut()?"]
    pub fn add_scalar(&self, rhs: T) -> OMatrix<T, R1, C1>
    where
        T: ClosedAddAssign,
        DefaultAllocator: Allocator<R1, C1>,
    {
        let mut res = self.clone_owned();
        res.add_scalar_mut(rhs);
        res
    }

    /// Adds a scalar to each element of the matrix or vector in-place.
    ///
    /// This method modifies `self` by adding the given scalar value to every element.
    /// This is the in-place version of [`add_scalar`](Self::add_scalar), which is more
    /// memory-efficient as it doesn't allocate a new matrix/vector.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar value to add to each element.
    ///
    /// # Examples
    ///
    /// ## Basic usage with matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut u = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    /// let s = 10.0;
    /// u.add_scalar_mut(s);
    /// let expected = Matrix2::new(11.0, 12.0, 13.0, 14.0);
    /// assert_eq!(u, expected)
    /// ```
    ///
    /// ## Using with vectors
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v = Vector3::new(1.0, 2.0, 3.0);
    /// v.add_scalar_mut(5.0);
    /// assert_eq!(v, Vector3::new(6.0, 7.0, 8.0));
    /// ```
    ///
    /// ## Practical use case: Progressive shifting
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let mut position = Vector2::new(10.0, 20.0);
    ///
    /// // Apply multiple shifts
    /// position.add_scalar_mut(5.0);
    /// assert_eq!(position, Vector2::new(15.0, 25.0));
    ///
    /// position.add_scalar_mut(3.0);
    /// assert_eq!(position, Vector2::new(18.0, 28.0));
    /// ```
    ///
    /// ## Bias adjustment in data processing
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let mut sensor_data = Vector4::new(2.0, 3.0, 4.0, 5.0);
    /// let bias_correction = -0.5;
    /// sensor_data.add_scalar_mut(bias_correction);
    /// assert_eq!(sensor_data, Vector4::new(1.5, 2.5, 3.5, 4.5));
    /// ```
    ///
    /// ## Incremental updates
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut counter = Vector3::new(0.0, 0.0, 0.0);
    ///
    /// // Increment all counters
    /// for _ in 0..5 {
    ///     counter.add_scalar_mut(1.0);
    /// }
    /// assert_eq!(counter, Vector3::new(5.0, 5.0, 5.0));
    /// ```
    ///
    /// ## Normalizing by offset
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut measurements = Vector3::new(100.0, 101.0, 102.0);
    /// // Remove baseline
    /// measurements.add_scalar_mut(-100.0);
    /// assert_eq!(measurements, Vector3::new(0.0, 1.0, 2.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`add_scalar`](Self::add_scalar): Non-mutating version that returns a new matrix/vector
    /// - [`component_mul_assign`](Self::component_mul_assign): In-place component-wise multiplication
    /// - [`component_div_assign`](Self::component_div_assign): In-place component-wise division
    #[inline]
    pub fn add_scalar_mut(&mut self, rhs: T)
    where
        T: ClosedAddAssign,
        SA: StorageMut<T, R1, C1>,
    {
        for e in self.iter_mut() {
            *e += rhs.clone()
        }
    }
}
