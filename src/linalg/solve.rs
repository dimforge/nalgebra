use simba::scalar::ComplexField;
use simba::simd::SimdComplexField;

use crate::base::allocator::Allocator;
use crate::base::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DVectorView, DefaultAllocator, Matrix, OMatrix, SquareMatrix, Vector};

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Solves a lower triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `L · x = b`, where:
    /// - `L` is this lower triangular matrix (only elements on and below the diagonal are used)
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # What is a Linear System?
    ///
    /// A linear system is an equation of the form `A · x = b`, where you know the matrix `A`
    /// and vector `b`, and you want to find the vector `x`. When `A` is triangular (all zeros
    /// above or below the diagonal), we can solve this efficiently using forward or backward
    /// substitution instead of more expensive methods like Gaussian elimination.
    ///
    /// # Lower Triangular Systems
    ///
    /// A lower triangular matrix has all zeros above the diagonal. For example:
    /// ```text
    /// [2  0  0]
    /// [3  4  0]
    /// [1  5  6]
    /// ```
    ///
    /// This special structure allows us to solve the system efficiently by forward substitution,
    /// starting from the first row and working downward.
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if the system cannot
    /// be solved (e.g., if any diagonal element is zero, which would require division by zero).
    ///
    /// # Examples
    ///
    /// ## Solving a Simple System
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Create a lower triangular matrix L
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// // Right-hand side vector b
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // Solve L·x = b
    /// let x = l.solve_lower_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify the solution: L·x should equal b
    /// let result = l * x;
    /// assert!((result - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Solving with Multiple Right-Hand Sides
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Matrix3x2};
    /// // Same lower triangular matrix
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// // Multiple right-hand sides (each column is a separate system)
    /// let b = Matrix3x2::new(
    ///     2.0, 4.0,
    ///     11.0, 7.0,
    ///     30.0, 18.0,
    /// );
    ///
    /// // Solve L·X = B for matrix X
    /// let x = l.solve_lower_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify: L·X = B
    /// let result = l * x;
    /// assert!((result - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Handling Singular Systems
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Matrix with zero on diagonal (singular)
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 0.0, 0.0,  // Zero diagonal element!
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // This will return None because we can't divide by zero
    /// assert!(l.solve_lower_triangular(&b).is_none());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut): In-place version that modifies `b`
    /// - [`solve_lower_triangular_unchecked`](Self::solve_lower_triangular_unchecked): Unchecked version (no zero-diagonal check)
    /// - [`solve_upper_triangular`](Self::solve_upper_triangular): For upper triangular systems
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular): Solves with the transpose
    #[must_use = "Did you mean to use solve_lower_triangular_mut()?"]
    #[inline]
    pub fn solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves an upper triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `U · x = b`, where:
    /// - `U` is this upper triangular matrix (only elements on and above the diagonal are used)
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # Upper Triangular Systems
    ///
    /// An upper triangular matrix has all zeros below the diagonal. For example:
    /// ```text
    /// [2  3  1]
    /// [0  4  5]
    /// [0  0  6]
    /// ```
    ///
    /// This special structure allows us to solve the system efficiently by backward substitution,
    /// starting from the last row and working upward.
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if the system cannot
    /// be solved (e.g., if any diagonal element is zero).
    ///
    /// # Examples
    ///
    /// ## Solving a Simple System
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Create an upper triangular matrix U
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// // Right-hand side vector b
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    ///
    /// // Solve U·x = b
    /// let x = u.solve_upper_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify the solution: U·x should equal b
    /// let result = u * x;
    /// assert!((result - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Real-World Example: Back Substitution After LU Decomposition
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // After performing LU decomposition, you often need to solve
    /// // an upper triangular system as the final step
    /// let u = Matrix3::new(
    ///     3.0, 1.0, 2.0,
    ///     0.0, 2.0, 1.0,
    ///     0.0, 0.0, 4.0,
    /// );
    ///
    /// let b = Vector3::new(8.0, 5.0, 8.0);
    /// let x = u.solve_upper_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify
    /// assert!((u * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Handling Singular Systems
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Matrix with zero on diagonal (singular)
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 0.0, 5.0,  // Zero diagonal element!
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    ///
    /// // This will return None because we can't divide by zero
    /// assert!(u.solve_upper_triangular(&b).is_none());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_upper_triangular_mut`](Self::solve_upper_triangular_mut): In-place version that modifies `b`
    /// - [`solve_upper_triangular_unchecked`](Self::solve_upper_triangular_unchecked): Unchecked version (no zero-diagonal check)
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular): For lower triangular systems
    /// - [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular): Solves with the transpose
    #[must_use = "Did you mean to use solve_upper_triangular_mut()?"]
    #[inline]
    pub fn solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves a lower triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `L · x = b`, where:
    /// - `L` is this lower triangular matrix (only elements on and below the diagonal are used)
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// The solution is stored directly in `b`, overwriting the original values. This is more
    /// memory-efficient than [`solve_lower_triangular`](Self::solve_lower_triangular) when you
    /// don't need to keep the original `b`.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the system was solved successfully, or `false` if any diagonal element
    /// is zero (which would make the system unsolvable).
    ///
    /// # Examples
    ///
    /// ## Basic In-Place Solving
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// // Start with the right-hand side
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // Solve in-place: x now contains the solution
    /// let success = l.solve_lower_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// // Verify: L·x = b
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((l * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Memory-Efficient Batch Solving
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Matrix3x2};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// // Multiple systems to solve
    /// let mut solutions = Matrix3x2::new(
    ///     2.0, 4.0,
    ///     11.0, 7.0,
    ///     30.0, 18.0,
    /// );
    ///
    /// // Solve all systems in-place
    /// let success = l.solve_lower_triangular_mut(&mut solutions);
    /// assert!(success);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular): Non-mutating version that returns a new matrix
    /// - [`solve_upper_triangular_mut`](Self::solve_upper_triangular_mut): For upper triangular systems
    pub fn solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.solve_lower_triangular_vector_mut(&mut b.column_mut(i)) {
                return false;
            }
        }

        true
    }

    fn solve_lower_triangular_vector_mut<R2: Dim, S2>(&self, b: &mut Vector<T, R2, S2>) -> bool
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in 0..dim {
            let coeff;

            unsafe {
                let diag = self.get_unchecked((i, i)).clone();

                if diag.is_zero() {
                    return false;
                }

                coeff = b.vget_unchecked(i).clone() / diag;
                *b.vget_unchecked_mut(i) = coeff.clone();
            }

            b.rows_range_mut(i + 1..)
                .axpy(-coeff, &self.view_range(i + 1.., i), T::one());
        }

        true
    }

    // TODO: add the same but for solving upper-triangular.
    /// Solves a lower triangular system with a uniform diagonal value.
    ///
    /// This method computes the solution `x` of the linear system `L · x = b`, where:
    /// - `L` is a lower triangular matrix with a constant value `diag` on the diagonal
    /// - The actual diagonal elements of `self` are ignored
    /// - Only the strictly lower triangular part of `self` is used
    ///
    /// This is useful when you have a matrix decomposition where the diagonal is stored
    /// separately or is known to be a constant value (like 1.0 in unit lower triangular matrices).
    ///
    /// # Return Value
    ///
    /// Returns `true` if successful, or `false` if `diag` is zero (which would require
    /// division by zero). The input `b` is not modified if `diag` is zero.
    ///
    /// # Examples
    ///
    /// ## Unit Lower Triangular System
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // A unit lower triangular matrix (diagonal = 1.0, but we ignore it)
    /// let l = Matrix3::new(
    ///     999.0, 0.0, 0.0,  // Diagonal values are ignored
    ///     3.0, 888.0, 0.0,
    ///     1.0, 5.0, 777.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 5.0, 12.0);
    ///
    /// // Solve treating the diagonal as all 1.0
    /// let success = l.solve_lower_triangular_with_diag_mut(&mut x, 1.0);
    /// assert!(success);
    ///
    /// // Verify with the actual unit lower triangular matrix
    /// let l_unit = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     3.0, 1.0, 0.0,
    ///     1.0, 5.0, 1.0,
    /// );
    /// let b = Vector3::new(2.0, 5.0, 12.0);
    /// assert!((l_unit * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## LU Decomposition with Implicit Unit Diagonal
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // In LU decomposition, L often has unit diagonal stored implicitly
    /// // The lower triangular part contains the L factors
    /// let lu = Matrix3::new(
    ///     2.0, 3.0, 1.0,  // U in upper part, L in lower part
    ///     0.5, 1.0, 2.5,  // L's diagonal is implicitly 1.0
    ///     0.25, 0.5, 3.0,
    /// );
    ///
    /// let mut y = Vector3::new(4.0, 3.0, 2.0);
    ///
    /// // Solve L·y = b where L has unit diagonal
    /// let success = lu.solve_lower_triangular_with_diag_mut(&mut y, 1.0);
    /// assert!(success);
    /// ```
    ///
    /// ## Handling Zero Diagonal
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     0.0, 0.0, 0.0,
    ///     3.0, 0.0, 0.0,
    ///     1.0, 5.0, 0.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 5.0, 12.0);
    ///
    /// // This returns false and doesn't modify x
    /// let success = l.solve_lower_triangular_with_diag_mut(&mut x, 0.0);
    /// assert!(!success);
    /// assert_eq!(x, Vector3::new(2.0, 5.0, 12.0)); // x unchanged
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut): Uses actual diagonal values
    /// - [`solve_lower_triangular_with_diag_unchecked_mut`](Self::solve_lower_triangular_with_diag_unchecked_mut): Unchecked version
    pub fn solve_lower_triangular_with_diag_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
        diag: T,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        if diag.is_zero() {
            return false;
        }

        let dim = self.nrows();
        let cols = b.ncols();

        for k in 0..cols {
            let mut bcol = b.column_mut(k);

            for i in 0..dim - 1 {
                let coeff = unsafe { bcol.vget_unchecked(i).clone() } / diag.clone();
                bcol.rows_range_mut(i + 1..)
                    .axpy(-coeff, &self.view_range(i + 1.., i), T::one());
            }
        }

        true
    }

    /// Solves an upper triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `U · x = b`, where:
    /// - `U` is this upper triangular matrix (only elements on and above the diagonal are used)
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// The solution is stored directly in `b`, overwriting the original values. This is more
    /// memory-efficient than [`solve_upper_triangular`](Self::solve_upper_triangular) when you
    /// don't need to keep the original `b`.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the system was solved successfully, or `false` if any diagonal element
    /// is zero.
    ///
    /// # Examples
    ///
    /// ## Basic In-Place Solving
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    ///
    /// // Solve in-place
    /// let success = u.solve_upper_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// // Verify
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((u * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_upper_triangular`](Self::solve_upper_triangular): Non-mutating version
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut): For lower triangular systems
    pub fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.solve_upper_triangular_vector_mut(&mut b.column_mut(i)) {
                return false;
            }
        }

        true
    }

    fn solve_upper_triangular_vector_mut<R2: Dim, S2>(&self, b: &mut Vector<T, R2, S2>) -> bool
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let coeff;

            unsafe {
                let diag = self.get_unchecked((i, i)).clone();

                if diag.is_zero() {
                    return false;
                }

                coeff = b.vget_unchecked(i).clone() / diag;
                *b.vget_unchecked_mut(i) = coeff.clone();
            }

            b.rows_range_mut(..i)
                .axpy(-coeff, &self.view_range(..i, i), T::one());
        }

        true
    }

    /*
     *
     * Transpose and adjoint versions
     *
     */
    /// Solves a transposed lower triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `L^T · x = b`, where:
    /// - `L^T` is the transpose of this lower triangular matrix
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # Why Solve with the Transpose?
    ///
    /// When you have a lower triangular matrix `L` and need to solve `L^T · x = b`, you're
    /// actually solving with an upper triangular matrix (since the transpose of a lower
    /// triangular matrix is upper triangular). This is common in:
    /// - Cholesky decomposition (solving `L^T · x = y` after solving `L · y = b`)
    /// - Least squares problems
    /// - Various matrix factorization algorithms
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if any diagonal
    /// element is zero.
    ///
    /// # Examples
    ///
    /// ## Basic Transpose Solve
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Lower triangular matrix L
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// // Its transpose L^T is:
    /// // [2  3  1]
    /// // [0  4  5]
    /// // [0  0  6]
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    ///
    /// // Solve L^T · x = b
    /// let x = l.tr_solve_lower_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify: L^T · x = b
    /// assert!((l.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Cholesky Decomposition Pattern
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // After Cholesky decomposition: A = L·L^T
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     1.0, 1.5, 0.0,
    ///     3.0, 2.0, 1.0,
    /// );
    ///
    /// // To solve A·x = b, we:
    /// // 1. First solve L·y = b
    /// let b = Vector3::new(6.0, 5.0, 10.0);
    /// let y = l.solve_lower_triangular(&b).unwrap();
    ///
    /// // 2. Then solve L^T·x = y
    /// let x = l.tr_solve_lower_triangular(&y).unwrap();
    ///
    /// // Verify: (L·L^T)·x = b
    /// let a = &l * l.transpose();
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_lower_triangular_mut`](Self::tr_solve_lower_triangular_mut): In-place version
    /// - [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular): For transposed upper triangular systems
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular): Non-transposed version
    /// - [`ad_solve_lower_triangular`](Self::ad_solve_lower_triangular): Adjoint (conjugate transpose) version
    #[must_use = "Did you mean to use tr_solve_lower_triangular_mut()?"]
    #[inline]
    pub fn tr_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.tr_solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves a transposed upper triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `U^T · x = b`, where:
    /// - `U^T` is the transpose of this upper triangular matrix
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # Note
    ///
    /// The transpose of an upper triangular matrix is lower triangular. This method is useful
    /// when you have an upper triangular matrix stored and need to solve with its transpose.
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if any diagonal
    /// element is zero.
    ///
    /// # Examples
    ///
    /// ## Basic Transpose Solve
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Upper triangular matrix U
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// // Its transpose U^T is:
    /// // [2  0  0]
    /// // [3  4  0]
    /// // [1  5  6]
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // Solve U^T · x = b
    /// let x = u.tr_solve_upper_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify: U^T · x = b
    /// assert!((u.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_upper_triangular_mut`](Self::tr_solve_upper_triangular_mut): In-place version
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular): For transposed lower triangular systems
    /// - [`solve_upper_triangular`](Self::solve_upper_triangular): Non-transposed version
    #[must_use = "Did you mean to use tr_solve_upper_triangular_mut()?"]
    #[inline]
    pub fn tr_solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.tr_solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves a transposed lower triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `L^T · x = b`, where:
    /// - `L^T` is the transpose of this lower triangular matrix
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// # Return Value
    ///
    /// Returns `true` if successful, or `false` if any diagonal element is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    /// let success = l.tr_solve_lower_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// // Verify: L^T · x = b
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((l.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular): Non-mutating version
    /// - [`tr_solve_upper_triangular_mut`](Self::tr_solve_upper_triangular_mut): For transposed upper triangular systems
    pub fn tr_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_lower_triangular_vector_mut(
                &mut b.column_mut(i),
                |e| e,
                |a, b| a.dot(b),
            ) {
                return false;
            }
        }

        true
    }

    /// Solves a transposed upper triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `U^T · x = b`, where:
    /// - `U^T` is the transpose of this upper triangular matrix
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// # Return Value
    ///
    /// Returns `true` if successful, or `false` if any diagonal element is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    /// let success = u.tr_solve_upper_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// // Verify: U^T · x = b
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((u.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular): Non-mutating version
    /// - [`tr_solve_lower_triangular_mut`](Self::tr_solve_lower_triangular_mut): For transposed lower triangular systems
    pub fn tr_solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_upper_triangular_vector_mut(
                &mut b.column_mut(i),
                |e| e,
                |a, b| a.dot(b),
            ) {
                return false;
            }
        }

        true
    }

    /// Solves an adjoint lower triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `L^H · x = b`, where:
    /// - `L^H` is the adjoint (conjugate transpose) of this lower triangular matrix
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # What is the Adjoint?
    ///
    /// The adjoint (also called Hermitian transpose) is the conjugate transpose of a matrix.
    /// For real numbers, it's the same as the transpose. For complex numbers, it transposes
    /// the matrix and takes the complex conjugate of each element.
    ///
    /// For a complex matrix element `a + bi`, its conjugate is `a - bi`.
    ///
    /// # When to Use This
    ///
    /// Use this method when working with:
    /// - Complex-valued linear systems
    /// - Hermitian positive definite matrices (like in complex Cholesky decomposition)
    /// - Quantum mechanics calculations
    /// - Signal processing with complex signals
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if any diagonal
    /// element is zero.
    ///
    /// # Examples
    ///
    /// ## Real-Valued Case (Same as Transpose)
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // For real numbers, adjoint equals transpose
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// let x = l.ad_solve_lower_triangular(&b).expect("Failed to solve");
    ///
    /// // For real matrices, L^H = L^T
    /// assert!((l.adjoint() * x - b).norm() < 1e-10);
    /// assert!((l.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Complex-Valued Case
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3, Complex};
    /// # use num_complex::Complex64;
    /// // Complex lower triangular matrix
    /// let l = Matrix3::new(
    ///     Complex::new(2.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
    ///     Complex::new(1.0, 1.0), Complex::new(3.0, 0.0), Complex::new(0.0, 0.0),
    ///     Complex::new(0.5, -0.5), Complex::new(2.0, 1.0), Complex::new(4.0, 0.0),
    /// );
    ///
    /// let b = Vector3::new(
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(5.0, 3.0),
    ///     Complex::new(10.0, 2.0),
    /// );
    ///
    /// // Solve L^H · x = b (conjugate transpose)
    /// let x = l.ad_solve_lower_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify: L^H · x = b
    /// assert!((l.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_lower_triangular_mut`](Self::ad_solve_lower_triangular_mut): In-place version
    /// - [`ad_solve_upper_triangular`](Self::ad_solve_upper_triangular): For adjoint upper triangular systems
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular): Transpose (non-conjugate) version
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular): Non-adjoint version
    #[must_use = "Did you mean to use ad_solve_lower_triangular_mut()?"]
    #[inline]
    pub fn ad_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.ad_solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves an adjoint upper triangular linear system.
    ///
    /// This method computes the solution `x` of the linear system `U^H · x = b`, where:
    /// - `U^H` is the adjoint (conjugate transpose) of this upper triangular matrix
    /// - `b` is the known right-hand side vector or matrix
    /// - `x` is the unknown solution we're solving for
    ///
    /// # Note
    ///
    /// The adjoint of an upper triangular matrix is lower triangular. For real matrices,
    /// this is equivalent to [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular).
    ///
    /// # Return Value
    ///
    /// Returns `Some(x)` containing the solution if successful, or `None` if any diagonal
    /// element is zero.
    ///
    /// # Examples
    ///
    /// ## Real-Valued Case
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// let x = u.ad_solve_upper_triangular(&b).expect("Failed to solve");
    ///
    /// // Verify: U^H · x = b (same as U^T for real matrices)
    /// assert!((u.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Complex-Valued Case
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3, Complex};
    /// let u = Matrix3::new(
    ///     Complex::new(2.0, 0.0), Complex::new(1.0, 1.0), Complex::new(0.5, -0.5),
    ///     Complex::new(0.0, 0.0), Complex::new(3.0, 0.0), Complex::new(2.0, 1.0),
    ///     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(4.0, 0.0),
    /// );
    ///
    /// let b = Vector3::new(
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(5.0, 3.0),
    ///     Complex::new(10.0, 2.0),
    /// );
    ///
    /// let x = u.ad_solve_upper_triangular(&b).expect("Failed to solve");
    /// assert!((u.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_upper_triangular_mut`](Self::ad_solve_upper_triangular_mut): In-place version
    /// - [`ad_solve_lower_triangular`](Self::ad_solve_lower_triangular): For adjoint lower triangular systems
    /// - [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular): Transpose (non-conjugate) version
    #[must_use = "Did you mean to use ad_solve_upper_triangular_mut()?"]
    #[inline]
    pub fn ad_solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.ad_solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves an adjoint lower triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `L^H · x = b`, where:
    /// - `L^H` is the adjoint (conjugate transpose) of this lower triangular matrix
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// # Return Value
    ///
    /// Returns `true` if successful, or `false` if any diagonal element is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    /// let success = l.ad_solve_lower_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((l.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_lower_triangular`](Self::ad_solve_lower_triangular): Non-mutating version
    /// - [`ad_solve_upper_triangular_mut`](Self::ad_solve_upper_triangular_mut): For adjoint upper triangular systems
    pub fn ad_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_lower_triangular_vector_mut(
                &mut b.column_mut(i),
                |e| e.conjugate(),
                |a, b| a.dotc(b),
            ) {
                return false;
            }
        }

        true
    }

    /// Solves an adjoint upper triangular linear system in-place.
    ///
    /// This method computes the solution `x` of the linear system `U^H · x = b`, where:
    /// - `U^H` is the adjoint (conjugate transpose) of this upper triangular matrix
    /// - `b` is both the input (right-hand side) and output (solution) - it is modified in-place
    ///
    /// # Return Value
    ///
    /// Returns `true` if successful, or `false` if any diagonal element is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    /// let success = u.ad_solve_upper_triangular_mut(&mut x);
    /// assert!(success);
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((u.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_upper_triangular`](Self::ad_solve_upper_triangular): Non-mutating version
    /// - [`ad_solve_lower_triangular_mut`](Self::ad_solve_lower_triangular_mut): For adjoint lower triangular systems
    pub fn ad_solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_upper_triangular_vector_mut(
                &mut b.column_mut(i),
                |e| e.conjugate(),
                |a, b| a.dotc(b),
            ) {
                return false;
            }
        }

        true
    }

    #[inline(always)]
    fn xx_solve_lower_triangular_vector_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<T, R2, S2>,
        conjugate: impl Fn(T) -> T,
        dot: impl Fn(
            &DVectorView<'_, T, S::RStride, S::CStride>,
            &DVectorView<'_, T, S2::RStride, S2::CStride>,
        ) -> T,
    ) -> bool
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let dot = dot(&self.view_range(i + 1.., i), &b.view_range(i + 1.., 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);

                let diag = conjugate(self.get_unchecked((i, i)).clone());

                if diag.is_zero() {
                    return false;
                }

                *b_i = (b_i.clone() - dot) / diag;
            }
        }

        true
    }

    #[inline(always)]
    fn xx_solve_upper_triangular_vector_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<T, R2, S2>,
        conjugate: impl Fn(T) -> T,
        dot: impl Fn(
            &DVectorView<'_, T, S::RStride, S::CStride>,
            &DVectorView<'_, T, S2::RStride, S2::CStride>,
        ) -> T,
    ) -> bool
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in 0..dim {
            let dot = dot(&self.view_range(..i, i), &b.view_range(..i, 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);
                let diag = conjugate(self.get_unchecked((i, i)).clone());

                if diag.is_zero() {
                    return false;
                }

                *b_i = (b_i.clone() - dot) / diag;
            }
        }

        true
    }
}

/*
 *
 * SIMD-compatible unchecked versions.
 *
 */

impl<T: SimdComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Solves a lower triangular linear system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`solve_lower_triangular`](Self::solve_lower_triangular)
    /// that does not check if diagonal elements are zero. It's optimized for SIMD operations
    /// and should only be used when you're certain that all diagonal elements are non-zero.
    ///
    /// # Safety Considerations
    ///
    /// While this function is not marked as `unsafe`, it will produce incorrect results or
    /// may panic if any diagonal element is zero. Use this only when:
    /// - You have verified that all diagonal elements are non-zero
    /// - Performance is critical and the safety check overhead is unacceptable
    /// - You're working with SIMD types where the check is not possible
    ///
    /// # Performance
    ///
    /// This version is optimized for SIMD operations and may be significantly faster than
    /// the checked version when working with SIMD types.
    ///
    /// # Examples
    ///
    /// ## Basic Usage (Use with Caution)
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Ensure diagonal elements are non-zero
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // Safe because we know the diagonal is non-zero
    /// let x = l.solve_lower_triangular_unchecked(&b);
    ///
    /// // Verify the solution
    /// assert!((l * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## When to Prefer the Checked Version
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // If you're unsure about the matrix, use the checked version
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 0.0, 0.0,  // Zero on diagonal - dangerous!
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    ///
    /// // Use the checked version instead
    /// match l.solve_lower_triangular(&b) {
    ///     Some(x) => println!("Solution: {:?}", x),
    ///     None => println!("Matrix is singular"),
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular): Checked version (recommended for most uses)
    /// - [`solve_lower_triangular_unchecked_mut`](Self::solve_lower_triangular_unchecked_mut): In-place unchecked version
    /// - [`solve_upper_triangular_unchecked`](Self::solve_upper_triangular_unchecked): For upper triangular systems
    #[must_use = "Did you mean to use solve_lower_triangular_unchecked_mut()?"]
    #[inline]
    pub fn solve_lower_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.solve_lower_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves an upper triangular linear system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`solve_upper_triangular`](Self::solve_upper_triangular)
    /// that does not check if diagonal elements are zero. It's optimized for SIMD operations
    /// and should only be used when you're certain that all diagonal elements are non-zero.
    ///
    /// # Safety Considerations
    ///
    /// This function will produce incorrect results or may panic if any diagonal element is zero.
    /// Only use when you have verified that all diagonal elements are non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// let x = u.solve_upper_triangular_unchecked(&b);
    ///
    /// assert!((u * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_upper_triangular`](Self::solve_upper_triangular): Checked version (recommended)
    /// - [`solve_upper_triangular_unchecked_mut`](Self::solve_upper_triangular_unchecked_mut): In-place unchecked version
    /// - [`solve_lower_triangular_unchecked`](Self::solve_lower_triangular_unchecked): For lower triangular systems
    #[must_use = "Did you mean to use solve_upper_triangular_unchecked_mut()?"]
    #[inline]
    pub fn solve_upper_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.solve_upper_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves a lower triangular linear system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations. It does not check
    /// if diagonal elements are zero and will produce incorrect results if they are.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    /// l.solve_lower_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((l * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut): Checked in-place version
    /// - [`solve_lower_triangular_unchecked`](Self::solve_lower_triangular_unchecked): Non-mutating unchecked version
    pub fn solve_lower_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.solve_lower_triangular_vector_unchecked_mut(&mut b.column_mut(i));
        }
    }

    fn solve_lower_triangular_vector_unchecked_mut<R2: Dim, S2>(&self, b: &mut Vector<T, R2, S2>)
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in 0..dim {
            let coeff;

            unsafe {
                let diag = self.get_unchecked((i, i)).clone();
                coeff = b.vget_unchecked(i).clone() / diag;
                *b.vget_unchecked_mut(i) = coeff.clone();
            }

            b.rows_range_mut(i + 1..)
                .axpy(-coeff.clone(), &self.view_range(i + 1.., i), T::one());
        }
    }

    // TODO: add the same but for solving upper-triangular.
    /// Solves a lower triangular system with uniform diagonal value, unchecked version.
    ///
    /// This is an unchecked, in-place version of
    /// [`solve_lower_triangular_with_diag_mut`](Self::solve_lower_triangular_with_diag_mut).
    /// It assumes `diag` is non-zero and optimized for SIMD operations.
    ///
    /// # Safety Considerations
    ///
    /// Will produce incorrect results if `diag` is zero. The caller must ensure `diag` is non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     999.0, 0.0, 0.0,  // Diagonal ignored
    ///     3.0, 888.0, 0.0,
    ///     1.0, 5.0, 777.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 5.0, 12.0);
    /// l.solve_lower_triangular_with_diag_unchecked_mut(&mut x, 1.0);
    ///
    /// let l_unit = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     3.0, 1.0, 0.0,
    ///     1.0, 5.0, 1.0,
    /// );
    /// let b = Vector3::new(2.0, 5.0, 12.0);
    /// assert!((l_unit * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular_with_diag_mut`](Self::solve_lower_triangular_with_diag_mut): Checked version
    pub fn solve_lower_triangular_with_diag_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
        diag: T,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();
        let cols = b.ncols();

        for k in 0..cols {
            let mut bcol = b.column_mut(k);

            for i in 0..dim - 1 {
                let coeff = unsafe { bcol.vget_unchecked(i).clone() } / diag.clone();
                bcol.rows_range_mut(i + 1..)
                    .axpy(-coeff, &self.view_range(i + 1.., i), T::one());
            }
        }
    }

    /// Solves an upper triangular linear system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations. It does not check
    /// if diagonal elements are zero and will produce incorrect results if they are.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    /// u.solve_upper_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((u * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_upper_triangular_mut`](Self::solve_upper_triangular_mut): Checked in-place version
    /// - [`solve_upper_triangular_unchecked`](Self::solve_upper_triangular_unchecked): Non-mutating unchecked version
    pub fn solve_upper_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.solve_upper_triangular_vector_unchecked_mut(&mut b.column_mut(i))
        }
    }

    fn solve_upper_triangular_vector_unchecked_mut<R2: Dim, S2>(&self, b: &mut Vector<T, R2, S2>)
    where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let coeff;

            unsafe {
                let diag = self.get_unchecked((i, i)).clone();
                coeff = b.vget_unchecked(i).clone() / diag;
                *b.vget_unchecked_mut(i) = coeff.clone();
            }

            b.rows_range_mut(..i)
                .axpy(-coeff, &self.view_range(..i, i), T::one());
        }
    }

    /*
     *
     * Transpose and adjoint versions
     *
     */
    /// Solves a transposed lower triangular system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular)
    /// optimized for SIMD operations. Assumes all diagonal elements are non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// let x = l.tr_solve_lower_triangular_unchecked(&b);
    ///
    /// assert!((l.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular): Checked version
    /// - [`tr_solve_lower_triangular_unchecked_mut`](Self::tr_solve_lower_triangular_unchecked_mut): In-place unchecked version
    #[must_use = "Did you mean to use tr_solve_lower_triangular_unchecked_mut()?"]
    #[inline]
    pub fn tr_solve_lower_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.tr_solve_lower_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves a transposed upper triangular system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular)
    /// optimized for SIMD operations. Assumes all diagonal elements are non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// let x = u.tr_solve_upper_triangular_unchecked(&b);
    ///
    /// assert!((u.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_upper_triangular`](Self::tr_solve_upper_triangular): Checked version
    /// - [`tr_solve_upper_triangular_unchecked_mut`](Self::tr_solve_upper_triangular_unchecked_mut): In-place unchecked version
    #[must_use = "Did you mean to use tr_solve_upper_triangular_unchecked_mut()?"]
    #[inline]
    pub fn tr_solve_upper_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.tr_solve_upper_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves a transposed lower triangular system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    /// l.tr_solve_lower_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((l.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_lower_triangular_mut`](Self::tr_solve_lower_triangular_mut): Checked in-place version
    pub fn tr_solve_lower_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.xx_solve_lower_triangular_vector_unchecked_mut(
                &mut b.column_mut(i),
                |e| e,
                |a, b| a.dot(b),
            )
        }
    }

    /// Solves a transposed upper triangular system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    /// u.tr_solve_upper_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((u.transpose() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_upper_triangular_mut`](Self::tr_solve_upper_triangular_mut): Checked in-place version
    pub fn tr_solve_upper_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.xx_solve_upper_triangular_vector_unchecked_mut(
                &mut b.column_mut(i),
                |e| e,
                |a, b| a.dot(b),
            )
        }
    }

    /// Solves an adjoint lower triangular system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`ad_solve_lower_triangular`](Self::ad_solve_lower_triangular)
    /// optimized for SIMD operations. Assumes all diagonal elements are non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// let x = l.ad_solve_lower_triangular_unchecked(&b);
    ///
    /// assert!((l.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_lower_triangular`](Self::ad_solve_lower_triangular): Checked version
    /// - [`ad_solve_lower_triangular_unchecked_mut`](Self::ad_solve_lower_triangular_unchecked_mut): In-place unchecked version
    #[must_use = "Did you mean to use ad_solve_lower_triangular_unchecked_mut()?"]
    #[inline]
    pub fn ad_solve_lower_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.ad_solve_lower_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves an adjoint upper triangular system without checking for zero diagonal elements.
    ///
    /// This is an unchecked version of [`ad_solve_upper_triangular`](Self::ad_solve_upper_triangular)
    /// optimized for SIMD operations. Assumes all diagonal elements are non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// let x = u.ad_solve_upper_triangular_unchecked(&b);
    ///
    /// assert!((u.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_upper_triangular`](Self::ad_solve_upper_triangular): Checked version
    /// - [`ad_solve_upper_triangular_unchecked_mut`](Self::ad_solve_upper_triangular_unchecked_mut): In-place unchecked version
    #[must_use = "Did you mean to use ad_solve_upper_triangular_unchecked_mut()?"]
    #[inline]
    pub fn ad_solve_upper_triangular_unchecked<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.ad_solve_upper_triangular_unchecked_mut(&mut res);
        res
    }

    /// Solves an adjoint lower triangular system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     1.0, 5.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(11.0, 23.0, 12.0);
    /// l.ad_solve_lower_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(11.0, 23.0, 12.0);
    /// assert!((l.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_lower_triangular_mut`](Self::ad_solve_lower_triangular_mut): Checked in-place version
    pub fn ad_solve_lower_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.xx_solve_lower_triangular_vector_unchecked_mut(
                &mut b.column_mut(i),
                |e| e.simd_conjugate(),
                |a, b| a.dotc(b),
            )
        }
    }

    /// Solves an adjoint upper triangular system in-place without checking for zero diagonal elements.
    ///
    /// This is an unchecked, in-place version optimized for SIMD operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let u = Matrix3::new(
    ///     2.0, 3.0, 1.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// let mut x = Vector3::new(2.0, 11.0, 30.0);
    /// u.ad_solve_upper_triangular_unchecked_mut(&mut x);
    ///
    /// let b = Vector3::new(2.0, 11.0, 30.0);
    /// assert!((u.adjoint() * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ad_solve_upper_triangular_mut`](Self::ad_solve_upper_triangular_mut): Checked in-place version
    pub fn ad_solve_upper_triangular_unchecked_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..b.ncols() {
            self.xx_solve_upper_triangular_vector_unchecked_mut(
                &mut b.column_mut(i),
                |e| e.simd_conjugate(),
                |a, b| a.dotc(b),
            )
        }
    }

    #[inline(always)]
    fn xx_solve_lower_triangular_vector_unchecked_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<T, R2, S2>,
        conjugate: impl Fn(T) -> T,
        dot: impl Fn(
            &DVectorView<'_, T, S::RStride, S::CStride>,
            &DVectorView<'_, T, S2::RStride, S2::CStride>,
        ) -> T,
    ) where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let dot = dot(&self.view_range(i + 1.., i), &b.view_range(i + 1.., 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);
                let diag = conjugate(self.get_unchecked((i, i)).clone());
                *b_i = (b_i.clone() - dot) / diag;
            }
        }
    }

    #[inline(always)]
    fn xx_solve_upper_triangular_vector_unchecked_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<T, R2, S2>,
        conjugate: impl Fn(T) -> T,
        dot: impl Fn(
            &DVectorView<'_, T, S::RStride, S::CStride>,
            &DVectorView<'_, T, S2::RStride, S2::CStride>,
        ) -> T,
    ) where
        S2: StorageMut<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..self.nrows() {
            let dot = dot(&self.view_range(..i, i), &b.view_range(..i, 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);
                let diag = conjugate(self.get_unchecked((i, i)).clone());
                *b_i = (b_i.clone() - dot) / diag;
            }
        }
    }
}
