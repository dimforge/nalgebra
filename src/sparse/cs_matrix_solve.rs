use crate::allocator::Allocator;
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::sparse::{CsMatrix, CsStorage, CsVector};
use crate::storage::{Storage, StorageMut};
use crate::{Const, DefaultAllocator, Dim, Matrix, OMatrix, OVector, RealField};

impl<T: RealField, D: Dim, S: CsStorage<T, D, D>> CsMatrix<T, D, D, S> {
    /// Solves the lower-triangular system Lx = b, where L is this sparse matrix.
    ///
    /// This method solves a linear system where the coefficient matrix (this matrix)
    /// is lower triangular. A lower-triangular matrix has all zeros above the diagonal,
    /// which allows for efficient forward substitution to find the solution.
    ///
    /// # What is a Lower-Triangular System?
    ///
    /// A lower-triangular matrix L has the form:
    /// ```text
    /// [a  0  0]
    /// [b  c  0]
    /// [d  e  f]
    /// ```
    ///
    /// This structure allows solving Lx = b efficiently using forward substitution:
    /// starting from the first equation and working downward.
    ///
    /// # Parameters
    ///
    /// - `b`: The right-hand-side vector or matrix (can be multiple columns for multiple systems)
    ///
    /// # Returns
    ///
    /// - `Some(x)`: The solution x where Lx = b
    /// - `None`: If the system is singular (has a zero on the diagonal)
    ///
    /// # When Returns None
    ///
    /// The method returns `None` if:
    /// - Any diagonal element is zero (matrix is singular)
    /// - A diagonal element is missing from the sparse structure
    ///
    /// # Performance
    ///
    /// - Time complexity: O(nnz) where nnz is the number of non-zeros in L
    /// - Much faster than general linear system solvers due to triangular structure
    /// - For sparse matrices, can be orders of magnitude faster than dense solvers
    ///
    /// # Examples
    ///
    /// ## Simple 3x3 System
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector, Vector3};
    ///
    /// // Create a lower-triangular matrix:
    /// // [2.0  0    0  ]
    /// // [1.0  3.0  0  ]
    /// // [0.5  1.0  4.0]
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Right-hand side
    /// let b = Vector3::new(4.0, 7.0, 10.0);
    ///
    /// // Solve Lx = b
    /// let x = lower.solve_lower_triangular(&b).expect("Singular matrix");
    ///
    /// // Verify solution (Lx should equal b)
    /// // x[0] = 4.0/2.0 = 2.0
    /// // x[1] = (7.0 - 1.0*2.0)/3.0 = 5.0/3.0
    /// // x[2] = (10.0 - 0.5*2.0 - 1.0*5.0/3.0)/4.0
    /// ```
    ///
    /// ## Solving Multiple Systems
    ///
    /// You can solve multiple systems with the same matrix by passing a matrix
    /// instead of a vector (each column is a separate right-hand side):
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DMatrix, Matrix3x2};
    ///
    /// // Lower triangular matrix
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Two right-hand sides (2 systems to solve)
    /// let b = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// // Solve both systems at once
    /// let x = lower.solve_lower_triangular(&b).expect("Singular matrix");
    ///
    /// assert_eq!(x.ncols(), 2); // Two solution vectors
    /// ```
    ///
    /// ## Handling Singular Matrices
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Vector3};
    ///
    /// // Lower triangular with zero on diagonal (singular)
    /// let irows = vec![0, 1, 1, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1]; // Missing diagonal element (2,2)
    /// let vals = vec![2.0, 1.0, 0.0, 0.5, 1.0]; // Zero at (1,1)
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let b = Vector3::new(4.0, 7.0, 10.0);
    ///
    /// // Returns None because of zero/missing diagonal
    /// assert!(lower.solve_lower_triangular(&b).is_none());
    /// ```
    ///
    /// ## Use in Cholesky Decomposition
    ///
    /// Lower-triangular solves are essential for Cholesky-based linear solvers:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// // For a positive definite matrix A = LL^T (Cholesky decomposition)
    /// // Solving Ax = b requires two triangular solves:
    /// //   1. Solve Ly = b for y (forward substitution)
    /// //   2. Solve L^T x = y for x (backward substitution)
    ///
    /// // Example: lower triangular Cholesky factor
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 1.73, 0.5, 0.87, 1.58];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// // Step 1: forward solve
    /// let y = lower.solve_lower_triangular(&b).expect("Cholesky factor should be non-singular");
    ///
    /// // Step 2: backward solve (with transpose)
    /// let x = lower.tr_solve_lower_triangular(&y).expect("Cholesky factor should be non-singular");
    /// ```
    ///
    /// # Common Use Cases
    ///
    /// - **Cholesky Decomposition**: Solving A = LL^T systems
    /// - **LU Decomposition**: Solving A = LU systems (forward substitution step)
    /// - **Incomplete Factorizations**: Preconditioners in iterative methods
    /// - **Forward Substitution**: Any algorithm requiring lower-triangular solves
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut) - In-place version (more efficient)
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular) - Solve with L^T (transposed)
    /// - [`solve_lower_triangular_cs`](Self::solve_lower_triangular_cs) - Solve with sparse right-hand side
    #[must_use = "Did you mean to use solve_lower_triangular_mut()?"]
    pub fn solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let mut b = b.clone_owned();
        if self.solve_lower_triangular_mut(&mut b) {
            Some(b)
        } else {
            None
        }
    }

    /// Solves the upper-triangular system L^T x = b, where L is this lower-triangular sparse matrix.
    ///
    /// This method solves a linear system with the transpose of a lower-triangular matrix,
    /// which is equivalent to solving an upper-triangular system. It uses backward
    /// substitution, starting from the last equation and working upward.
    ///
    /// # What This Does
    ///
    /// If L is lower-triangular like this:
    /// ```text
    /// L = [a  0  0]
    ///     [b  c  0]
    ///     [d  e  f]
    /// ```
    ///
    /// Then L^T is upper-triangular:
    /// ```text
    /// L^T = [a  b  d]
    ///       [0  c  e]
    ///       [0  0  f]
    /// ```
    ///
    /// This method solves L^T x = b using backward substitution without explicitly
    /// transposing the matrix (more efficient for sparse matrices).
    ///
    /// # Parameters
    ///
    /// - `b`: The right-hand-side vector or matrix
    ///
    /// # Returns
    ///
    /// - `Some(x)`: The solution x where L^T x = b
    /// - `None`: If the system is singular (zero on diagonal)
    ///
    /// # Performance
    ///
    /// - Time complexity: O(nnz) where nnz is the number of non-zeros
    /// - More efficient than explicitly transposing then solving
    /// - Exploits sparse structure for speed
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Vector3};
    ///
    /// // Create a lower-triangular matrix L:
    /// // [2.0  0    0  ]
    /// // [1.0  3.0  0  ]
    /// // [0.5  1.0  4.0]
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // We want to solve L^T x = b, where L^T is:
    /// // [2.0  1.0  0.5]
    /// // [0    3.0  1.0]
    /// // [0    0    4.0]
    ///
    /// let b = Vector3::new(5.0, 10.0, 8.0);
    ///
    /// // Solve L^T x = b using backward substitution
    /// let x = lower.tr_solve_lower_triangular(&b).expect("Matrix should be non-singular");
    ///
    /// // x[2] = 8.0/4.0 = 2.0
    /// // x[1] = (10.0 - 1.0*2.0)/3.0 = 8.0/3.0
    /// // x[0] = (5.0 - 1.0*8.0/3.0 - 0.5*2.0)/2.0
    /// ```
    ///
    /// ## Use in Cholesky Solver
    ///
    /// This is the second step in solving Ax = b using Cholesky decomposition A = LL^T:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// // Cholesky factor L of a positive definite matrix A
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 1.73, 0.5, 0.87, 1.58];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// // To solve Ax = b where A = LL^T:
    /// // Step 1: Solve Ly = b (forward substitution)
    /// let y = lower.solve_lower_triangular(&b)
    ///     .expect("Cholesky factor is non-singular");
    ///
    /// // Step 2: Solve L^T x = y (backward substitution)
    /// let x = lower.tr_solve_lower_triangular(&y)
    ///     .expect("Cholesky factor is non-singular");
    ///
    /// // Now x is the solution to Ax = b
    /// ```
    ///
    /// ## Multiple Right-Hand Sides
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Matrix3x2};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Two systems to solve simultaneously
    /// let b = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// // Solve L^T X = B (each column independently)
    /// let x = lower.tr_solve_lower_triangular(&b)
    ///     .expect("Matrix should be non-singular");
    ///
    /// assert_eq!(x.ncols(), 2);
    /// ```
    ///
    /// ## Iterative Refinement
    ///
    /// Combining forward and backward solves for iterative refinement:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 2.0, 1.0, 1.0, 2.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// // First pass: forward then backward
    /// let y = lower.solve_lower_triangular(&b).unwrap();
    /// let x = lower.tr_solve_lower_triangular(&y).unwrap();
    ///
    /// // x now solves (LL^T)x = b, equivalent to Ax = b where A = LL^T
    /// ```
    ///
    /// # Why Use This Instead of Transposing?
    ///
    /// For sparse matrices, explicitly computing the transpose is expensive:
    /// - Requires reorganizing the sparse storage structure
    /// - Takes O(nnz) time and space
    /// - This method avoids the transpose entirely while achieving the same result
    ///
    /// # Common Use Cases
    ///
    /// - **Cholesky Solvers**: Second step in solving A = LL^T systems (backward substitution)
    /// - **LU Decomposition**: Upper triangular solve step when L is stored
    /// - **Least Squares**: Normal equations with A^T A
    /// - **Iterative Methods**: Preconditioner applications
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular) - Forward substitution (solve Lx = b)
    /// - [`tr_solve_lower_triangular_mut`](Self::tr_solve_lower_triangular_mut) - In-place version
    /// - [`CsMatrix::transpose`] - Explicit matrix transpose (less efficient for solving)
    #[must_use = "Did you mean to use tr_solve_lower_triangular_mut()?"]
    pub fn tr_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let mut b = b.clone_owned();
        if self.tr_solve_lower_triangular_mut(&mut b) {
            Some(b)
        } else {
            None
        }
    }

    /// Solves the lower-triangular system Lx = b in-place, modifying b to store the solution.
    ///
    /// This is the in-place, more efficient version of [`solve_lower_triangular`](Self::solve_lower_triangular).
    /// Instead of allocating a new vector/matrix for the result, it overwrites the input `b`
    /// with the solution. This saves memory allocation and is faster.
    ///
    /// # Parameters
    ///
    /// - `b`: Input right-hand side, overwritten with solution x on success
    ///
    /// # Returns
    ///
    /// - `true`: Successfully solved, `b` now contains the solution
    /// - `false`: System is singular (zero diagonal), `b` may be partially modified
    ///
    /// # When to Use This vs `solve_lower_triangular`
    ///
    /// Use this method when:
    /// - You don't need to preserve the original right-hand side
    /// - Performance is critical (avoids allocation)
    /// - Solving many systems in a loop (reuse storage)
    ///
    /// Use `solve_lower_triangular` when:
    /// - You need to keep the original `b` unchanged
    /// - Code clarity is more important than performance
    ///
    /// # Examples
    ///
    /// ## Basic In-Place Solve
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // b will be overwritten with solution
    /// let mut b = DVector::from_vec(vec![4.0, 7.0, 10.0]);
    ///
    /// let success = lower.solve_lower_triangular_mut(&mut b);
    /// assert!(success);
    /// // b now contains the solution x
    /// ```
    ///
    /// ## Efficient Loop Over Multiple Systems
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Reuse storage for multiple solves
    /// let mut x = DVector::zeros(3);
    ///
    /// let right_hand_sides = vec![
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    ///     vec![7.0, 8.0, 9.0],
    /// ];
    ///
    /// for rhs in right_hand_sides {
    ///     // Copy new RHS into x
    ///     x.copy_from_slice(&rhs);
    ///
    ///     // Solve in-place (no allocation!)
    ///     if lower.solve_lower_triangular_mut(&mut x) {
    ///         // Process solution in x
    ///     }
    /// }
    /// ```
    ///
    /// ## Checking for Singularity
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// // Matrix with zero diagonal element
    /// let irows = vec![0, 1, 2, 2];
    /// let icols = vec![0, 0, 0, 2];
    /// let vals = vec![2.0, 1.0, 0.5, 4.0]; // Missing (1,1)
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let mut b = DVector::from_vec(vec![4.0, 7.0, 10.0]);
    ///
    /// if !lower.solve_lower_triangular_mut(&mut b) {
    ///     println!("Matrix is singular, cannot solve");
    ///     // b may be partially modified
    /// }
    /// ```
    ///
    /// ## Performance-Critical Inner Loop
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Iterative method that solves many systems
    /// let mut residual = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let mut correction = DVector::zeros(3);
    ///
    /// for _iteration in 0..10 {
    ///     // Copy residual to correction
    ///     correction.copy_from(&residual);
    ///
    ///     // Solve in-place (fast!)
    ///     lower.solve_lower_triangular_mut(&mut correction);
    ///
    ///     // Update solution (not shown)
    ///     // ...
    ///
    ///     // Compute new residual (not shown)
    ///     // ...
    /// }
    /// ```
    ///
    /// ## Handling Multiple Columns
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DMatrix};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Multiple right-hand sides in columns
    /// let mut b = DMatrix::from_row_slice(3, 2, &[
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// ]);
    ///
    /// // Solves all columns in-place
    /// if lower.solve_lower_triangular_mut(&mut b) {
    ///     // b now contains solutions for both systems
    /// }
    /// ```
    ///
    /// # Performance Benefits
    ///
    /// Compared to `solve_lower_triangular`:
    /// - **No allocation**: Reuses existing storage
    /// - **Better cache locality**: Works on existing memory
    /// - **Ideal for loops**: Avoids repeated allocations
    ///
    /// For a system of size n with nnz non-zeros:
    /// - Time: Same O(nnz) for both versions
    /// - Space: This saves O(n) allocation overhead
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular) - Allocating version
    /// - [`tr_solve_lower_triangular_mut`](Self::tr_solve_lower_triangular_mut) - Transposed in-place solve
    pub fn solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let (nrows, ncols) = self.data.shape();
        assert_eq!(nrows.value(), ncols.value(), "The matrix must be square.");
        assert_eq!(nrows.value(), b.len(), "Mismatched matrix dimensions.");

        for j2 in 0..b.ncols() {
            let mut b = b.column_mut(j2);

            for j in 0..ncols.value() {
                let mut column = self.data.column_entries(j);
                let mut diag_found = false;

                for (i, val) in &mut column {
                    if i == j {
                        if val.is_zero() {
                            return false;
                        }

                        b[j] /= val;
                        diag_found = true;
                        break;
                    }
                }

                if !diag_found {
                    return false;
                }

                for (i, val) in column {
                    let bj = b[j].clone();
                    b[i] -= bj * val;
                }
            }
        }

        true
    }

    /// Solves the upper-triangular system L^T x = b in-place, where L is this lower-triangular matrix.
    ///
    /// This is the in-place version of [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular).
    /// It performs backward substitution with the transpose without explicitly computing the
    /// transpose, and overwrites the input `b` with the solution, saving memory allocation.
    ///
    /// # Parameters
    ///
    /// - `b`: Input right-hand side, overwritten with solution x on success
    ///
    /// # Returns
    ///
    /// - `true`: Successfully solved, `b` now contains the solution
    /// - `false`: System is singular (zero diagonal), `b` may be partially modified
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 3.0, 0.5, 1.0, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let mut b = DVector::from_vec(vec![5.0, 10.0, 8.0]);
    ///
    /// // Solve L^T x = b in-place
    /// let success = lower.tr_solve_lower_triangular_mut(&mut b);
    /// assert!(success);
    /// // b now contains the solution
    /// ```
    ///
    /// ## Cholesky Solve Pattern
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// // Cholesky factor L where A = LL^T
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 1.73, 0.5, 0.87, 1.58];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let mut x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// // Solve Ax = b where A = LL^T:
    /// // First: Ly = b (forward)
    /// lower.solve_lower_triangular_mut(&mut x);
    ///
    /// // Second: L^T x = y (backward)
    /// lower.tr_solve_lower_triangular_mut(&mut x);
    ///
    /// // x now solves Ax = original_b
    /// ```
    ///
    /// ## Performance-Critical Loop
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DVector};
    ///
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 2.0, 1.0, 1.0, 2.0];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let mut work_vector = DVector::zeros(3);
    ///
    /// // Iterative method
    /// for _iter in 0..100 {
    ///     // Set up RHS (not shown)
    ///     work_vector.fill(1.0);
    ///
    ///     // Forward solve
    ///     lower.solve_lower_triangular_mut(&mut work_vector);
    ///
    ///     // Backward solve (reuses same storage!)
    ///     lower.tr_solve_lower_triangular_mut(&mut work_vector);
    ///
    ///     // No allocations in the loop!
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`tr_solve_lower_triangular`](Self::tr_solve_lower_triangular) - Allocating version
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut) - Forward substitution in-place
    pub fn tr_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<D, R2>,
    {
        let (nrows, ncols) = self.data.shape();
        assert_eq!(nrows.value(), ncols.value(), "The matrix must be square.");
        assert_eq!(nrows.value(), b.len(), "Mismatched matrix dimensions.");

        for j2 in 0..b.ncols() {
            let mut b = b.column_mut(j2);

            for j in (0..ncols.value()).rev() {
                let mut column = self.data.column_entries(j);
                let mut diag = None;

                for (i, val) in &mut column {
                    if i == j {
                        if val.is_zero() {
                            return false;
                        }

                        diag = Some(val);
                        break;
                    }
                }

                if let Some(diag) = diag {
                    for (i, val) in column {
                        let bi = b[i].clone();
                        b[j] -= val * bi;
                    }

                    b[j] /= diag;
                } else {
                    return false;
                }
            }
        }

        true
    }

    /// Solves the lower-triangular system Lx = b where both L and b are sparse.
    ///
    /// This specialized method solves a lower-triangular system when the right-hand side
    /// is also sparse (has many zeros). It exploits the sparsity of both the matrix and
    /// the right-hand side to compute only the non-zero entries of the solution, making
    /// it much more efficient than dense methods for highly sparse problems.
    ///
    /// # When to Use This
    ///
    /// Use this method instead of [`solve_lower_triangular`](Self::solve_lower_triangular) when:
    /// - The right-hand side `b` is sparse (few non-zeros)
    /// - You expect the solution to also be sparse
    /// - You're working with very large systems where memory matters
    /// - Performance is critical and both inputs are sparse
    ///
    /// # What is Sparse-Sparse Solving?
    ///
    /// When b is sparse with few non-zeros, most entries of x will also be zero (due to
    /// the lower-triangular structure). This method:
    /// 1. Computes the "reach" - which entries of x can be non-zero
    /// 2. Only computes those entries, skipping the rest
    /// 3. Returns a sparse vector containing only non-zero results
    ///
    /// This can be orders of magnitude faster than dense solving for sparse right-hand sides.
    ///
    /// # Parameters
    ///
    /// - `b`: Sparse right-hand side vector (CsVector)
    ///
    /// # Returns
    ///
    /// - `Some(x)`: Sparse solution vector with only non-zero entries
    /// - `None`: If the matrix is singular (zero on diagonal)
    ///
    /// # Performance
    ///
    /// For a sparse b with k non-zeros:
    /// - Time: O(k × average_column_size) - much less than O(n) for dense solve
    /// - Space: O(result_nnz) - only stores non-zero solution entries
    ///
    /// # Examples
    ///
    /// ## Basic Sparse-Sparse Solve
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsVector, DVector};
    ///
    /// // Large lower-triangular matrix
    /// let irows = vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3];
    /// let icols = vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3];
    /// let vals = vec![1.0, 0.5, 2.0, 0.3, 0.4, 3.0, 0.1, 0.2, 0.3, 4.0];
    /// let lower = CsMatrix::<f64>::from_triplet(4, 4, &irows, &icols, &vals);
    ///
    /// // Sparse right-hand side (only 2 non-zeros out of 4)
    /// let b_rows = vec![0, 3];
    /// let b_vals = vec![1.0, 2.0];
    /// let b = CsVector::<f64>::from_triplet(4, &b_rows, &b_vals);
    ///
    /// // Solve exploiting sparsity
    /// let x = lower.solve_lower_triangular_cs(&b).expect("Should succeed");
    ///
    /// // x is also sparse - solution has few non-zeros
    /// assert!(x.len() <= 4); // At most 4 non-zeros
    /// ```
    ///
    /// ## Comparison: Sparse vs Dense Solve
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsVector, DVector};
    ///
    /// let size = 1000;
    /// let mut irows = Vec::new();
    /// let mut icols = Vec::new();
    /// let mut vals = Vec::new();
    ///
    /// // Create a large lower-triangular matrix (e.g., from Cholesky)
    /// for i in 0..size {
    ///     irows.push(i);
    ///     icols.push(i);
    ///     vals.push(1.0 + i as f64);
    /// }
    /// let lower = CsMatrix::<f64>::from_triplet(size, size, &irows, &icols, &vals);
    ///
    /// // Sparse RHS: only 5 non-zeros out of 1000
    /// let b_rows = vec![0, 100, 200, 300, 400];
    /// let b_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let b_sparse = CsVector::<f64>::from_triplet(size, &b_rows, &b_vals);
    ///
    /// // Dense solve would process all 1000 entries
    /// let b_dense = DVector::from_vec(vec![1.0; size]);
    /// // let x_dense = lower.solve_lower_triangular(&b_dense);
    ///
    /// // Sparse solve only processes ~5 entries and their dependencies
    /// let x_sparse = lower.solve_lower_triangular_cs(&b_sparse);
    /// // Much faster for sparse b!
    /// ```
    ///
    /// ## Use in Sparse Cholesky
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsVector};
    ///
    /// // Cholesky factor from sparse positive definite matrix
    /// let irows = vec![0, 1, 1, 2, 2, 2];
    /// let icols = vec![0, 0, 1, 0, 1, 2];
    /// let vals = vec![2.0, 1.0, 1.73, 0.5, 0.87, 1.58];
    /// let lower = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Sparse right-hand side
    /// let b_rows = vec![0, 2];
    /// let b_vals = vec![1.0, 3.0];
    /// let b = CsVector::<f64>::from_triplet(3, &b_rows, &b_vals);
    ///
    /// // Forward solve (sparse)
    /// let y = lower.solve_lower_triangular_cs(&b)
    ///     .expect("Cholesky factor is non-singular");
    ///
    /// // Note: For backward solve with sparse y, you'd need tr_solve_lower_triangular_cs
    /// // (which isn't shown here but would follow similar pattern)
    /// ```
    ///
    /// ## Exploiting Problem Structure
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsVector};
    ///
    /// // System representing a physical problem with localized forcing
    /// let n = 100;
    /// let mut irows = Vec::new();
    /// let mut icols = Vec::new();
    /// let mut vals = Vec::new();
    ///
    /// // Sparse lower-triangular matrix (e.g., from discretization)
    /// for i in 0..n {
    ///     irows.push(i);
    ///     icols.push(i);
    ///     vals.push(2.0);
    ///     if i > 0 {
    ///         irows.push(i);
    ///         icols.push(i - 1);
    ///         vals.push(-1.0);
    ///     }
    /// }
    /// let lower = CsMatrix::<f64>::from_triplet(n, n, &irows, &icols, &vals);
    ///
    /// // Localized forcing (sparse RHS) - only affects a few points
    /// let b_rows = vec![10];
    /// let b_vals = vec![1.0];
    /// let b = CsVector::<f64>::from_triplet(n, &b_rows, &b_vals);
    ///
    /// // Sparse solve - solution propagates from forced point
    /// let x = lower.solve_lower_triangular_cs(&b)
    ///     .expect("Matrix is non-singular");
    ///
    /// // x will have non-zeros only where the forcing propagated
    /// println!("Solution sparsity: {}/{}", x.len(), n);
    /// ```
    ///
    /// # Algorithm Details
    ///
    /// The sparse triangular solve uses a "reachability" algorithm:
    /// 1. Starting from non-zero entries in b, determines which entries in x can be non-zero
    /// 2. Performs forward substitution only on those entries
    /// 3. Returns a sparse vector with sorted indices
    ///
    /// This is much more efficient than computing all entries when b is sparse.
    ///
    /// # Common Use Cases
    ///
    /// - **Sparse Linear Systems**: Both matrix and RHS are sparse
    /// - **Structural Analysis**: Localized loads (sparse b) on structures (sparse matrix)
    /// - **Circuit Simulation**: Sparse forcing terms
    /// - **Iterative Methods**: Sparse preconditioner applications
    /// - **Sparse Cholesky/LU**: Forward/backward substitution with sparse vectors
    ///
    /// # See Also
    ///
    /// - [`solve_lower_triangular`](Self::solve_lower_triangular) - Dense RHS version
    /// - [`solve_lower_triangular_mut`](Self::solve_lower_triangular_mut) - In-place dense version
    #[must_use]
    pub fn solve_lower_triangular_cs<D2: Dim, S2>(
        &self,
        b: &CsVector<T, D2, S2>,
    ) -> Option<CsVector<T, D2>>
    where
        S2: CsStorage<T, D2>,
        DefaultAllocator: Allocator<D> + Allocator<D2>,
        ShapeConstraint: SameNumberOfRows<D, D2>,
    {
        let mut reach = Vec::new();
        // We don't compute a postordered reach here because it will be sorted after anyway.
        self.lower_triangular_reach(b, &mut reach);
        // We sort the reach so the result matrix has sorted indices.
        reach.sort_unstable();
        let mut workspace = Matrix::zeros_generic(b.data.shape().0, Const::<1>);

        for i in reach.iter().cloned() {
            workspace[i] = T::zero();
        }

        for (i, val) in b.data.column_entries(0) {
            workspace[i] = val;
        }

        for j in reach.iter().cloned() {
            let mut column = self.data.column_entries(j);
            let mut diag_found = false;

            for (i, val) in &mut column {
                if i == j {
                    if val.is_zero() {
                        break;
                    }

                    workspace[j] /= val;
                    diag_found = true;
                    break;
                }
            }

            if !diag_found {
                return None;
            }

            for (i, val) in column {
                let wj = workspace[j].clone();
                workspace[i] -= wj * val;
            }
        }

        // Copy the result into a sparse vector.
        let mut result =
            CsVector::new_uninitialized_generic(b.data.shape().0, Const::<1>, reach.len());

        for (i, val) in reach.iter().zip(result.data.vals.iter_mut()) {
            *val = workspace[*i].clone();
        }

        result.data.i = reach;
        Some(result)
    }

    /*
    // Computes the reachable, post-ordered, nodes from `b`.
    fn lower_triangular_reach_postordered<D2: Dim, S2>(
        &self,
        b: &CsVector<T, D2, S2>,
        xi: &mut Vec<usize>,
    ) where
        S2: CsStorage<T, D2>,
        DefaultAllocator: Allocator<D>,
    {
        let mut visited = OVector::repeat_generic(self.data.shape().1, U1, false);
        let mut stack = Vec::new();

        for i in b.data.column_range(0) {
            let row_index = b.data.row_index(i);

            if !visited[row_index] {
                let rng = self.data.column_range(row_index);
                stack.push((row_index, rng));
                self.lower_triangular_dfs(visited.as_mut_slice(), &mut stack, xi);
            }
        }
    }

    fn lower_triangular_dfs(
        &self,
        visited: &mut [bool],
        stack: &mut Vec<(usize, Range<usize>)>,
        xi: &mut Vec<usize>,
    )
    {
        'recursion: while let Some((j, rng)) = stack.pop() {
            visited[j] = true;

            for i in rng.clone() {
                let row_id = self.data.row_index(i);
                if row_id > j && !visited[row_id] {
                    stack.push((j, (i + 1)..rng.end));
                    stack.push((row_id, self.data.column_range(row_id)));
                    continue 'recursion;
                }
            }

            xi.push(j)
        }
    }
    */

    // Computes the nodes reachable from `b` in an arbitrary order.
    fn lower_triangular_reach<D2: Dim, S2>(&self, b: &CsVector<T, D2, S2>, xi: &mut Vec<usize>)
    where
        S2: CsStorage<T, D2>,
        DefaultAllocator: Allocator<D>,
    {
        let mut visited = OVector::repeat_generic(self.data.shape().1, Const::<1>, false);
        let mut stack = Vec::new();

        for irow in b.data.column_row_indices(0) {
            self.lower_triangular_bfs(irow, visited.as_mut_slice(), &mut stack, xi);
        }
    }

    fn lower_triangular_bfs(
        &self,
        start: usize,
        visited: &mut [bool],
        stack: &mut Vec<usize>,
        xi: &mut Vec<usize>,
    ) {
        if !visited[start] {
            stack.clear();
            stack.push(start);
            xi.push(start);
            visited[start] = true;

            while let Some(j) = stack.pop() {
                for irow in self.data.column_row_indices(j) {
                    if irow > j && !visited[irow] {
                        stack.push(irow);
                        xi.push(irow);
                        visited[irow] = true;
                    }
                }
            }
        }
    }
}
