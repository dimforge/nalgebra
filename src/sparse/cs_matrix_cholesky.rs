use std::iter;
use std::mem;

use crate::allocator::Allocator;
use crate::sparse::{CsMatrix, CsStorage, CsStorageIter, CsStorageIterMut, CsVecStorage};
use crate::{Const, DefaultAllocator, Dim, Matrix, OVector, RealField};

/// The cholesky decomposition of a column compressed sparse matrix.
pub struct CsCholesky<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    // Non-zero pattern of the original matrix upper-triangular part.
    // Unlike the original matrix, the `original_p` array does contain the last sentinel value
    // equal to `original_i.len()` at the end.
    original_p: Vec<usize>,
    original_i: Vec<usize>,
    // Decomposition result.
    l: CsMatrix<T, D, D>,
    // Used only for the pattern.
    // TODO: store only the nonzero pattern instead.
    u: CsMatrix<T, D, D>,
    ok: bool,
    // Workspaces.
    work_x: OVector<T, D>,
    work_c: OVector<usize, D>,
}

impl<T: RealField, D: Dim> CsCholesky<T, D>
where
    DefaultAllocator: Allocator<D> + Allocator<D>,
{
    /// Computes the Cholesky decomposition of a sparse symmetric positive-definite matrix.
    ///
    /// The Cholesky decomposition factors a symmetric positive-definite matrix A into
    /// the form A = L L^T, where L is a lower-triangular matrix. For sparse matrices,
    /// this is particularly valuable because L often remains sparse even though A^(-1)
    /// would be dense.
    ///
    /// # Requirements
    ///
    /// The input matrix must be:
    /// - **Square**: Same number of rows and columns
    /// - **Symmetric**: A = A^T (only upper or lower triangle is used)
    /// - **Positive-Definite**: All eigenvalues are positive
    ///
    /// If these conditions are not met, the decomposition will fail (check with `l()`).
    ///
    /// # Algorithm
    ///
    /// This uses a left-looking sparse Cholesky factorization algorithm:
    /// 1. Symbolic analysis determines the non-zero pattern of L
    /// 2. Numerical factorization computes the values of L
    ///
    /// # Examples
    ///
    /// ## Basic Decomposition
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Create a symmetric positive-definite matrix:
    /// // [4.0  2.0  0.0]
    /// // [2.0  5.0  1.0]
    /// // [0.0  1.0  3.0]
    /// let triplets = vec![
    ///     (0, 0, 4.0), (0, 1, 2.0),
    ///     (1, 0, 2.0), (1, 1, 5.0), (1, 2, 1.0),
    ///     (2, 1, 1.0), (2, 2, 3.0),
    /// ];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
    ///
    /// // Compute Cholesky decomposition
    /// let chol = CsCholesky::new(&m);
    ///
    /// // Check if decomposition succeeded
    /// assert!(chol.l().is_some());
    /// ```
    ///
    /// ## Solving Linear Systems
    ///
    /// Cholesky decomposition is efficient for solving A x = b:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, DVector, Dyn};
    ///
    /// // Symmetric positive-definite matrix (tridiagonal)
    /// let mut triplets = vec![];
    /// for i in 0..5 {
    ///     triplets.push((i, i, 2.0));
    ///     if i > 0 {
    ///         triplets.push((i, i-1, -1.0));
    ///         triplets.push((i-1, i, -1.0));
    ///     }
    /// }
    /// let a = CsMatrix::<f64, Dyn, Dyn>::from_triplet(5, 5, &triplets);
    ///
    /// let chol = CsCholesky::new(&a);
    /// assert!(chol.l().is_some());
    ///
    /// // Now can efficiently solve A x = b for many right-hand sides
    /// ```
    ///
    /// ## Finite Element Analysis
    ///
    /// Typical use in structural analysis:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Stiffness matrix from finite element assembly (symmetric positive-definite)
    /// let k = CsMatrix::<f64, Dyn, Dyn>::from_triplet(4, 4, &[
    ///     (0, 0, 3.0), (0, 1, -1.0),
    ///     (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
    ///     (2, 1, -1.0), (2, 2, 4.0), (2, 3, -1.0),
    ///     (3, 2, -1.0), (3, 3, 3.0),
    /// ]);
    ///
    /// // Factor once, solve many times (for different load cases)
    /// let chol = CsCholesky::new(&k);
    /// assert!(chol.l().is_some());
    /// ```
    ///
    /// # Performance
    ///
    /// - Symbolic analysis: O(n^2) in worst case, often much better for sparse matrices
    /// - Numerical factorization: Depends on sparsity pattern
    /// - For many sparse structures (e.g., arising from 2D/3D meshes), remains sparse
    ///
    /// # Applications
    ///
    /// - **Linear System Solving**: Fast solution of A x = b
    /// - **Least Squares**: Solving normal equations A^T A x = A^T b
    /// - **Optimization**: Newton methods with positive-definite Hessians
    /// - **Finite Element Analysis**: Structural mechanics, heat transfer
    /// - **Statistics**: Computing with covariance matrices
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::new_symbolic`] - Perform only symbolic analysis
    /// - [`CsCholesky::l`] - Access the lower-triangular factor
    /// - [`CsCholesky::decompose_left_looking`] - Refactor with new values
    pub fn new(m: &CsMatrix<T, D, D>) -> Self {
        let mut me = Self::new_symbolic(m);
        let _ = me.decompose_left_looking(&m.data.vals);
        me
    }
    /// Performs symbolic analysis to determine the structure of the Cholesky factor.
    ///
    /// This method analyzes the sparsity pattern of the input matrix to determine
    /// which entries of the lower-triangular factor L will be non-zero. Importantly,
    /// it does NOT use the numerical values - only the positions of non-zeros matter.
    ///
    /// # Why Symbolic Analysis?
    ///
    /// Separating symbolic and numeric phases allows efficient refactorization:
    /// - Symbolic phase: Expensive, determines the non-zero pattern of L
    /// - Numeric phase: Cheaper, fills in the values of L
    ///
    /// If you need to factor many matrices with the same sparsity pattern (common
    /// in time-stepping, optimization, or parametric studies), you can perform
    /// symbolic analysis once and reuse it.
    ///
    /// # Algorithm
    ///
    /// Uses elimination tree and reachability analysis to predict fill-in
    /// (new non-zeros that appear in L even though they're zero in A).
    ///
    /// # Examples
    ///
    /// ## Symbolic Analysis and Reuse
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Create a tridiagonal matrix structure
    /// let mut triplets1 = vec![];
    /// for i in 0..100 {
    ///     triplets1.push((i, i, 2.0));
    ///     if i > 0 {
    ///         triplets1.push((i, i-1, -1.0));
    ///         triplets1.push((i-1, i, -1.0));
    ///     }
    /// }
    /// let m1 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(100, 100, &triplets1);
    ///
    /// // Perform symbolic analysis once
    /// let mut chol = CsCholesky::new_symbolic(&m1);
    ///
    /// // Now factorize numerically
    /// let success = chol.decompose_left_looking(&m1.data.values());
    /// assert!(success);
    /// ```
    ///
    /// ## Time-Stepping Problems
    ///
    /// Many time integration schemes solve systems with the same structure:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Mass matrix (constant sparsity pattern)
    /// let mass_pattern = vec![
    ///     (0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0),
    /// ];
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &mass_pattern);
    ///
    /// // Perform symbolic analysis once
    /// let mut chol = CsCholesky::new_symbolic(&m);
    ///
    /// // Time stepping: each step has different values but same structure
    /// for _time_step in 0..10 {
    ///     // Create effective stiffness: K_eff = M + dt * K
    ///     // (would have same sparsity pattern as mass matrix in this example)
    ///     let k_eff_values = vec![2.0, 2.5, 2.1]; // New values each step
    ///
    ///     // Reuse symbolic structure, just update values
    ///     chol.decompose_left_looking(&k_eff_values);
    ///
    ///     // Solve system for this time step
    ///     // ... (solution code)
    /// }
    /// ```
    ///
    /// ## Parametric Studies
    ///
    /// When studying how solutions vary with parameters:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Initial stiffness matrix structure
    /// let k0 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 4.0), (0, 1, -2.0),
    ///     (1, 0, -2.0), (1, 1, 4.0), (1, 2, -2.0),
    ///     (2, 1, -2.0), (2, 2, 4.0),
    /// ]);
    ///
    /// // Symbolic analysis once
    /// let mut chol = CsCholesky::new_symbolic(&k0);
    ///
    /// // Study different material parameters
    /// for stiffness_param in [1.0, 2.0, 3.0, 4.0] {
    ///     // Create new values (same positions, different magnitudes)
    ///     let values: Vec<f64> = k0.data.values()
    ///         .iter()
    ///         .map(|v| v * stiffness_param)
    ///         .collect();
    ///
    ///     // Refactor with new parameter values
    ///     chol.decompose_left_looking(&values);
    ///
    ///     // Solve and analyze results
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - Symbolic analysis: O(n^2) worst case, often O(n) for typical sparse patterns
    /// - Only depends on structure, not values
    /// - Much faster than full factorization for large matrices
    ///
    /// # Applications
    ///
    /// - **Time Integration**: Same structure each time step
    /// - **Parametric Studies**: Varying parameters, fixed structure
    /// - **Optimization**: Many iterations with similar Hessian structure
    /// - **Sensitivity Analysis**: Repeated solves with perturbed matrices
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::new`] - Combined symbolic + numeric factorization
    /// - [`CsCholesky::decompose_left_looking`] - Numeric refactorization
    /// - [`CsCholesky::decompose_up_looking`] - Alternative numeric algorithm
    pub fn new_symbolic(m: &CsMatrix<T, D, D>) -> Self {
        assert!(
            m.is_square(),
            "The matrix `m` must be square to compute its elimination tree."
        );

        let (l, u) = Self::nonzero_pattern(m);

        // Workspaces.
        let work_x = Matrix::zeros_generic(m.data.shape().0, Const::<1>);
        let work_c = Matrix::zeros_generic(m.data.shape().1, Const::<1>);
        let mut original_p = m.data.p.as_slice().to_vec();
        original_p.push(m.data.i.len());

        CsCholesky {
            original_p,
            original_i: m.data.i.clone(),
            l,
            u,
            ok: false,
            work_x,
            work_c,
        }
    }

    /// Returns a reference to the lower-triangular Cholesky factor L.
    ///
    /// If the decomposition was successful, this returns `Some(&L)` where L satisfies
    /// A = L L^T. If the decomposition failed (matrix not positive-definite), returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Positive-definite matrix
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
    ///     (0, 0, 4.0), (0, 1, 2.0),
    ///     (1, 0, 2.0), (1, 1, 3.0),
    /// ]);
    ///
    /// let chol = CsCholesky::new(&m);
    ///
    /// if let Some(l) = chol.l() {
    ///     println!("Decomposition succeeded");
    ///     println!("L has {} non-zeros", l.len());
    /// } else {
    ///     println!("Decomposition failed - matrix not positive-definite");
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::unwrap_l`] - Extracts L by consuming self
    #[must_use]
    pub fn l(&self) -> Option<&CsMatrix<T, D, D>> {
        if self.ok { Some(&self.l) } else { None }
    }

    /// Extracts and returns the lower-triangular Cholesky factor L by consuming self.
    ///
    /// If the decomposition was successful, this returns `Some(L)` where L satisfies
    /// A = L L^T. If the decomposition failed, returns `None`.
    ///
    /// This method is useful when you want to take ownership of L rather than just
    /// borrowing it.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 4.0),
    ///     (1, 1, 5.0),
    ///     (2, 2, 6.0),
    /// ]);
    ///
    /// let chol = CsCholesky::new(&m);
    ///
    /// // Take ownership of L
    /// if let Some(l) = chol.unwrap_l() {
    ///     // Can now use l without borrowing from chol
    ///     let l_squared = &l * &l.transpose();
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::l`] - Borrows L without consuming self
    pub fn unwrap_l(self) -> Option<CsMatrix<T, D, D>> {
        if self.ok { Some(self.l) } else { None }
    }

    /// Performs numerical Cholesky factorization using the left-looking algorithm.
    ///
    /// This method computes the numerical values of the Cholesky factor L for a matrix
    /// with the same sparsity pattern as the one used in symbolic analysis, but with
    /// different numerical values. This is efficient for refactorization when you need
    /// to factor many matrices with the same structure.
    ///
    /// # Parameters
    ///
    /// - `values`: Array of non-zero values in the same order as the original matrix
    ///
    /// # Returns
    ///
    /// - `true` if factorization succeeded (matrix is positive-definite)
    /// - `false` if factorization failed (matrix is not positive-definite or has numerical issues)
    ///
    /// # Left-Looking Algorithm
    ///
    /// The left-looking algorithm processes columns from left to right. For each column j:
    /// 1. Gathers contributions from all previous columns that affect column j
    /// 2. Computes the new entries of column j of L
    ///
    /// This approach is cache-friendly and works well for many sparse patterns.
    ///
    /// # Examples
    ///
    /// ## Basic Refactorization
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Original matrix
    /// let m1 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0),
    /// ]);
    ///
    /// // Symbolic analysis
    /// let mut chol = CsCholesky::new_symbolic(&m1);
    ///
    /// // First factorization
    /// let success = chol.decompose_left_looking(&m1.data.values());
    /// assert!(success);
    ///
    /// // New values, same structure
    /// let new_values = vec![9.0, 16.0, 25.0]; // Different diagonal values
    /// let success2 = chol.decompose_left_looking(&new_values);
    /// assert!(success2);
    /// ```
    ///
    /// ## Time-Dependent Problems
    ///
    /// Solving heat equation with implicit time stepping:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Spatial discretization matrix (Laplacian)
    /// let laplacian = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 2.0), (0, 1, -1.0),
    ///     (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
    ///     (2, 1, -1.0), (2, 2, 2.0),
    /// ]);
    ///
    /// // Initial symbolic analysis
    /// let mut chol = CsCholesky::new_symbolic(&laplacian);
    ///
    /// let dt = 0.01; // time step
    /// for step in 0..10 {
    ///     // System matrix: I + dt * Laplacian
    ///     // (in practice, would compute this more efficiently)
    ///     let sys_values: Vec<f64> = laplacian.data.values()
    ///         .iter()
    ///         .enumerate()
    ///         .map(|(i, &v)| {
    ///             // Add identity contribution to diagonal
    ///             if i % 3 == 0 { v + 1.0/dt } else { v }
    ///         })
    ///         .collect();
    ///
    ///     // Refactor for this time step
    ///     let ok = chol.decompose_left_looking(&sys_values);
    ///     assert!(ok);
    ///
    ///     // Solve system for this step...
    /// }
    /// ```
    ///
    /// ## Newton's Method
    ///
    /// Nonlinear solver with changing Jacobian:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// // Initial Jacobian structure
    /// let j0 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0),
    /// ]);
    ///
    /// let mut chol = CsCholesky::new_symbolic(&j0);
    ///
    /// // Newton iterations
    /// for _iter in 0..5 {
    ///     // Compute new Jacobian values at current iterate
    ///     // (same sparsity, different values)
    ///     let jacobian_values = vec![1.5, 2.0, 1.8];
    ///
    ///     if chol.decompose_left_looking(&jacobian_values) {
    ///         // Solve Newton step...
    ///     } else {
    ///         println!("Jacobian not positive-definite, may need regularization");
    ///         break;
    ///     }
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - Much faster than full factorization (symbolic + numeric)
    /// - Efficiency depends on sparsity pattern
    /// - For matrices from 2D/3D meshes, typically very efficient
    ///
    /// # Failure Cases
    ///
    /// Returns `false` if:
    /// - Matrix is not positive-definite (negative or zero pivot encountered)
    /// - Numerical issues (very small pivots, near-singular matrix)
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::decompose_up_looking`] - Alternative algorithm
    /// - [`CsCholesky::new_symbolic`] - Required before calling this method
    /// - [`CsCholesky::l`] - Access the computed factor
    pub fn decompose_left_looking(&mut self, values: &[T]) -> bool {
        assert!(
            values.len() >= self.original_i.len(),
            "The set of values is too small."
        );

        let n = self.l.nrows();

        // Reset `work_c` to the column pointers of `l`.
        self.work_c.copy_from(&self.l.data.p);

        unsafe {
            for k in 0..n {
                // Scatter the k-th column of the original matrix with the values provided.
                let range_k =
                    *self.original_p.get_unchecked(k)..*self.original_p.get_unchecked(k + 1);

                *self.work_x.vget_unchecked_mut(k) = T::zero();
                for p in range_k.clone() {
                    let irow = *self.original_i.get_unchecked(p);

                    if irow >= k {
                        *self.work_x.vget_unchecked_mut(irow) = values.get_unchecked(p).clone();
                    }
                }

                for j in self.u.data.column_row_indices(k) {
                    let factor = -self
                        .l
                        .data
                        .vals
                        .get_unchecked(*self.work_c.vget_unchecked(j))
                        .clone();
                    *self.work_c.vget_unchecked_mut(j) += 1;

                    if j < k {
                        for (z, val) in self.l.data.column_entries(j) {
                            if z >= k {
                                *self.work_x.vget_unchecked_mut(z) += val * factor.clone();
                            }
                        }
                    }
                }

                let diag = self.work_x.vget_unchecked(k).clone();

                if diag > T::zero() {
                    let denom = diag.sqrt();
                    *self
                        .l
                        .data
                        .vals
                        .get_unchecked_mut(*self.l.data.p.vget_unchecked(k)) = denom.clone();

                    for (p, val) in self.l.data.column_entries_mut(k) {
                        *val = self.work_x.vget_unchecked(p).clone() / denom.clone();
                        *self.work_x.vget_unchecked_mut(p) = T::zero();
                    }
                } else {
                    self.ok = false;
                    return false;
                }
            }
        }

        self.ok = true;
        true
    }

    /// Performs numerical Cholesky factorization using the up-looking algorithm.
    ///
    /// This is an alternative algorithm to `decompose_left_looking` that computes the
    /// Cholesky factor L. While mathematically equivalent, it processes the matrix
    /// differently and may have different performance characteristics depending on the
    /// sparsity pattern.
    ///
    /// # Parameters
    ///
    /// - `values`: Array of non-zero values in the same order as the original matrix
    ///
    /// # Returns
    ///
    /// - `true` if factorization succeeded (matrix is positive-definite)
    /// - `false` if factorization failed (matrix is not positive-definite)
    ///
    /// # Up-Looking Algorithm
    ///
    /// The up-looking algorithm also processes columns left to right, but for each column j:
    /// 1. Performs a triangular solve with the already-computed part of L
    /// 2. Updates the diagonal entry
    ///
    /// This can be more efficient for certain sparsity patterns, particularly those
    /// arising from nested dissection orderings.
    ///
    /// # Algorithm Choice
    ///
    /// Choose between `decompose_left_looking` and `decompose_up_looking` based on:
    /// - **Left-looking**: Better for natural orderings, simpler access patterns
    /// - **Up-looking**: Can be faster for sophisticated orderings, more triangular solves
    ///
    /// For most applications, left-looking is a good default choice.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
    ///     (0, 0, 4.0),
    ///     (1, 1, 5.0),
    ///     (2, 2, 6.0),
    /// ]);
    ///
    /// let mut chol = CsCholesky::new_symbolic(&m);
    ///
    /// // Use up-looking algorithm
    /// let success = chol.decompose_up_looking(&m.data.values());
    /// assert!(success);
    /// ```
    ///
    /// ## Comparing Algorithms
    ///
    /// Both algorithms produce the same result:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, CsCholesky, Dyn};
    ///
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(4, 4, &[
    ///     (0, 0, 4.0), (0, 1, -1.0),
    ///     (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
    ///     (2, 1, -1.0), (2, 2, 4.0), (2, 3, -1.0),
    ///     (3, 2, -1.0), (3, 3, 4.0),
    /// ]);
    ///
    /// // Left-looking
    /// let mut chol1 = CsCholesky::new_symbolic(&m);
    /// chol1.decompose_left_looking(&m.data.values());
    ///
    /// // Up-looking
    /// let mut chol2 = CsCholesky::new_symbolic(&m);
    /// chol2.decompose_up_looking(&m.data.values());
    ///
    /// // Both produce valid Cholesky factors
    /// assert!(chol1.l().is_some());
    /// assert!(chol2.l().is_some());
    /// ```
    ///
    /// # Performance
    ///
    /// - Complexity similar to left-looking for most matrices
    /// - Can be faster with reordered matrices (nested dissection, minimum degree)
    /// - More opportunities for BLAS-style optimizations in dense subblocks
    ///
    /// # See Also
    ///
    /// - [`CsCholesky::decompose_left_looking`] - Alternative algorithm
    /// - [`CsCholesky::new_symbolic`] - Required before calling this method
    pub fn decompose_up_looking(&mut self, values: &[T]) -> bool {
        assert!(
            values.len() >= self.original_i.len(),
            "The set of values is too small."
        );

        // Reset `work_c` to the column pointers of `l`.
        self.work_c.copy_from(&self.l.data.p);

        // Perform the decomposition.
        for k in 0..self.l.nrows() {
            unsafe {
                // Scatter the k-th column of the original matrix with the values provided.
                let column_range =
                    *self.original_p.get_unchecked(k)..*self.original_p.get_unchecked(k + 1);

                *self.work_x.vget_unchecked_mut(k) = T::zero();
                for p in column_range.clone() {
                    let irow = *self.original_i.get_unchecked(p);

                    if irow <= k {
                        *self.work_x.vget_unchecked_mut(irow) = values.get_unchecked(p).clone();
                    }
                }

                let mut diag = self.work_x.vget_unchecked(k).clone();
                *self.work_x.vget_unchecked_mut(k) = T::zero();

                // Triangular solve.
                for irow in self.u.data.column_row_indices(k) {
                    if irow >= k {
                        continue;
                    }

                    let lki = self.work_x.vget_unchecked(irow).clone()
                        / self
                            .l
                            .data
                            .vals
                            .get_unchecked(*self.l.data.p.vget_unchecked(irow))
                            .clone();
                    *self.work_x.vget_unchecked_mut(irow) = T::zero();

                    for p in
                        *self.l.data.p.vget_unchecked(irow) + 1..*self.work_c.vget_unchecked(irow)
                    {
                        *self
                            .work_x
                            .vget_unchecked_mut(*self.l.data.i.get_unchecked(p)) -=
                            self.l.data.vals.get_unchecked(p).clone() * lki.clone();
                    }

                    diag -= lki.clone() * lki.clone();
                    let p = *self.work_c.vget_unchecked(irow);
                    *self.work_c.vget_unchecked_mut(irow) += 1;
                    *self.l.data.i.get_unchecked_mut(p) = k;
                    *self.l.data.vals.get_unchecked_mut(p) = lki;
                }

                if diag <= T::zero() {
                    self.ok = false;
                    return false;
                }

                // Deal with the diagonal element.
                let p = *self.work_c.vget_unchecked(k);
                *self.work_c.vget_unchecked_mut(k) += 1;
                *self.l.data.i.get_unchecked_mut(p) = k;
                *self.l.data.vals.get_unchecked_mut(p) = diag.sqrt();
            }
        }

        self.ok = true;
        true
    }

    fn elimination_tree<S: CsStorage<T, D, D>>(m: &CsMatrix<T, D, D, S>) -> Vec<usize> {
        let nrows = m.nrows();
        let mut forest: Vec<_> = iter::repeat(usize::MAX).take(nrows).collect();
        let mut ancestor: Vec<_> = iter::repeat(usize::MAX).take(nrows).collect();

        for k in 0..nrows {
            for irow in m.data.column_row_indices(k) {
                let mut i = irow;

                while i < k {
                    let i_ancestor = ancestor[i];
                    ancestor[i] = k;

                    if i_ancestor == usize::MAX {
                        forest[i] = k;
                        break;
                    }

                    i = i_ancestor;
                }
            }
        }

        forest
    }

    fn reach<S: CsStorage<T, D, D>>(
        m: &CsMatrix<T, D, D, S>,
        j: usize,
        max_j: usize,
        tree: &[usize],
        marks: &mut Vec<bool>,
        out: &mut Vec<usize>,
    ) {
        marks.clear();
        marks.resize(tree.len(), false);

        // TODO: avoid all those allocations.
        let mut tmp = Vec::new();
        let mut res = Vec::new();

        for irow in m.data.column_row_indices(j) {
            let mut curr = irow;
            while curr != usize::MAX && curr <= max_j && !marks[curr] {
                marks[curr] = true;
                tmp.push(curr);
                curr = tree[curr];
            }

            tmp.append(&mut res);
            mem::swap(&mut tmp, &mut res);
        }

        out.append(&mut res);
    }

    fn nonzero_pattern<S: CsStorage<T, D, D>>(
        m: &CsMatrix<T, D, D, S>,
    ) -> (CsMatrix<T, D, D>, CsMatrix<T, D, D>) {
        let etree = Self::elimination_tree(m);
        let (nrows, ncols) = m.data.shape();
        let mut rows = Vec::with_capacity(m.len());
        let mut cols = Matrix::zeros_generic(m.data.shape().0, Const::<1>);
        let mut marks = Vec::new();

        // NOTE: the following will actually compute the non-zero pattern of
        // the transpose of l.
        for i in 0..nrows.value() {
            cols[i] = rows.len();
            Self::reach(m, i, i, &etree, &mut marks, &mut rows);
        }

        let mut vals = Vec::with_capacity(rows.len());
        unsafe {
            vals.set_len(rows.len());
        }
        vals.shrink_to_fit();

        let data = CsVecStorage {
            shape: (nrows, ncols),
            p: cols,
            i: rows,
            vals,
        };

        let u = CsMatrix::from_data(data);
        // XXX: avoid this transpose.
        let l = u.transpose();

        (l, u)
    }

    /*
     *
     * NOTE: All the following methods are untested and currently unused.
     *
     *
    fn column_counts<S: CsStorage<T, D, D>>(
        m: &CsMatrix<T, D, D, S>,
        tree: &[usize],
    ) -> Vec<usize> {
        let len = m.data.shape().0.value();
        let mut counts: Vec<_> = iter::repeat(0).take(len).collect();
        let mut reach = Vec::new();
        let mut marks = Vec::new();

        for i in 0..len {
            Self::reach(m, i, i, tree, &mut marks, &mut reach);

            for j in reach.drain(..) {
                counts[j] += 1;
            }
        }

        counts
    }

    fn tree_postorder(tree: &[usize]) -> Vec<usize> {
        // TODO: avoid all those allocations?
        let mut first_child: Vec<_> = iter::repeat(usize::MAX).take(tree.len()).collect();
        let mut other_children: Vec<_> =
            iter::repeat(usize::MAX).take(tree.len()).collect();

        // Build the children list from the parent list.
        // The set of children of the node `i` is given by the linked list
        // starting at `first_child[i]`. The nodes of this list are then:
        // { first_child[i], other_children[first_child[i]], other_children[other_children[first_child[i]], ... }
        for (i, parent) in tree.iter().enumerate() {
            if *parent != usize::MAX {
                let brother = first_child[*parent];
                first_child[*parent] = i;
                other_children[i] = brother;
            }
        }

        let mut stack = Vec::with_capacity(tree.len());
        let mut postorder = Vec::with_capacity(tree.len());

        for (i, node) in tree.iter().enumerate() {
            if *node == usize::MAX {
                Self::dfs(
                    i,
                    &mut first_child,
                    &other_children,
                    &mut stack,
                    &mut postorder,
                )
            }
        }

        postorder
    }

    fn dfs(
        i: usize,
        first_child: &mut [usize],
        other_children: &[usize],
        stack: &mut Vec<usize>,
        result: &mut Vec<usize>,
    ) {
        stack.clear();
        stack.push(i);

        while let Some(n) = stack.pop() {
            let child = first_child[n];

            if child == usize::MAX {
                // No children left.
                result.push(n);
            } else {
                stack.push(n);
                stack.push(child);
                first_child[n] = other_children[child];
            }
        }
    }
    */
}
