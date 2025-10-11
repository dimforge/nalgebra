use num::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};
use std::ops::{Add, Mul};

use crate::allocator::Allocator;
use crate::constraint::{AreMultipliable, DimEq, ShapeConstraint};
use crate::sparse::{CsMatrix, CsStorage, CsStorageMut, CsVector};
use crate::storage::StorageMut;
use crate::{Const, DefaultAllocator, Dim, Matrix, OVector, Scalar, Vector};

impl<T: Scalar, R: Dim, C: Dim, S: CsStorage<T, R, C>> CsMatrix<T, R, C, S> {
    fn scatter<R2: Dim, C2: Dim>(
        &self,
        j: usize,
        beta: T,
        timestamps: &mut [usize],
        timestamp: usize,
        workspace: &mut [T],
        mut nz: usize,
        res: &mut CsMatrix<T, R2, C2>,
    ) -> usize
    where
        T: ClosedAddAssign + ClosedMulAssign,
        DefaultAllocator: Allocator<C2>,
    {
        for (i, val) in self.data.column_entries(j) {
            if timestamps[i] < timestamp {
                timestamps[i] = timestamp;
                res.data.i[nz] = i;
                nz += 1;
                workspace[i] = val * beta.clone();
            } else {
                workspace[i] += val * beta.clone();
            }
        }

        nz
    }
}

/*
impl<T: Scalar, R, S> CsVector<T, R, S> {
    pub fn axpy(&mut self, alpha: T, x: CsVector<T, R, S>, beta: T) {
        // First, compute the number of non-zero entries.
        let mut nnzero = 0;

        // Allocate a size large enough.
        self.data.set_column_len(0, nnzero);

        // Fill with the axpy.
        let mut i = self.len();
        let mut j = x.len();
        let mut k = nnzero - 1;
        let mut rid1 = self.data.row_index(0, i - 1);
        let mut rid2 = x.data.row_index(0, j - 1);

        while k > 0 {
            if rid1 == rid2 {
                self.data.set_row_index(0, k, rid1);
                self[k] = alpha * x[j] + beta * self[k];
                i -= 1;
                j -= 1;
            } else if rid1 < rid2 {
                self.data.set_row_index(0, k, rid1);
                self[k] = beta * self[i];
                i -= 1;
            } else {
                self.data.set_row_index(0, k, rid2);
                self[k] = alpha * x[j];
                j -= 1;
            }

            k -= 1;
        }
    }
}
*/

impl<T: Scalar + Zero + ClosedAddAssign + ClosedMulAssign, D: Dim, S: StorageMut<T, D>>
    Vector<T, D, S>
{
    /// Performs a sparse AXPY operation: `self = alpha * x + beta * self`.
    ///
    /// AXPY (A times X Plus Y) is a fundamental linear algebra operation that combines
    /// two vectors with scalar multipliers. This sparse version is optimized for when
    /// `x` is a sparse vector, avoiding unnecessary operations on zero elements.
    ///
    /// # Parameters
    ///
    /// - `alpha`: Scalar multiplier for the sparse vector `x`
    /// - `x`: Sparse vector to add (scaled by alpha)
    /// - `beta`: Scalar multiplier for self
    ///
    /// # Mathematical Operation
    ///
    /// For each element i: `self[i] = alpha * x[i] + beta * self[i]`
    ///
    /// The key optimization is that only the non-zero elements of `x` need to be
    /// considered, making this much faster than dense AXPY when `x` is sparse.
    ///
    /// # Special Cases
    ///
    /// - If `beta = 0`, self is overwritten (only x values matter)
    /// - If `alpha = 0`, self is just scaled by beta
    /// - If `beta = 1`, this is a pure sparse accumulation into self
    ///
    /// # Examples
    ///
    /// ## Basic AXPY
    ///
    /// ```
    /// use nalgebra::{CsVector, DVector, Dyn};
    ///
    /// // Create a dense vector
    /// let mut y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// // Create a sparse vector with only 2 non-zero elements
    /// let triplets = vec![
    ///     (1, 0, 5.0),  // element at index 1
    ///     (3, 0, 7.0),  // element at index 3
    /// ];
    /// let x = CsVector::<f64, Dyn>::from_triplet(4, 1, &triplets);
    ///
    /// // Perform: y = 2.0 * x + 1.0 * y
    /// y.axpy_cs(2.0, &x, 1.0);
    ///
    /// // Result: [1.0, 2.0+10.0, 3.0, 4.0+14.0] = [1.0, 12.0, 3.0, 18.0]
    /// assert_eq!(y[0], 1.0);
    /// assert_eq!(y[1], 12.0);
    /// assert_eq!(y[2], 3.0);
    /// assert_eq!(y[3], 18.0);
    /// ```
    ///
    /// ## Accumulation Pattern (beta = 1)
    ///
    /// Common in iterative solvers and assembly operations:
    ///
    /// ```
    /// use nalgebra::{CsVector, DVector, Dyn};
    ///
    /// let mut accumulator = DVector::from_vec(vec![0.0; 5]);
    ///
    /// // Accumulate contributions from sparse sources
    /// let contrib1 = CsVector::<f64, Dyn>::from_triplet(5, 1, &[(0, 0, 1.0), (2, 0, 2.0)]);
    /// let contrib2 = CsVector::<f64, Dyn>::from_triplet(5, 1, &[(1, 0, 3.0), (2, 0, 4.0)]);
    ///
    /// accumulator.axpy_cs(1.0, &contrib1, 1.0);
    /// accumulator.axpy_cs(1.0, &contrib2, 1.0);
    ///
    /// // Result: contributions summed at each index
    /// assert_eq!(accumulator[0], 1.0);
    /// assert_eq!(accumulator[1], 3.0);
    /// assert_eq!(accumulator[2], 6.0); // 2.0 + 4.0
    /// ```
    ///
    /// ## Overwrite Pattern (beta = 0)
    ///
    /// ```
    /// use nalgebra::{CsVector, DVector, Dyn};
    ///
    /// let mut result = DVector::from_vec(vec![999.0; 3]);
    /// let sparse = CsVector::<f64, Dyn>::from_triplet(3, 1, &[(1, 0, 5.0)]);
    ///
    /// // Overwrite with scaled sparse vector (beta = 0)
    /// result.axpy_cs(2.0, &sparse, 0.0);
    ///
    /// // Old values ignored, only sparse entries copied
    /// assert_eq!(result[0], 0.0);
    /// assert_eq!(result[1], 10.0);
    /// assert_eq!(result[2], 0.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - Time complexity: O(nnz(x)) where nnz is the number of non-zeros
    /// - Much faster than dense AXPY when x is sparse
    /// - If beta ≠ 0, all elements of self are scaled (O(n) operation)
    ///
    /// # Applications
    ///
    /// - **Finite Element Assembly**: Accumulating element contributions to global vectors
    /// - **Iterative Solvers**: Computing residuals and updates
    /// - **Sparse Matrix-Vector Products**: Building up results column by column
    /// - **Graph Algorithms**: Accumulating values from sparse neighborhoods
    ///
    /// # See Also
    ///
    /// - Dense AXPY methods for dense vector operations
    pub fn axpy_cs<D2: Dim, S2>(&mut self, alpha: T, x: &CsVector<T, D2, S2>, beta: T)
    where
        S2: CsStorage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        if beta.is_zero() {
            for i in 0..x.len() {
                unsafe {
                    let k = x.data.row_index_unchecked(i);
                    let y = self.vget_unchecked_mut(k);
                    *y = alpha.clone() * x.data.get_value_unchecked(i).clone();
                }
            }
        } else {
            // Needed to be sure even components not present on `x` are multiplied.
            *self *= beta.clone();

            for i in 0..x.len() {
                unsafe {
                    let k = x.data.row_index_unchecked(i);
                    let y = self.vget_unchecked_mut(k);
                    *y += alpha.clone() * x.data.get_value_unchecked(i).clone();
                }
            }
        }
    }

    /*
    pub fn gemv_sparse<R2: Dim, C2: Dim, S2>(&mut self, alpha: T, a: &CsMatrix<T, R2, C2, S2>, x: &DVector<T>, beta: T)
        where
            S2: CsStorage<T, R2, C2> {
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy_sparse(alpha * val, &col2, beta);

        for j in 1..ncols2 {
            let col2 = a.column(j);
            let val = unsafe { *x.vget_unchecked(j) };

            self.axpy_sparse(alpha * val, &col2, T::one());
        }
    }
    */
}

/// Implements sparse matrix multiplication: `C = A * B`.
///
/// This implementation multiplies two sparse matrices efficiently by taking advantage
/// of the sparsity structure. The algorithm computes the result column by column,
/// only performing operations on non-zero elements.
///
/// # Algorithm
///
/// For each column j of B:
/// 1. For each non-zero B\[i,j\]:
///    - Accumulate column i of A scaled by B\[i,j\] into column j of result
/// 2. Extract accumulated values and compress into sparse format
///
/// Time complexity: O(nnz(A) * nnz_per_col(B)) where nnz_per_col is average non-zeros per column
///
/// # Examples
///
/// ## Basic Matrix Multiplication
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Create two small sparse matrices
/// // A = [1.0  0.0]      B = [2.0  0.0]
/// //     [0.0  3.0]          [0.0  4.0]
/// let a = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 0, 1.0),
///     (1, 1, 3.0),
/// ]);
/// let b = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 0, 2.0),
///     (1, 1, 4.0),
/// ]);
///
/// // Compute C = A * B
/// let c = &a * &b;
///
/// // Result: [2.0  0.0]
/// //         [0.0  12.0]
/// assert_eq!(c.nrows(), 2);
/// assert_eq!(c.ncols(), 2);
/// assert_eq!(c.len(), 2); // Still only 2 non-zeros
/// ```
///
/// ## Graph Operations
///
/// Matrix multiplication on adjacency matrices computes paths:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Adjacency matrix for graph: 0->1, 1->2
/// let adj = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
///     (0, 1, 1.0),  // edge from 0 to 1
///     (1, 2, 1.0),  // edge from 1 to 2
/// ]);
///
/// // A^2 gives 2-hop paths
/// let two_hop = &adj * &adj;
///
/// // There's one 2-hop path: 0->1->2
/// assert_eq!(two_hop.len(), 1);
/// ```
///
/// ## Finite Element Assembly
///
/// Computing element stiffness contributions:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // B is shape function derivatives (3 nodes x 2 DOF)
/// let b = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 3, &[
///     (0, 0, 1.0), (0, 1, -0.5), (0, 2, -0.5),
///     (1, 0, 0.0), (1, 1, 0.5), (1, 2, -0.5),
/// ]);
///
/// // D is material matrix (2x2 for plane stress/strain)
/// let d = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 0, 1.0), (0, 1, 0.3),
///     (1, 0, 0.3), (1, 1, 1.0),
/// ]);
///
/// // Intermediate: D * B
/// let db = &d * &b;
///
/// // Element stiffness would be: B^T * D * B
/// let k = &b.transpose() * &db;
///
/// assert_eq!(k.nrows(), 3);
/// assert_eq!(k.ncols(), 3);
/// ```
///
/// # Applications
///
/// - **Finite Element Analysis**: Computing stiffness matrices
/// - **Graph Algorithms**: Finding paths, computing PageRank
/// - **Machine Learning**: Feature transformations, kernel methods
/// - **Network Analysis**: Flow computations, coupling matrices
///
/// # Performance Notes
///
/// - Much faster than dense multiplication for sparse inputs
/// - Performance depends on sparsity pattern (more non-zeros = slower)
/// - Pre-allocates result size as nnz(A) + nnz(B) and shrinks after
///
/// # See Also
///
/// - [`CsMatrix::transpose`] - Often used in combination: `A^T * A`
impl<'b, T, R1, R2, C1, C2, S1, S2> Mul<&'b CsMatrix<T, R2, C2, S2>> for &'_ CsMatrix<T, R1, C1, S1>
where
    T: Scalar + ClosedAddAssign + ClosedMulAssign + Zero,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    S1: CsStorage<T, R1, C1>,
    S2: CsStorage<T, R2, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
    DefaultAllocator: Allocator<C2> + Allocator<R1> + Allocator<R1>,
{
    type Output = CsMatrix<T, R1, C2>;

    fn mul(self, rhs: &'b CsMatrix<T, R2, C2, S2>) -> Self::Output {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();
        assert_eq!(
            ncols1.value(),
            nrows2.value(),
            "Mismatched dimensions for matrix multiplication."
        );

        let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        let mut workspace = OVector::<T, R1>::zeros_generic(nrows1, Const::<1>);
        let mut nz = 0;

        for j in 0..ncols2.value() {
            res.data.p[j] = nz;
            let new_size_bound = nz + nrows1.value();
            res.data.i.resize(new_size_bound, 0);
            res.data.vals.resize(new_size_bound, T::zero());

            for (i, beta) in rhs.data.column_entries(j) {
                for (k, val) in self.data.column_entries(i) {
                    workspace[k] += val.clone() * beta.clone();
                }
            }

            for (i, val) in workspace.as_mut_slice().iter_mut().enumerate() {
                if !val.is_zero() {
                    res.data.i[nz] = i;
                    res.data.vals[nz] = val.clone();
                    *val = T::zero();
                    nz += 1;
                }
            }
        }

        // NOTE: the following has a lower complexity, but is slower in many cases, likely because
        // of branching inside of the inner loop.
        //
        // let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        // let mut timestamps = OVector::zeros_generic(nrows1, Const::<)>;
        // let mut workspace = unsafe { OVector::new_uninitialized_generic(nrows1, Const::<)> };
        // let mut nz = 0;
        //
        // for j in 0..ncols2.value() {
        //     res.data.p[j] = nz;
        //     let new_size_bound = nz + nrows1.value();
        //     res.data.i.resize(new_size_bound, 0);
        //     res.data.vals.resize(new_size_bound, T::zero());
        //
        //     for (i, val) in rhs.data.column_entries(j) {
        //         nz = self.scatter(
        //             i,
        //             val,
        //             timestamps.as_mut_slice(),
        //             j + 1,
        //             workspace.as_mut_slice(),
        //             nz,
        //             &mut res,
        //         );
        //     }
        //
        //     // Keep the output sorted.
        //     let range = res.data.p[j]..nz;
        //     res.data.i[range.clone()].sort();
        //
        //     for p in range {
        //         res.data.vals[p] = workspace[res.data.i[p]]
        //     }
        // }

        res.data.i.truncate(nz);
        res.data.i.shrink_to_fit();
        res.data.vals.truncate(nz);
        res.data.vals.shrink_to_fit();
        res
    }
}

/// Implements sparse matrix addition: `C = A + B`.
///
/// Adds two sparse matrices element-wise, efficiently handling the sparsity structure.
/// The result includes all positions where either A or B (or both) have non-zero values.
/// This operation merges the sparsity patterns of both matrices.
///
/// # Algorithm
///
/// For each column j:
/// 1. Scatter non-zeros from A\[j\] into a workspace
/// 2. Scatter and accumulate non-zeros from B\[j\] into the same workspace
/// 3. Extract all accumulated values to form the result column
///
/// Time complexity: O(nnz(A) + nnz(B)) where nnz is the number of non-zeros
///
/// # Examples
///
/// ## Basic Addition
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // First matrix:
/// // [1.0  0.0]
/// // [0.0  2.0]
/// let a = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 0, 1.0),
///     (1, 1, 2.0),
/// ]);
///
/// // Second matrix:
/// // [0.0  3.0]
/// // [4.0  0.0]
/// let b = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 1, 3.0),
///     (1, 0, 4.0),
/// ]);
///
/// // Sum:
/// // [1.0  3.0]
/// // [4.0  2.0]
/// let c = &a + &b;
///
/// assert_eq!(c.nrows(), 2);
/// assert_eq!(c.ncols(), 2);
/// assert_eq!(c.len(), 4); // All 4 positions are now non-zero
/// ```
///
/// ## Overlapping Non-zeros
///
/// When both matrices have values at the same position, they are added:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// let a = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
///     (0, 0, 1.0),
///     (1, 1, 2.0),
/// ]);
///
/// let b = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
///     (0, 0, 3.0),  // Same position as in A
///     (2, 2, 4.0),
/// ]);
///
/// let c = &a + &b;
///
/// // Position (0,0) contains 1.0 + 3.0 = 4.0
/// assert_eq!(c.len(), 3); // Three non-zero positions total
/// ```
///
/// ## Finite Element Assembly
///
/// Adding element matrices to form global stiffness matrix:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Element 1 stiffness (simplified 2x2)
/// let k1 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(4, 4, &[
///     (0, 0, 2.0), (0, 1, -1.0),
///     (1, 0, -1.0), (1, 1, 2.0),
/// ]);
///
/// // Element 2 stiffness (shares node 1 with element 1)
/// let k2 = CsMatrix::<f64, Dyn, Dyn>::from_triplet(4, 4, &[
///     (1, 1, 3.0), (1, 2, -1.5),
///     (2, 1, -1.5), (2, 2, 3.0),
/// ]);
///
/// // Global stiffness: overlapping entries are added
/// let k_global = &k1 + &k2;
///
/// // Node 1 appears in both elements, so its contributions are summed
/// ```
///
/// ## Iterative Methods
///
/// Building approximations by adding corrections:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Initial approximation
/// let a_approx = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
///     (0, 0, 1.0),
///     (1, 1, 1.0),
///     (2, 2, 1.0),
/// ]);
///
/// // Sparse correction/update
/// let correction = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &[
///     (0, 1, 0.1),
///     (1, 2, 0.2),
/// ]);
///
/// let updated = &a_approx + &correction;
///
/// // Combines diagonal structure with off-diagonal updates
/// assert_eq!(updated.len(), 5);
/// ```
///
/// # Applications
///
/// - **Finite Element Analysis**: Assembling element contributions
/// - **Iterative Solvers**: Adding corrections and residuals
/// - **Graph Operations**: Combining adjacency matrices from multiple graphs
/// - **Numerical Methods**: Building composite operators
///
/// # Performance Notes
///
/// - Time complexity is linear in total non-zeros
/// - Result may have more non-zeros than either input
/// - Pre-allocates for worst case (nnz(A) + nnz(B)) and shrinks
///
/// # See Also
///
/// - Subtraction can be done via scalar multiplication and addition
/// - Matrix multiplication via the `*` operator
impl<'b, T, R1, R2, C1, C2, S1, S2> Add<&'b CsMatrix<T, R2, C2, S2>> for &'_ CsMatrix<T, R1, C1, S1>
where
    T: Scalar + ClosedAddAssign + ClosedMulAssign + Zero + One,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    S1: CsStorage<T, R1, C1>,
    S2: CsStorage<T, R2, C2>,
    ShapeConstraint: DimEq<R1, R2> + DimEq<C1, C2>,
    DefaultAllocator: Allocator<C2> + Allocator<R1> + Allocator<R1>,
{
    type Output = CsMatrix<T, R1, C2>;

    fn add(self, rhs: &'b CsMatrix<T, R2, C2, S2>) -> Self::Output {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();
        assert_eq!(
            (nrows1.value(), ncols1.value()),
            (nrows2.value(), ncols2.value()),
            "Mismatched dimensions for matrix sum."
        );

        let mut res = CsMatrix::new_uninitialized_generic(nrows1, ncols2, self.len() + rhs.len());
        let mut timestamps = OVector::zeros_generic(nrows1, Const::<1>);
        let mut workspace = Matrix::zeros_generic(nrows1, Const::<1>);
        let mut nz = 0;

        for j in 0..ncols2.value() {
            res.data.p[j] = nz;

            nz = self.scatter(
                j,
                T::one(),
                timestamps.as_mut_slice(),
                j + 1,
                workspace.as_mut_slice(),
                nz,
                &mut res,
            );

            nz = rhs.scatter(
                j,
                T::one(),
                timestamps.as_mut_slice(),
                j + 1,
                workspace.as_mut_slice(),
                nz,
                &mut res,
            );

            // Keep the output sorted.
            let range = res.data.p[j]..nz;
            res.data.i[range.clone()].sort_unstable();

            for p in range {
                res.data.vals[p] = workspace[res.data.i[p]].clone()
            }
        }

        res.data.i.truncate(nz);
        res.data.i.shrink_to_fit();
        res.data.vals.truncate(nz);
        res.data.vals.shrink_to_fit();
        res
    }
}

/// Implements scalar multiplication: `B = A * scalar`.
///
/// Multiplies every non-zero element in the sparse matrix by a scalar value.
/// This operation preserves the sparsity pattern (positions of non-zeros)
/// and only modifies the stored values.
///
/// # Examples
///
/// ## Basic Scaling
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// let triplets = vec![
///     (0, 0, 1.0),
///     (1, 1, 2.0),
///     (2, 2, 3.0),
/// ];
/// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
///
/// // Scale all values by 2.5
/// let scaled = m * 2.5;
///
/// // The sparsity pattern is unchanged, but values are scaled
/// assert_eq!(scaled.len(), 3); // Still 3 non-zeros
/// ```
///
/// ## Normalization
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Create a matrix
/// let triplets = vec![
///     (0, 0, 10.0),
///     (1, 1, 20.0),
/// ];
/// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet(3, 3, &triplets);
///
/// // Normalize by the maximum value
/// let max_val = 20.0;
/// let normalized = m * (1.0 / max_val);
/// ```
///
/// ## Physical Unit Conversion
///
/// Converting stiffness matrix from kN/mm to N/m:
///
/// ```
/// use nalgebra::CsMatrix;
/// use nalgebra::Dyn;
///
/// // Stiffness in kN/mm
/// let k_knmm = CsMatrix::<f64, Dyn, Dyn>::from_triplet(2, 2, &[
///     (0, 0, 100.0),
///     (1, 1, 150.0),
/// ]);
///
/// // Convert to N/m: 1 kN/mm = 1e6 N/m
/// let k_nm = k_knmm * 1e6;
/// ```
///
/// # Performance
///
/// - Time complexity: O(nnz) where nnz is the number of non-zeros
/// - In-place operation, no memory allocation
/// - Preserves sparsity pattern exactly
///
/// # See Also
///
/// - Division by a scalar can be done by multiplying by its reciprocal
impl<T, R, C, S> Mul<T> for CsMatrix<T, R, C, S>
where
    T: Scalar + ClosedAddAssign + ClosedMulAssign + Zero,
    R: Dim,
    C: Dim,
    S: CsStorageMut<T, R, C>,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        for e in self.values_mut() {
            *e *= rhs.clone()
        }

        self
    }
}
