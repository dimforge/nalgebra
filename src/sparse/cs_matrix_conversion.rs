use num::Zero;
use simba::scalar::ClosedAddAssign;

use crate::allocator::Allocator;
use crate::sparse::cs_utils;
use crate::sparse::{CsMatrix, CsStorage};
use crate::storage::Storage;
use crate::{DefaultAllocator, Dim, Dyn, Matrix, OMatrix, Scalar};

impl<T: Scalar + Zero + ClosedAddAssign> CsMatrix<T> {
    /// Creates a column-compressed sparse matrix from data in triplet (coordinate) format.
    ///
    /// The triplet format is the most intuitive way to represent sparse matrices: you provide
    /// three parallel arrays specifying the row index, column index, and value for each
    /// non-zero element. This method is ideal for constructing sparse matrices when you
    /// know the individual non-zero entries.
    ///
    /// # Parameters
    ///
    /// - `nrows`: Number of rows in the resulting matrix
    /// - `ncols`: Number of columns in the resulting matrix
    /// - `irows`: Row indices of non-zero elements (0-indexed)
    /// - `icols`: Column indices of non-zero elements (0-indexed)
    /// - `vals`: Values of the non-zero elements
    ///
    /// # Triplet Format Explained
    ///
    /// The triplet format represents each non-zero element as `(row, column, value)`.
    /// For example, to represent this matrix:
    /// ```text
    /// [1.0  0.0  2.0]
    /// [0.0  3.0  0.0]
    /// [4.0  0.0  5.0]
    /// ```
    ///
    /// You would use:
    /// - `irows = [0, 0, 1, 2, 2]` (row indices)
    /// - `icols = [0, 2, 1, 0, 2]` (column indices)
    /// - `vals  = [1.0, 2.0, 3.0, 4.0, 5.0]` (values)
    ///
    /// # Important Behaviors
    ///
    /// - **Duplicate Handling**: If multiple entries have the same `(row, col)` position,
    ///   their values are summed together. This is useful for finite element assembly.
    /// - **Automatic Sorting**: The resulting matrix is automatically sorted and compressed
    ///   into efficient CSC (Compressed Sparse Column) format.
    /// - **Order Independence**: Triplets can be provided in any order; they will be sorted.
    ///
    /// # Panics
    ///
    /// Panics if the three input slices (`irows`, `icols`, `vals`) have different lengths.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    ///
    /// // Create a 3x3 identity matrix
    /// let irows = vec![0, 1, 2];
    /// let icols = vec![0, 1, 2];
    /// let vals = vec![1.0, 1.0, 1.0];
    ///
    /// let identity = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// assert_eq!(identity.nrows(), 3);
    /// assert_eq!(identity.ncols(), 3);
    /// assert_eq!(identity.len(), 3); // Only 3 non-zero elements stored
    /// ```
    ///
    /// ## Duplicate Entries (Useful for Assembly)
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    ///
    /// // Multiple entries at position (0, 0) are summed
    /// let irows = vec![0, 0, 0, 1];
    /// let icols = vec![0, 0, 0, 1];
    /// let vals = vec![1.0, 2.0, 3.0, 4.0];
    ///
    /// let m = CsMatrix::<f64>::from_triplet(2, 2, &irows, &icols, &vals);
    ///
    /// // The (0,0) entry should be 1.0 + 2.0 + 3.0 = 6.0
    /// assert_eq!(m.len(), 2); // Only 2 unique positions
    /// ```
    ///
    /// ## Finite Element Assembly Pattern
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    ///
    /// // Simulating assembly from multiple finite elements
    /// // Each element contributes to overlapping matrix entries
    /// let mut irows = Vec::new();
    /// let mut icols = Vec::new();
    /// let mut vals = Vec::new();
    ///
    /// // Element 1: connects nodes 0 and 1
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         irows.push(i);
    ///         icols.push(j);
    ///         vals.push(1.0);
    ///     }
    /// }
    ///
    /// // Element 2: connects nodes 1 and 2 (overlaps at node 1)
    /// for i in 1..3 {
    ///     for j in 1..3 {
    ///         irows.push(i);
    ///         icols.push(j);
    ///         vals.push(1.0);
    ///     }
    /// }
    ///
    /// // Contributions are automatically summed
    /// let stiffness = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    /// ```
    ///
    /// ## Creating from Coordinate Data
    ///
    /// ```
    /// use nalgebra::CsMatrix;
    ///
    /// // Data can be in any order
    /// let triplets = vec![
    ///     (2, 1, 5.0),  // (row, col, value)
    ///     (0, 0, 1.0),
    ///     (1, 2, 3.0),
    ///     (0, 1, 2.0),
    /// ];
    ///
    /// let (irows, icols, vals): (Vec<_>, Vec<_>, Vec<_>) =
    ///     triplets.into_iter()
    ///         .map(|(i, j, v)| (i, j, v))
    ///         .multiunzip();
    ///
    /// let m = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 3);
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - Construction time is O(n log n) where n is the number of triplets, due to sorting
    /// - Memory usage during construction is approximately 2x the final sparse matrix size
    /// - For incremental assembly, consider collecting all triplets first, then calling
    ///   this method once at the end
    ///
    /// # Common Use Cases
    ///
    /// - **Finite Element Assembly**: Building stiffness/mass matrices element by element
    /// - **Graph Construction**: Building adjacency matrices from edge lists
    /// - **Data Import**: Reading sparse matrix data from files in coordinate format
    /// - **Iterative Construction**: When you don't know the sparsity pattern upfront
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::from_triplet_generic`] - Generic dimension version of this method
    /// - [`From<Matrix<T, R, C, S>> for CsMatrix<T, R, C>`] - Convert from a dense matrix
    pub fn from_triplet(
        nrows: usize,
        ncols: usize,
        irows: &[usize],
        icols: &[usize],
        vals: &[T],
    ) -> Self {
        Self::from_triplet_generic(Dyn(nrows), Dyn(ncols), irows, icols, vals)
    }
}

impl<T: Scalar + Zero + ClosedAddAssign, R: Dim, C: Dim> CsMatrix<T, R, C>
where
    DefaultAllocator: Allocator<C> + Allocator<R>,
{
    /// Creates a column-compressed sparse matrix from triplet format with generic dimensions.
    ///
    /// This is the generic dimension version of [`CsMatrix::from_triplet`]. It allows you to
    /// specify matrix dimensions using nalgebra's dimension types (`Const`, `Dyn`, etc.),
    /// enabling compile-time size checking when dimensions are known at compile time.
    ///
    /// # Generic Dimensions Explained
    ///
    /// Nalgebra supports both static (compile-time) and dynamic (runtime) dimensions:
    /// - **Static**: `Const<N>` - Size known at compile time (e.g., `Const<3>` for size 3)
    /// - **Dynamic**: `Dyn(n)` - Size determined at runtime
    ///
    /// Using static dimensions enables additional compiler optimizations and type safety.
    ///
    /// # Parameters
    ///
    /// - `nrows`: Number of rows as a dimension type (`R`)
    /// - `ncols`: Number of columns as a dimension type (`C`)
    /// - `irows`: Row indices of non-zero elements (0-indexed)
    /// - `icols`: Column indices of non-zero elements (0-indexed)
    /// - `vals`: Values of the non-zero elements
    ///
    /// # How It Works
    ///
    /// This method performs several steps to construct an efficient sparse matrix:
    /// 1. **Validates** that all input arrays have the same length
    /// 2. **Counts** non-zeros per column to build column pointers
    /// 3. **Distributes** values into CSC format (Compressed Sparse Column)
    /// 4. **Sorts** row indices within each column
    /// 5. **Deduplicates** by summing values at the same position
    ///
    /// # Panics
    ///
    /// Panics if the three input slices (`irows`, `icols`, `vals`) have different lengths.
    ///
    /// # Examples
    ///
    /// ## Dynamic Dimensions (Runtime-Known)
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Dyn};
    ///
    /// let irows = vec![0, 1, 2];
    /// let icols = vec![0, 1, 2];
    /// let vals = vec![1.0, 2.0, 3.0];
    ///
    /// // Dynamic dimensions - like from_triplet
    /// let m = CsMatrix::<f64, Dyn, Dyn>::from_triplet_generic(
    ///     Dyn(3),
    ///     Dyn(3),
    ///     &irows,
    ///     &icols,
    ///     &vals
    /// );
    ///
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 3);
    /// ```
    ///
    /// ## Static Dimensions (Compile-Time Known)
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Const};
    ///
    /// let irows = vec![0, 1, 2];
    /// let icols = vec![0, 1, 2];
    /// let vals = vec![1.0, 2.0, 3.0];
    ///
    /// // Static dimensions for a 3x3 matrix
    /// let m = CsMatrix::<f64, Const<3>, Const<3>>::from_triplet_generic(
    ///     Const::<3>,
    ///     Const::<3>,
    ///     &irows,
    ///     &icols,
    ///     &vals
    /// );
    ///
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 3);
    /// ```
    ///
    /// ## Mixed Dimensions
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Const, Dyn};
    ///
    /// // Fixed number of rows, dynamic columns
    /// let irows = vec![0, 1, 2];
    /// let icols = vec![0, 0, 1];
    /// let vals = vec![1.0, 2.0, 3.0];
    ///
    /// let m = CsMatrix::<f64, Const<3>, Dyn>::from_triplet_generic(
    ///     Const::<3>,
    ///     Dyn(2),
    ///     &irows,
    ///     &icols,
    ///     &vals
    /// );
    ///
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 2);
    /// ```
    ///
    /// ## Building a Graph Laplacian
    ///
    /// ```
    /// use nalgebra::{CsMatrix, Dyn};
    ///
    /// // Build a graph Laplacian matrix for a simple graph
    /// // Graph: 0 -- 1 -- 2
    /// let mut irows = Vec::new();
    /// let mut icols = Vec::new();
    /// let mut vals = Vec::new();
    ///
    /// // Diagonal: degree of each node
    /// for i in 0..3 {
    ///     let degree = if i == 1 { 2.0 } else { 1.0 };
    ///     irows.push(i);
    ///     icols.push(i);
    ///     vals.push(degree);
    /// }
    ///
    /// // Off-diagonal: -1 for each edge
    /// let edges = [(0, 1), (1, 0), (1, 2), (2, 1)];
    /// for (i, j) in edges.iter() {
    ///     irows.push(*i);
    ///     icols.push(*j);
    ///     vals.push(-1.0);
    /// }
    ///
    /// let laplacian = CsMatrix::<f64>::from_triplet_generic(
    ///     Dyn(3),
    ///     Dyn(3),
    ///     &irows,
    ///     &icols,
    ///     &vals
    /// );
    ///
    /// assert_eq!(laplacian.len(), 7); // 3 diagonal + 4 off-diagonal
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Using static dimensions may enable better compiler optimizations
    /// - The algorithm is O(n log n) where n is the number of triplets
    /// - Memory overhead during construction is approximately 2x the final size
    ///
    /// # See Also
    ///
    /// - [`CsMatrix::from_triplet`] - Simpler version with dynamic dimensions only
    /// - [`From<Matrix<T, R, C, S>> for CsMatrix<T, R, C>`] - Convert from dense matrix
    pub fn from_triplet_generic(
        nrows: R,
        ncols: C,
        irows: &[usize],
        icols: &[usize],
        vals: &[T],
    ) -> Self {
        assert!(vals.len() == irows.len());
        assert!(vals.len() == icols.len());

        let mut res = CsMatrix::new_uninitialized_generic(nrows, ncols, vals.len());
        let mut workspace = res.data.p.clone();

        // Column count.
        for j in icols.iter().cloned() {
            workspace[j] += 1;
        }

        let _ = cs_utils::cumsum(&mut workspace, &mut res.data.p);

        // Fill i and vals.
        for ((i, j), val) in irows
            .iter()
            .cloned()
            .zip(icols.iter().cloned())
            .zip(vals.iter().cloned())
        {
            let offset = workspace[j];
            res.data.i[offset] = i;
            res.data.vals[offset] = val;
            workspace[j] = offset + 1;
        }

        // Sort the result.
        res.sort();
        res.dedup();
        res
    }
}

impl<T: Scalar + Zero, R: Dim, C: Dim, S> From<CsMatrix<T, R, C, S>> for OMatrix<T, R, C>
where
    S: CsStorage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    /// Converts a sparse matrix (CSC format) into a dense matrix.
    ///
    /// This conversion expands the sparse representation into a full dense matrix,
    /// filling in all the zero entries that were implicitly stored. This is useful
    /// when you need to perform operations that are more efficient on dense matrices,
    /// or when interfacing with libraries that only work with dense storage.
    ///
    /// # Memory Considerations
    ///
    /// Converting to dense format can dramatically increase memory usage. A sparse
    /// matrix with only 1,000 non-zeros might expand to a 1,000,000 element dense
    /// matrix. Only convert to dense when:
    /// - The matrix is small enough to fit in memory when dense
    /// - You need dense-specific operations (like certain decompositions)
    /// - The matrix is not very sparse (>10% non-zeros)
    ///
    /// # How It Works
    ///
    /// 1. Allocates a dense matrix filled with zeros
    /// 2. Iterates through each column of the sparse matrix
    /// 3. For each non-zero entry, sets the corresponding position in the dense matrix
    ///
    /// Time complexity: O(n + m) where n is the number of non-zeros and m is nrows × ncols.
    ///
    /// # Examples
    ///
    /// ## Basic Conversion
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DMatrix};
    ///
    /// // Create a small sparse identity matrix
    /// let irows = vec![0, 1, 2];
    /// let icols = vec![0, 1, 2];
    /// let vals = vec![1.0, 1.0, 1.0];
    /// let sparse = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// // Convert to dense matrix
    /// let dense: DMatrix<f64> = sparse.into();
    ///
    /// assert_eq!(dense[(0, 0)], 1.0);
    /// assert_eq!(dense[(1, 1)], 1.0);
    /// assert_eq!(dense[(2, 2)], 1.0);
    /// assert_eq!(dense[(0, 1)], 0.0); // Zeros are now explicit
    /// ```
    ///
    /// ## Interfacing with Dense Algorithms
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DMatrix};
    ///
    /// // Build a small sparse matrix
    /// let irows = vec![0, 0, 1];
    /// let icols = vec![0, 1, 1];
    /// let vals = vec![4.0, 1.0, 3.0];
    /// let sparse = CsMatrix::<f64>::from_triplet(2, 2, &irows, &icols, &vals);
    ///
    /// // Convert to dense to use dense matrix operations
    /// let dense: DMatrix<f64> = sparse.into();
    ///
    /// // Now you can use dense-specific operations
    /// assert_eq!(dense[(0, 0)], 4.0);
    /// assert_eq!(dense[(0, 1)], 1.0);
    /// assert_eq!(dense[(1, 0)], 0.0);
    /// assert_eq!(dense[(1, 1)], 3.0);
    /// ```
    ///
    /// ## Visualizing Sparse Structure
    ///
    /// Converting to dense can be useful for debugging and visualization:
    ///
    /// ```
    /// use nalgebra::{CsMatrix, DMatrix};
    ///
    /// // Create a tridiagonal matrix
    /// let irows = vec![0, 1, 0, 1, 2, 1, 2];
    /// let icols = vec![0, 0, 1, 1, 1, 2, 2];
    /// let vals = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
    /// let sparse = CsMatrix::<f64>::from_triplet(3, 3, &irows, &icols, &vals);
    ///
    /// let dense: DMatrix<f64> = sparse.into();
    ///
    /// // Now easy to print/inspect:
    /// // [ 2.0, -1.0,  0.0 ]
    /// // [-1.0,  2.0, -1.0 ]
    /// // [ 0.0, -1.0,  2.0 ]
    /// ```
    ///
    /// # Performance Warning
    ///
    /// This operation is O(nrows × ncols) in time and space, regardless of the
    /// sparsity of the input. For large sparse matrices, this can be:
    /// - **Slow**: Must fill millions of zeros
    /// - **Memory-intensive**: Dense storage requires nrows × ncols × sizeof(T) bytes
    ///
    /// Example: A 10,000 × 10,000 sparse matrix with 50,000 non-zeros (~0.05% sparse):
    /// - Sparse: ~400 KB (50,000 values)
    /// - Dense: ~800 MB (100,000,000 values)
    ///
    /// # See Also
    ///
    /// - [`From<Matrix> for CsMatrix`] - Convert dense to sparse (opposite direction)
    /// - [`CsMatrix::from_triplet`] - Create sparse matrices from triplet format
    fn from(m: CsMatrix<T, R, C, S>) -> Self {
        let (nrows, ncols) = m.data.shape();
        let mut res = OMatrix::zeros_generic(nrows, ncols);

        for j in 0..ncols.value() {
            for (i, val) in m.data.column_entries(j) {
                res[(i, j)] = val;
            }
        }

        res
    }
}

impl<T: Scalar + Zero, R: Dim, C: Dim, S> From<Matrix<T, R, C, S>> for CsMatrix<T, R, C>
where
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C> + Allocator<C>,
{
    /// Converts a dense matrix into a sparse matrix (CSC format) by extracting non-zeros.
    ///
    /// This conversion scans the dense matrix and stores only the non-zero elements
    /// in Compressed Sparse Column (CSC) format. This is useful when you have a dense
    /// matrix that is mostly zeros and want to save memory or use sparse algorithms.
    ///
    /// # When to Use Sparse Format
    ///
    /// Consider converting to sparse when:
    /// - **High Sparsity**: Less than 10-20% of elements are non-zero
    /// - **Large Matrices**: Memory savings outweigh conversion cost
    /// - **Sparse Algorithms**: Using solvers optimized for sparse matrices
    /// - **Storage/Transmission**: Reducing data size for I/O operations
    ///
    /// # How It Works
    ///
    /// 1. Counts non-zero elements in the dense matrix
    /// 2. Allocates sparse storage for exactly that many elements
    /// 3. Scans column-by-column, extracting non-zero values and their row indices
    /// 4. Builds the CSC structure (column pointers, row indices, values)
    ///
    /// Time complexity: O(nrows × ncols) - must scan entire dense matrix.
    ///
    /// # Zero Detection
    ///
    /// Elements are considered zero if they satisfy `element.is_zero()`. For
    /// floating-point numbers, this means exactly `0.0`, not "close to zero".
    /// If you want to treat small values as zero, threshold them first.
    ///
    /// # Examples
    ///
    /// ## Basic Conversion
    ///
    /// ```
    /// use nalgebra::{DMatrix, CsMatrix};
    ///
    /// // Create a dense matrix with some zeros
    /// let dense = DMatrix::from_row_slice(3, 3, &[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 3.0,
    /// ]);
    ///
    /// // Convert to sparse - only stores non-zeros
    /// let sparse: CsMatrix<f64> = dense.into();
    ///
    /// assert_eq!(sparse.nrows(), 3);
    /// assert_eq!(sparse.ncols(), 3);
    /// assert_eq!(sparse.len(), 3); // Only 3 non-zeros stored
    /// ```
    ///
    /// ## Memory Savings Example
    ///
    /// ```
    /// use nalgebra::{DMatrix, CsMatrix};
    ///
    /// // Create a large sparse diagonal matrix
    /// let size = 1000;
    /// let mut dense = DMatrix::zeros(size, size);
    /// for i in 0..size {
    ///     dense[(i, i)] = 1.0;
    /// }
    ///
    /// // Dense: 1,000,000 values stored (8 MB for f64)
    /// assert_eq!(dense.len(), 1_000_000);
    ///
    /// // Convert to sparse
    /// let sparse: CsMatrix<f64> = dense.into();
    ///
    /// // Sparse: only 1,000 values stored (~8 KB for f64)
    /// assert_eq!(sparse.len(), 1000);
    /// // Memory reduction: ~1000x smaller!
    /// ```
    ///
    /// ## Converting After Thresholding
    ///
    /// For numerical matrices, you might want to treat small values as zero:
    ///
    /// ```
    /// use nalgebra::{DMatrix, CsMatrix};
    ///
    /// let mut dense = DMatrix::from_row_slice(2, 2, &[
    ///     1.0, 0.0001,
    ///     0.0002, 2.0,
    /// ]);
    ///
    /// // Threshold small values to exact zero before conversion
    /// let threshold = 0.001;
    /// for val in dense.iter_mut() {
    ///     if val.abs() < threshold {
    ///         *val = 0.0;
    ///     }
    /// }
    ///
    /// // Now convert to sparse
    /// let sparse: CsMatrix<f64> = dense.into();
    ///
    /// assert_eq!(sparse.len(), 2); // Only values >= threshold stored
    /// ```
    ///
    /// ## Working with Structured Matrices
    ///
    /// ```
    /// use nalgebra::{DMatrix, CsMatrix};
    ///
    /// // Create a tridiagonal matrix (common in PDEs)
    /// let n = 5;
    /// let mut dense = DMatrix::zeros(n, n);
    /// for i in 0..n {
    ///     dense[(i, i)] = 2.0;
    ///     if i > 0 {
    ///         dense[(i, i-1)] = -1.0;
    ///     }
    ///     if i < n-1 {
    ///         dense[(i, i+1)] = -1.0;
    ///     }
    /// }
    ///
    /// // Convert to sparse - stores only 13 values instead of 25
    /// let sparse: CsMatrix<f64> = dense.into();
    ///
    /// assert_eq!(sparse.len(), 13); // 5 diagonal + 8 off-diagonal
    /// assert_eq!(sparse.nrows(), 5);
    /// assert_eq!(sparse.ncols(), 5);
    /// ```
    ///
    /// ## Checking Sparsity Before Conversion
    ///
    /// ```
    /// use nalgebra::{DMatrix, CsMatrix};
    ///
    /// fn maybe_convert_to_sparse(dense: DMatrix<f64>) -> CsMatrix<f64> {
    ///     let total = dense.nrows() * dense.ncols();
    ///     let non_zeros = dense.iter().filter(|&&x| x != 0.0).count();
    ///     let sparsity = 1.0 - (non_zeros as f64 / total as f64);
    ///
    ///     println!("Matrix is {:.1}% sparse", sparsity * 100.0);
    ///
    ///     // Always convert, but now you know the sparsity
    ///     dense.into()
    /// }
    ///
    /// let dense = DMatrix::from_row_slice(3, 3, &[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 3.0,
    /// ]);
    ///
    /// let sparse = maybe_convert_to_sparse(dense);
    /// // Prints: "Matrix is 66.7% sparse"
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - **Conversion Cost**: Must scan all nrows × ncols elements
    /// - **Memory During Conversion**: Temporarily requires space for both representations
    /// - **Worth It When**: Sparse representation is significantly smaller than dense
    ///
    /// For a 1000×1000 matrix with 1% non-zeros:
    /// - Dense: ~8 MB (1,000,000 × 8 bytes)
    /// - Sparse: ~80 KB (10,000 × 8 bytes)
    /// - Savings: 99% memory reduction
    ///
    /// # See Also
    ///
    /// - [`From<CsMatrix> for OMatrix`] - Convert sparse back to dense
    /// - [`CsMatrix::from_triplet`] - Build sparse directly from non-zero entries
    fn from(m: Matrix<T, R, C, S>) -> Self {
        let (nrows, ncols) = m.data.shape();
        let len = m.iter().filter(|e| !e.is_zero()).count();
        let mut res = CsMatrix::new_uninitialized_generic(nrows, ncols, len);
        let mut nz = 0;

        for j in 0..ncols.value() {
            let column = m.column(j);
            res.data.p[j] = nz;

            for i in 0..nrows.value() {
                if !column[i].is_zero() {
                    res.data.i[nz] = i;
                    res.data.vals[nz] = column[i].clone();
                    nz += 1;
                }
            }
        }

        res
    }
}
