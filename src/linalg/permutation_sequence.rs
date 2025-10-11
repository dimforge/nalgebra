#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use num::One;
use simba::scalar::ClosedNeg;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, OVector, Scalar};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::dimension::Dyn;
use crate::dimension::{Const, Dim, DimName};
use crate::storage::StorageMut;

/// A compact representation of a sequence of row or column permutations.
///
/// # What is a Permutation Sequence?
///
/// A permutation sequence is a memory-efficient way to represent multiple row or column swaps
/// applied to a matrix. Instead of storing a full permutation matrix (which would be mostly zeros
/// and ones), it stores only the sequence of swap operations needed to achieve the permutation.
///
/// For example, instead of storing a 1000x1000 permutation matrix (which would use ~8MB for f64),
/// a permutation sequence with 10 swaps only stores 10 pairs of indices (using ~160 bytes).
///
/// # Common Use Cases
///
/// Permutation sequences are primarily used in numerical linear algebra for:
///
/// - **LU Decomposition with Pivoting**: Row swaps improve numerical stability by avoiding
///   division by very small numbers.
/// - **QR Decomposition with Column Pivoting**: Column swaps help identify the most important
///   columns in a matrix.
/// - **Sparse Matrix Reordering**: Reducing fill-in during factorization.
/// - **Solving Linear Systems**: Applying row permutations to right-hand side vectors.
///
/// # Example: Basic Usage
///
/// ```
/// use nalgebra::{Matrix3, PermutationSequence, Const};
///
/// // Create an identity permutation (no swaps)
/// let mut p = PermutationSequence::<Const<3>>::identity();
///
/// // Record a swap between rows 0 and 2
/// p.append_permutation(0, 2);
///
/// // Apply the permutation to a matrix
/// let mut m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// p.permute_rows(&mut m);
///
/// // Rows 0 and 2 are now swapped
/// assert_eq!(m.row(0)[0], 7.0);  // Row 0 now contains what was row 2
/// assert_eq!(m.row(2)[0], 1.0);  // Row 2 now contains what was row 0
/// ```
///
/// # Example: LU Decomposition Context
///
/// ```
/// use nalgebra::Matrix3;
///
/// let m = Matrix3::new(
///     2.0, 1.0, 1.0,
///     4.0, 3.0, 3.0,
///     8.0, 7.0, 9.0,
/// );
///
/// // Compute LU decomposition (which uses permutation internally)
/// let lu = m.lu();
/// let p = lu.p();  // Get the permutation sequence
///
/// // The permutation sequence tells us which rows were swapped during decomposition
/// println!("Number of row swaps: {}", p.len());
///
/// // Apply the permutation to reconstruct P * A = L * U
/// let mut pa = m.clone();
/// p.permute_rows(&mut pa);
/// let l = lu.l();
/// let u = lu.u();
/// assert!((pa - l * u).norm() < 1e-10);
/// ```
///
/// # See Also
///
/// - [`LU`]: LU decomposition with partial pivoting using permutation sequences
/// - [`FullPivLU`]: LU decomposition with full pivoting (row and column permutations)
/// - [`ColPivQR`]: QR decomposition with column pivoting
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<D>,
         OVector<(usize, usize), D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<D>,
         OVector<(usize, usize), D>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct PermutationSequence<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    len: usize,
    ipiv: OVector<(usize, usize), D>,
}

impl<D: Dim> Copy for PermutationSequence<D>
where
    DefaultAllocator: Allocator<D>,
    OVector<(usize, usize), D>: Copy,
{
}

impl<D: DimName> PermutationSequence<D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new statically-allocated identity permutation sequence.
    ///
    /// An identity permutation represents "no swaps" - when applied to a matrix, it leaves
    /// the matrix unchanged. This is the starting point for building up a sequence of
    /// permutations using [`append_permutation`](Self::append_permutation).
    ///
    /// This version creates a permutation sequence with compile-time known dimensions,
    /// which can be more efficient and allows the compiler to perform additional optimizations.
    ///
    /// # Returns
    ///
    /// A new identity permutation sequence of dimension `D` with no recorded swaps.
    ///
    /// # Example: Creating an Identity Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// // Create an identity permutation for 3x3 matrices
    /// let p = PermutationSequence::<Const<3>>::identity();
    ///
    /// // Initially, it has no swaps recorded
    /// assert_eq!(p.len(), 0);
    /// assert!(p.is_empty());
    ///
    /// // Applying it to a matrix leaves it unchanged
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    /// let original = m.clone();
    ///
    /// p.permute_rows(&mut m);
    /// assert_eq!(m, original);
    /// ```
    ///
    /// # Example: Building Up Permutations
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix4};
    ///
    /// // Start with identity and add swaps
    /// let mut p = PermutationSequence::<Const<4>>::identity();
    ///
    /// // Add a sequence of row swaps
    /// p.append_permutation(0, 3);  // Swap rows 0 and 3
    /// p.append_permutation(1, 2);  // Then swap rows 1 and 2
    ///
    /// assert_eq!(p.len(), 2);  // Two swaps recorded
    ///
    /// // Apply to a matrix
    /// let mut m = Matrix4::<f64>::identity();
    /// p.permute_rows(&mut m);
    ///
    /// // Verify the swaps were applied
    /// assert_eq!(m[(0, 0)], 0.0);  // Original row 0 is now at row 3
    /// assert_eq!(m[(0, 3)], 1.0);  // Row 0 now has original row 3
    /// ```
    ///
    /// # Example: Use in Custom Algorithms
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// fn partial_pivot_row(matrix: &mut Matrix3<f64>, col: usize) -> PermutationSequence<Const<3>> {
    ///     // Find row with largest absolute value in column
    ///     let mut max_row = col;
    ///     let mut max_val = matrix[(col, col)].abs();
    ///
    ///     for i in (col + 1)..3 {
    ///         let val = matrix[(i, col)].abs();
    ///         if val > max_val {
    ///             max_val = val;
    ///             max_row = i;
    ///         }
    ///     }
    ///
    ///     // Create permutation and apply swap if needed
    ///     let mut p = PermutationSequence::<Const<3>>::identity();
    ///     if max_row != col {
    ///         p.append_permutation(col, max_row);
    ///         p.permute_rows(matrix);
    ///     }
    ///
    ///     p
    /// }
    ///
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,  // Largest in first column
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let p = partial_pivot_row(&mut m, 0);
    /// assert_eq!(p.len(), 1);  // One swap was performed
    /// ```
    ///
    /// # See Also
    ///
    /// - [`PermutationSequence::identity`](PermutationSequence#method.identity-1): Dynamic version for runtime-sized dimensions
    /// - [`identity_generic`](Self::identity_generic): Generic version accepting any dimension type
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    /// - [`len`](Self::len): Check how many swaps are recorded
    #[inline]
    pub fn identity() -> Self {
        Self::identity_generic(D::name())
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl PermutationSequence<Dyn>
where
    DefaultAllocator: Allocator<Dyn>,
{
    /// Creates a new dynamically-allocated identity permutation sequence.
    ///
    /// An identity permutation represents "no swaps" - when applied to a matrix, it leaves
    /// the matrix unchanged. This is the starting point for building up a sequence of
    /// permutations using [`append_permutation`](Self::append_permutation).
    ///
    /// This version creates a permutation sequence with runtime-determined dimensions,
    /// which is useful when working with matrices whose size is not known at compile time.
    ///
    /// # Arguments
    ///
    /// * `n` - The dimension of the permutation sequence (typically matching the matrix dimensions)
    ///
    /// # Returns
    ///
    /// A new identity permutation sequence of dimension `n` with no recorded swaps.
    ///
    /// # Example: Creating a Dynamic Identity Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Dyn, DMatrix};
    ///
    /// // Create an identity permutation for an n×n matrix (known at runtime)
    /// let n = 5;
    /// let p = PermutationSequence::<Dyn>::identity(n);
    ///
    /// // Initially, it has no swaps recorded
    /// assert_eq!(p.len(), 0);
    /// assert!(p.is_empty());
    ///
    /// // Applying it to a matrix leaves it unchanged
    /// let mut m = DMatrix::from_row_slice(3, 3, &[
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// ]);
    /// let original = m.clone();
    ///
    /// let p3 = PermutationSequence::<Dyn>::identity(3);
    /// p3.permute_rows(&mut m);
    /// assert_eq!(m, original);
    /// ```
    ///
    /// # Example: Runtime Size Determination
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Dyn, DMatrix};
    ///
    /// fn process_matrix(size: usize) -> PermutationSequence<Dyn> {
    ///     // Size is determined at runtime
    ///     let mut p = PermutationSequence::<Dyn>::identity(size);
    ///
    ///     // Add some swaps based on runtime logic
    ///     if size > 2 {
    ///         p.append_permutation(0, size - 1);
    ///     }
    ///
    ///     p
    /// }
    ///
    /// let p = process_matrix(4);
    /// assert_eq!(p.len(), 1);  // One swap was added
    /// ```
    ///
    /// # Example: Building Permutations for Large Matrices
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Dyn, DMatrix};
    ///
    /// // Work with a large dynamically-sized matrix
    /// let matrix_size = 100;
    /// let mut p = PermutationSequence::<Dyn>::identity(matrix_size);
    ///
    /// // Record some swaps
    /// p.append_permutation(0, 99);
    /// p.append_permutation(1, 98);
    ///
    /// // Apply to a large matrix
    /// let mut m = DMatrix::<f64>::identity(matrix_size, matrix_size);
    /// p.permute_rows(&mut m);
    ///
    /// // First and last rows are swapped
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(0, 99)], 1.0);
    /// ```
    ///
    /// # Example: Reading Matrix Size from Input
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Dyn, DMatrix};
    ///
    /// fn create_permutation_from_user_input() -> PermutationSequence<Dyn> {
    ///     // Imagine this comes from user input or file
    ///     let n = 5;
    ///
    ///     let mut p = PermutationSequence::<Dyn>::identity(n);
    ///
    ///     // Add swaps based on some algorithm
    ///     for i in 0..n/2 {
    ///         p.append_permutation(i, n - 1 - i);
    ///     }
    ///
    ///     p
    /// }
    ///
    /// let p = create_permutation_from_user_input();
    /// assert_eq!(p.len(), 2);  // Two swaps recorded
    /// ```
    ///
    /// # See Also
    ///
    /// - [`PermutationSequence::identity`](PermutationSequence#method.identity): Static version for compile-time dimensions
    /// - [`identity_generic`](Self::identity_generic): Generic version accepting any dimension type
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    /// - [`len`](Self::len): Check how many swaps are recorded
    #[inline]
    pub fn identity(n: usize) -> Self {
        Self::identity_generic(Dyn(n))
    }
}

impl<D: Dim> PermutationSequence<D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new identity permutation sequence with generic dimension type.
    ///
    /// This is the most general constructor for creating identity permutation sequences.
    /// It accepts any dimension type that implements the `Dim` trait, making it useful
    /// for generic code that needs to work with both static and dynamic dimensions.
    ///
    /// An identity permutation represents "no swaps" - when applied to a matrix, it leaves
    /// the matrix unchanged.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the permutation sequence (can be `Const<N>` or `Dyn(n)`)
    ///
    /// # Returns
    ///
    /// A new identity permutation sequence with no recorded swaps.
    ///
    /// # Example: Generic Function
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Dim, DefaultAllocator, Const, Dyn};
    /// use nalgebra::allocator::Allocator;
    ///
    /// fn create_identity_perm<D: Dim>(dim: D) -> PermutationSequence<D>
    /// where
    ///     DefaultAllocator: Allocator<D>,
    /// {
    ///     PermutationSequence::identity_generic(dim)
    /// }
    ///
    /// // Works with static dimensions
    /// let p_static = create_identity_perm(Const::<5>);
    /// assert_eq!(p_static.len(), 0);
    ///
    /// // Also works with dynamic dimensions
    /// let p_dynamic = create_identity_perm(Dyn(5));
    /// assert_eq!(p_dynamic.len(), 0);
    /// ```
    ///
    /// # Example: In a Generic Algorithm
    ///
    /// ```
    /// use nalgebra::{Matrix, PermutationSequence, Scalar, Dim, DefaultAllocator};
    /// use nalgebra::allocator::Allocator;
    /// use nalgebra::storage::StorageMut;
    /// use simba::scalar::ComplexField;
    ///
    /// fn find_pivot_and_swap<T, D, S>(
    ///     matrix: &mut Matrix<T, D, D, S>,
    ///     col: usize,
    /// ) -> PermutationSequence<D>
    /// where
    ///     T: ComplexField,
    ///     D: Dim,
    ///     S: StorageMut<T, D, D>,
    ///     DefaultAllocator: Allocator<D>,
    /// {
    ///     let dim = matrix.shape_generic().0;
    ///     let mut p = PermutationSequence::identity_generic(dim);
    ///
    ///     // Simple pivoting logic (finds first non-zero)
    ///     if col < matrix.nrows() && matrix[(col, col)].is_zero() {
    ///         for i in (col + 1)..matrix.nrows() {
    ///             if !matrix[(i, col)].is_zero() {
    ///                 p.append_permutation(col, i);
    ///                 p.permute_rows(matrix);
    ///                 break;
    ///             }
    ///         }
    ///     }
    ///
    ///     p
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`PermutationSequence::identity`](PermutationSequence#method.identity): Static version (compile-time dimensions)
    /// - [`PermutationSequence::identity`](PermutationSequence#method.identity-1): Dynamic version (runtime dimensions)
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    #[inline]
    pub fn identity_generic(dim: D) -> Self {
        Self {
            len: 0,
            // TODO: using a uninitialized matrix would save some computation, but
            //       that loos difficult to setup with MaybeUninit.
            ipiv: Matrix::repeat_generic(dim, Const::<1>, (0, 0)),
        }
    }

    /// Appends a swap operation to this permutation sequence.
    ///
    /// This method records a swap between indices `i` and `i2`. When the permutation
    /// is later applied to a matrix, this swap will be performed in the order it was added.
    ///
    /// Note: If `i` equals `i2`, no swap is recorded (this is a no-op).
    ///
    /// # Arguments
    ///
    /// * `i` - The first index to swap
    /// * `i2` - The second index to swap
    ///
    /// # Panics
    ///
    /// Panics if the maximum number of permutations (equal to the dimension) has been exceeded.
    /// This happens when you try to record more swaps than the dimension of the sequence.
    ///
    /// # Example: Building a Permutation Sequence
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// // Start with an identity permutation
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    ///
    /// // Record a swap between indices 0 and 2
    /// p.append_permutation(0, 2);
    ///
    /// // Record another swap between indices 0 and 1
    /// p.append_permutation(0, 1);
    ///
    /// assert_eq!(p.len(), 2);  // Two swaps recorded
    ///
    /// // Apply to a matrix - swaps are applied in order
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// p.permute_rows(&mut m);
    ///
    /// // After first swap (0,2): rows become [7,8,9], [4,5,6], [1,2,3]
    /// // After second swap (0,1): rows become [4,5,6], [7,8,9], [1,2,3]
    /// assert_eq!(m.row(0)[0], 4.0);
    /// assert_eq!(m.row(1)[0], 7.0);
    /// assert_eq!(m.row(2)[0], 1.0);
    /// ```
    ///
    /// # Example: No-Op When Indices Are Equal
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    ///
    /// // Swapping an index with itself does nothing
    /// p.append_permutation(1, 1);
    ///
    /// // No swap was recorded
    /// assert_eq!(p.len(), 0);
    /// assert!(p.is_empty());
    /// ```
    ///
    /// # Example: LU Decomposition Pivoting
    ///
    /// ```
    /// use nalgebra::{Matrix3, PermutationSequence, Const};
    ///
    /// fn simple_pivot_step(
    ///     matrix: &mut Matrix3<f64>,
    ///     col: usize,
    /// ) -> PermutationSequence<Const<3>> {
    ///     let mut p = PermutationSequence::<Const<3>>::identity();
    ///
    ///     // Find the row with largest absolute value in current column
    ///     let mut max_row = col;
    ///     let mut max_val = matrix[(col, col)].abs();
    ///
    ///     for i in (col + 1)..3 {
    ///         let val = matrix[(i, col)].abs();
    ///         if val > max_val {
    ///             max_val = val;
    ///             max_row = i;
    ///         }
    ///     }
    ///
    ///     // Record the pivot swap
    ///     if max_row != col {
    ///         p.append_permutation(col, max_row);
    ///         matrix.swap_rows(col, max_row);
    ///     }
    ///
    ///     p
    /// }
    ///
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     8.0, 7.0, 6.0,  // Largest element in first column
    ///     4.0, 5.0, 9.0,
    /// );
    ///
    /// let p = simple_pivot_step(&mut m, 0);
    /// assert_eq!(p.len(), 1);  // One pivot swap was needed
    /// ```
    ///
    /// # Example: Composing Permutations
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix4};
    ///
    /// // Build a complex permutation by adding multiple swaps
    /// let mut p = PermutationSequence::<Const<4>>::identity();
    ///
    /// // Reverse the order of rows by swapping outer pairs
    /// p.append_permutation(0, 3);  // Swap first and last
    /// p.append_permutation(1, 2);  // Swap middle two
    ///
    /// let mut m = Matrix4::from_row_slice(&[
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ]);
    ///
    /// p.permute_rows(&mut m);
    ///
    /// // Rows are now reversed
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(0, 3)], 1.0);  // Last row is now first
    /// ```
    ///
    /// # See Also
    ///
    /// - [`permute_rows`](Self::permute_rows): Apply the permutation to matrix rows
    /// - [`permute_columns`](Self::permute_columns): Apply the permutation to matrix columns
    /// - [`len`](Self::len): Get the number of recorded swaps
    #[inline]
    pub fn append_permutation(&mut self, i: usize, i2: usize) {
        if i != i2 {
            assert!(
                self.len < self.ipiv.len(),
                "Maximum number of permutations exceeded."
            );
            self.ipiv[self.len] = (i, i2);
            self.len += 1;
        }
    }

    /// Applies this permutation sequence to the rows of a matrix.
    ///
    /// This method performs all recorded row swaps in the order they were added to the
    /// permutation sequence. Each swap exchanges two rows of the matrix.
    ///
    /// # What This Does
    ///
    /// If the permutation sequence records swaps (i₁, j₁), (i₂, j₂), ..., (iₙ, jₙ),
    /// this method applies them in that exact order: first swap rows i₁ and j₁, then
    /// swap rows i₂ and j₂, and so on.
    ///
    /// This is commonly used in:
    /// - **LU decomposition**: Applying pivoting to matrices or right-hand sides
    /// - **Linear system solving**: Permuting the RHS vector before solving
    /// - **Matrix reordering**: Rearranging rows for numerical stability
    ///
    /// # Arguments
    ///
    /// * `rhs` - A mutable reference to the matrix whose rows will be permuted
    ///
    /// # Example: Basic Row Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);  // Swap rows 0 and 2
    ///
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,   // Row 0
    ///     4.0, 5.0, 6.0,   // Row 1
    ///     7.0, 8.0, 9.0,   // Row 2
    /// );
    ///
    /// p.permute_rows(&mut m);
    ///
    /// // Rows 0 and 2 have been swapped
    /// assert_eq!(m.row(0)[0], 7.0);  // Row 0 now contains old row 2
    /// assert_eq!(m.row(1)[0], 4.0);  // Row 1 unchanged
    /// assert_eq!(m.row(2)[0], 1.0);  // Row 2 now contains old row 0
    /// ```
    ///
    /// # Example: LU Decomposition - Reconstructing P*A
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// // Compute LU decomposition
    /// let lu = m.lu();
    /// let (p, l, u) = lu.unpack();
    ///
    /// // Apply permutation to original matrix to get P*A
    /// let mut pa = m.clone();
    /// p.permute_rows(&mut pa);
    ///
    /// // Verify that P*A = L*U
    /// let lu_product = l * u;
    /// assert!((pa - lu_product).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving Linear Systems with Pivoting
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// let lu = a.lu();
    /// let p = lu.p();
    ///
    /// // To solve manually, we need to permute b first
    /// let mut pb = b.clone();
    /// p.permute_rows(&mut pb);
    ///
    /// // Now solve L*y = P*b and U*x = y
    /// let l = lu.l();
    /// let u = lu.u();
    /// let y = l.solve_lower_triangular(&pb).unwrap();
    /// let x = u.solve_upper_triangular(&y).unwrap();
    ///
    /// // Verify solution
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Multiple Sequential Swaps
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix4};
    ///
    /// let mut p = PermutationSequence::<Const<4>>::identity();
    ///
    /// // Add multiple swaps
    /// p.append_permutation(0, 1);  // First swap rows 0 and 1
    /// p.append_permutation(2, 3);  // Then swap rows 2 and 3
    /// p.append_permutation(1, 2);  // Finally swap rows 1 and 2
    ///
    /// let mut m = Matrix4::from_row_slice(&[
    ///     1.0, 0.0, 0.0, 0.0,   // Row 0
    ///     0.0, 1.0, 0.0, 0.0,   // Row 1
    ///     0.0, 0.0, 1.0, 0.0,   // Row 2
    ///     0.0, 0.0, 0.0, 1.0,   // Row 3
    /// ]);
    ///
    /// p.permute_rows(&mut m);
    ///
    /// // Verify the swaps were applied in order
    /// // After (0,1): [0,1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]
    /// // After (2,3): [0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]
    /// // After (1,2): [0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,0]
    /// assert_eq!(m[(0, 1)], 1.0);
    /// assert_eq!(m[(1, 3)], 1.0);
    /// assert_eq!(m[(2, 0)], 1.0);
    /// assert_eq!(m[(3, 2)], 1.0);
    /// ```
    ///
    /// # Performance
    ///
    /// The time complexity is O(k * n) where k is the number of swaps ([`len`](Self::len))
    /// and n is the number of columns in the matrix. Each swap operation is O(n).
    ///
    /// # See Also
    ///
    /// - [`inv_permute_rows`](Self::inv_permute_rows): Apply the inverse permutation (undo these swaps)
    /// - [`permute_columns`](Self::permute_columns): Apply permutation to columns instead
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    /// - [`determinant`](Self::determinant): Compute the sign of the permutation
    #[inline]
    pub fn permute_rows<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for i in self.ipiv.rows_range(..self.len).iter() {
            rhs.swap_rows(i.0, i.1)
        }
    }

    /// Applies the inverse of this permutation sequence to the rows of a matrix.
    ///
    /// This method undoes the permutation by applying all recorded swaps in **reverse order**.
    /// This is equivalent to applying the transpose of the permutation matrix P, denoted P^T.
    ///
    /// # What This Does
    ///
    /// If [`permute_rows`](Self::permute_rows) transforms a matrix A to get P*A, then
    /// `inv_permute_rows` transforms P*A back to the original A.
    ///
    /// Mathematically: If P is the permutation matrix, then `inv_permute_rows` applies P^T = P^(-1).
    ///
    /// # Why Reverse Order?
    ///
    /// Since permutations are their own inverse when applied in reverse order, if the
    /// forward permutation applies swaps (i₁, j₁), (i₂, j₂), ..., (iₙ, jₙ), the inverse
    /// applies them as (iₙ, jₙ), ..., (i₂, j₂), (i₁, j₁).
    ///
    /// # Arguments
    ///
    /// * `rhs` - A mutable reference to the matrix whose rows will be permuted
    ///
    /// # Example: Undoing a Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);
    /// p.append_permutation(1, 2);
    ///
    /// let original = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Apply the permutation
    /// let mut m = original.clone();
    /// p.permute_rows(&mut m);
    ///
    /// // The matrix has been permuted
    /// assert_ne!(m, original);
    ///
    /// // Apply the inverse permutation to undo it
    /// p.inv_permute_rows(&mut m);
    ///
    /// // Back to the original matrix
    /// assert_eq!(m, original);
    /// ```
    ///
    /// # Example: LU Decomposition - Recovering Original Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let (p, l, u) = lu.unpack();
    ///
    /// // L*U gives us P*A, not A
    /// let mut lu_product = l * u;
    ///
    /// // Apply inverse permutation to recover A
    /// p.inv_permute_rows(&mut lu_product);
    ///
    /// // Now we have A = P^T * L * U
    /// assert!((lu_product - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving Systems - Alternative Approach
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// let lu = a.lu();
    /// let (p, l, u) = lu.unpack();
    ///
    /// // Alternative to permuting b: compute x = P*b without modifying b
    /// let mut pb = b.clone();
    /// p.permute_rows(&mut pb);
    ///
    /// // Solve L*y = P*b and U*x' = y
    /// let y = l.solve_lower_triangular(&pb).unwrap();
    /// let x = u.solve_upper_triangular(&y).unwrap();
    ///
    /// // x already equals the solution since we permuted b
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Verifying Inverse Property
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 1);
    /// p.append_permutation(1, 2);
    ///
    /// let original = Matrix3::<f64>::identity();
    ///
    /// // Apply permutation then its inverse
    /// let mut m = original.clone();
    /// p.permute_rows(&mut m);
    /// p.inv_permute_rows(&mut m);
    ///
    /// // Should get back to identity
    /// assert_eq!(m, original);
    ///
    /// // Also works in opposite order
    /// let mut m2 = original.clone();
    /// p.inv_permute_rows(&mut m2);
    /// p.permute_rows(&mut m2);
    ///
    /// // Still get back to identity
    /// assert_eq!(m2, original);
    /// ```
    ///
    /// # Example: Computing P^T Explicitly
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);
    ///
    /// // Build P matrix
    /// let mut p_matrix = Matrix3::<f64>::identity();
    /// p.permute_rows(&mut p_matrix);
    ///
    /// // Build P^T matrix
    /// let mut pt_matrix = Matrix3::<f64>::identity();
    /// p.inv_permute_rows(&mut pt_matrix);
    ///
    /// // P^T should be the transpose of P
    /// assert_eq!(pt_matrix, p_matrix.transpose());
    ///
    /// // And P * P^T = I
    /// let identity = p_matrix * pt_matrix;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Performance
    ///
    /// Same as [`permute_rows`](Self::permute_rows): O(k * n) where k is the number of swaps
    /// and n is the number of columns.
    ///
    /// # See Also
    ///
    /// - [`permute_rows`](Self::permute_rows): Apply the forward permutation
    /// - [`inv_permute_columns`](Self::inv_permute_columns): Inverse permutation for columns
    /// - [`determinant`](Self::determinant): The determinant is the same for P and P^T
    #[inline]
    pub fn inv_permute_rows<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for i in 0..self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_rows(i1, i2)
        }
    }

    /// Applies this permutation sequence to the columns of a matrix.
    ///
    /// This method performs all recorded column swaps in the order they were added to the
    /// permutation sequence. Each swap exchanges two columns of the matrix.
    ///
    /// This is the column-wise analogue of [`permute_rows`](Self::permute_rows) and is used
    /// when you need to permute columns instead of rows, such as in:
    /// - **Full pivoting LU decomposition**: Both row and column pivoting
    /// - **Column pivoting QR decomposition**: Reordering columns for better conditioning
    /// - **Matrix reordering algorithms**: Permuting columns for sparse matrices
    ///
    /// # Arguments
    ///
    /// * `rhs` - A mutable reference to the matrix whose columns will be permuted
    ///
    /// # Example: Basic Column Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);  // Swap columns 0 and 2
    ///
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// p.permute_columns(&mut m);
    ///
    /// // Columns 0 and 2 have been swapped
    /// assert_eq!(m[(0, 0)], 3.0);  // Column 0 now contains old column 2
    /// assert_eq!(m[(0, 1)], 2.0);  // Column 1 unchanged
    /// assert_eq!(m[(0, 2)], 1.0);  // Column 2 now contains old column 0
    /// ```
    ///
    /// # Example: Full Pivoting LU
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Full pivoting LU uses both row and column permutations
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    /// let (p, l, u, q) = lu.unpack();
    ///
    /// // Apply both row and column permutations
    /// let mut paq = m.clone();
    /// p.permute_rows(&mut paq);
    /// q.permute_columns(&mut paq);
    ///
    /// // Verify: P*A*Q = L*U
    /// let lu_product = l * u;
    /// assert!((paq - lu_product).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Reordering Matrix Columns
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// // Reverse the order of columns
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);  // Swap outer columns
    /// // Middle column stays in place
    ///
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// p.permute_columns(&mut m);
    ///
    /// // First and last columns are swapped
    /// assert_eq!(m.column(0)[0], 3.0);
    /// assert_eq!(m.column(1)[0], 2.0);
    /// assert_eq!(m.column(2)[0], 1.0);
    /// ```
    ///
    /// # Example: Computing A*Q
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut q = PermutationSequence::<Const<3>>::identity();
    /// q.append_permutation(1, 2);
    ///
    /// // Build the permutation matrix Q
    /// let mut q_matrix = Matrix3::identity();
    /// q.permute_columns(&mut q_matrix);
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Compute A*Q by permuting columns of A
    /// let mut aq = a.clone();
    /// q.permute_columns(&mut aq);
    ///
    /// // Verify it matches A * Q_matrix
    /// let aq_direct = a * q_matrix;
    /// assert_eq!(aq, aq_direct);
    /// ```
    ///
    /// # Performance
    ///
    /// The time complexity is O(k * m) where k is the number of swaps ([`len`](Self::len))
    /// and m is the number of rows in the matrix. Each swap operation is O(m).
    ///
    /// Note: Column swaps can be slower than row swaps due to memory layout (matrices
    /// are stored in column-major order in nalgebra).
    ///
    /// # See Also
    ///
    /// - [`permute_rows`](Self::permute_rows): Apply permutation to rows instead
    /// - [`inv_permute_columns`](Self::inv_permute_columns): Apply the inverse permutation
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    #[inline]
    pub fn permute_columns<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for i in self.ipiv.rows_range(..self.len).iter() {
            rhs.swap_columns(i.0, i.1)
        }
    }

    /// Applies the inverse of this permutation sequence to the columns of a matrix.
    ///
    /// This method undoes column permutations by applying all recorded swaps in **reverse order**.
    /// This is the column-wise analogue of [`inv_permute_rows`](Self::inv_permute_rows).
    ///
    /// # What This Does
    ///
    /// If [`permute_columns`](Self::permute_columns) transforms a matrix A to get A*Q, then
    /// `inv_permute_columns` transforms A*Q back to the original A.
    ///
    /// Mathematically: If Q is the permutation matrix for columns, then `inv_permute_columns`
    /// applies Q^T = Q^(-1).
    ///
    /// # Arguments
    ///
    /// * `rhs` - A mutable reference to the matrix whose columns will be permuted
    ///
    /// # Example: Undoing Column Permutation
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut q = PermutationSequence::<Const<3>>::identity();
    /// q.append_permutation(0, 2);
    /// q.append_permutation(1, 2);
    ///
    /// let original = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Apply column permutation
    /// let mut m = original.clone();
    /// q.permute_columns(&mut m);
    ///
    /// // Columns have been permuted
    /// assert_ne!(m, original);
    ///
    /// // Apply inverse to undo
    /// q.inv_permute_columns(&mut m);
    ///
    /// // Back to original
    /// assert_eq!(m, original);
    /// ```
    ///
    /// # Example: Full Pivoting LU - Recovering Original Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    /// let (p, l, u, q) = lu.unpack();
    ///
    /// // L*U gives us P*A*Q
    /// let mut lu_product = l * u;
    ///
    /// // Apply inverse permutations to recover A
    /// p.inv_permute_rows(&mut lu_product);    // Undo row permutation
    /// q.inv_permute_columns(&mut lu_product); // Undo column permutation
    ///
    /// // Now we have A = P^T * L * U * Q^T
    /// assert!((lu_product - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Verifying Inverse Property for Columns
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut q = PermutationSequence::<Const<3>>::identity();
    /// q.append_permutation(0, 1);
    /// q.append_permutation(1, 2);
    ///
    /// let original = Matrix3::<f64>::identity();
    ///
    /// // Apply permutation then inverse
    /// let mut m = original.clone();
    /// q.permute_columns(&mut m);
    /// q.inv_permute_columns(&mut m);
    ///
    /// assert_eq!(m, original);
    ///
    /// // Also works in opposite order
    /// let mut m2 = original.clone();
    /// q.inv_permute_columns(&mut m2);
    /// q.permute_columns(&mut m2);
    ///
    /// assert_eq!(m2, original);
    /// ```
    ///
    /// # Performance
    ///
    /// Same as [`permute_columns`](Self::permute_columns): O(k * m) where k is the number
    /// of swaps and m is the number of rows.
    ///
    /// # See Also
    ///
    /// - [`permute_columns`](Self::permute_columns): Apply the forward permutation
    /// - [`inv_permute_rows`](Self::inv_permute_rows): Inverse permutation for rows
    /// - [`determinant`](Self::determinant): The determinant is the same for Q and Q^T
    #[inline]
    pub fn inv_permute_columns<T: Scalar, R2: Dim, C2: Dim, S2>(
        &self,
        rhs: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
    {
        for i in 0..self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_columns(i1, i2)
        }
    }

    /// Returns the number of swaps recorded in this permutation sequence.
    ///
    /// This counts only the non-identity swaps (where the two indices differ). Swaps
    /// where both indices are the same are not counted.
    ///
    /// # Returns
    ///
    /// The number of swaps that will be applied when calling [`permute_rows`](Self::permute_rows)
    /// or [`permute_columns`](Self::permute_columns).
    ///
    /// # Example: Counting Swaps
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const};
    ///
    /// let mut p = PermutationSequence::<Const<5>>::identity();
    ///
    /// // Initially empty
    /// assert_eq!(p.len(), 0);
    ///
    /// // Add some swaps
    /// p.append_permutation(0, 3);
    /// assert_eq!(p.len(), 1);
    ///
    /// p.append_permutation(1, 4);
    /// assert_eq!(p.len(), 2);
    ///
    /// // Identity swaps don't count
    /// p.append_permutation(2, 2);
    /// assert_eq!(p.len(), 2);  // Still 2, not 3
    /// ```
    ///
    /// # Example: In LU Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let p = lu.p();
    ///
    /// // Check how many row swaps were needed during decomposition
    /// println!("Number of pivoting swaps: {}", p.len());
    ///
    /// // The determinant sign depends on this count
    /// let det_sign: f64 = p.determinant();
    /// if p.len() % 2 == 0 {
    ///     assert_eq!(det_sign, 1.0);
    /// } else {
    ///     assert_eq!(det_sign, -1.0);
    /// }
    /// ```
    ///
    /// # Example: Maximum Length
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const};
    ///
    /// // A permutation sequence of dimension n can have at most n swaps
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    ///
    /// p.append_permutation(0, 1);
    /// p.append_permutation(1, 2);
    /// p.append_permutation(0, 2);
    ///
    /// assert_eq!(p.len(), 3);
    ///
    /// // Attempting to add more swaps would panic
    /// // p.append_permutation(0, 1); // This would panic!
    /// ```
    ///
    /// # See Also
    ///
    /// - [`is_empty`](Self::is_empty): Check if the sequence has no swaps
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    /// - [`determinant`](Self::determinant): The sign changes with each swap
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this permutation sequence contains no swaps.
    ///
    /// An empty permutation sequence is equivalent to an identity permutation - it
    /// leaves any matrix unchanged when applied.
    ///
    /// # Returns
    ///
    /// `true` if no swaps have been recorded, `false` otherwise.
    ///
    /// # Example: Checking for Identity
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    ///
    /// // Identity permutation is empty
    /// assert!(p.is_empty());
    /// assert_eq!(p.len(), 0);
    ///
    /// // Add a swap
    /// p.append_permutation(0, 1);
    ///
    /// // No longer empty
    /// assert!(!p.is_empty());
    /// assert_eq!(p.len(), 1);
    /// ```
    ///
    /// # Example: Testing Matrix Unchanged
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// let p = PermutationSequence::<Const<3>>::identity();
    ///
    /// if p.is_empty() {
    ///     // We know the permutation won't change the matrix
    ///     let mut m = Matrix3::new(
    ///         1.0, 2.0, 3.0,
    ///         4.0, 5.0, 6.0,
    ///         7.0, 8.0, 9.0,
    ///     );
    ///     let original = m.clone();
    ///
    ///     p.permute_rows(&mut m);
    ///     assert_eq!(m, original);
    /// }
    /// ```
    ///
    /// # Example: Optimizing Operations
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Matrix3, Const};
    ///
    /// fn apply_permutation_if_needed(
    ///     p: &PermutationSequence<Const<3>>,
    ///     m: &mut Matrix3<f64>,
    /// ) {
    ///     // Skip the work if there's nothing to do
    ///     if !p.is_empty() {
    ///         p.permute_rows(m);
    ///     }
    /// }
    ///
    /// let p_identity = PermutationSequence::<Const<3>>::identity();
    /// let mut m = Matrix3::identity();
    ///
    /// // No work done for identity permutation
    /// apply_permutation_if_needed(&p_identity, &mut m);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`len`](Self::len): Get the number of swaps
    /// - [`append_permutation`](Self::append_permutation): Add swaps to the sequence
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Computes the determinant of the permutation matrix.
    ///
    /// The determinant of a permutation matrix is always +1 or -1, depending on whether
    /// the permutation is even or odd:
    /// - **Even permutation** (even number of swaps): determinant = +1
    /// - **Odd permutation** (odd number of swaps): determinant = -1
    ///
    /// This is important for computing matrix determinants using LU decomposition, where
    /// `det(A) = det(P) * det(L) * det(U)`.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The scalar type for the result (typically `f64`, `f32`, or `i32`)
    ///
    /// # Returns
    ///
    /// Either `T::one()` (if even number of swaps) or `-T::one()` (if odd number of swaps).
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const};
    ///
    /// let mut p = PermutationSequence::<Const<4>>::identity();
    ///
    /// // No swaps: even permutation
    /// assert_eq!(p.determinant::<i32>(), 1);
    ///
    /// // One swap: odd permutation
    /// p.append_permutation(0, 1);
    /// assert_eq!(p.determinant::<i32>(), -1);
    ///
    /// // Two swaps: even permutation
    /// p.append_permutation(2, 3);
    /// assert_eq!(p.determinant::<i32>(), 1);
    ///
    /// // Three swaps: odd permutation
    /// p.append_permutation(1, 2);
    /// assert_eq!(p.determinant::<i32>(), -1);
    /// ```
    ///
    /// # Example: Computing Matrix Determinant with LU
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let lu = m.lu();
    ///
    /// // The determinant formula: det(A) = det(P) * det(L) * det(U)
    /// // Since det(L) = 1 (unit diagonal), we have: det(A) = det(P) * det(U)
    /// let det_p: f64 = lu.p().determinant();
    /// let det_u = (0..3).map(|i| lu.lu_internal()[(i, i)]).product::<f64>();
    /// let det_a = det_p * det_u;
    ///
    /// // Compare with direct computation
    /// let det_direct = lu.determinant();
    /// assert!((det_a - det_direct).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding Sign Changes
    ///
    /// ```
    /// use nalgebra::{PermutationSequence, Const, Matrix3};
    ///
    /// // Build a permutation matrix explicitly
    /// let mut p = PermutationSequence::<Const<3>>::identity();
    /// p.append_permutation(0, 2);  // One swap
    ///
    /// let mut p_matrix = Matrix3::identity();
    /// p.permute_rows(&mut p_matrix);
    ///
    /// // The determinant of the matrix form should match
    /// let det_from_sequence: f64 = p.determinant();
    /// let det_from_matrix = p_matrix.determinant();
    ///
    /// assert_eq!(det_from_sequence, det_from_matrix);
    /// assert_eq!(det_from_sequence, -1.0);  // Odd permutation
    /// ```
    ///
    /// # Example: Relationship to Matrix Determinant
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let lu = m.lu();
    ///
    /// // For the identity matrix (no pivoting needed)
    /// // The permutation should be identity with determinant +1
    /// if lu.p().is_empty() {
    ///     assert_eq!(lu.p().determinant::<f64>(), 1.0);
    /// }
    ///
    /// // The determinant of the identity matrix is 1
    /// assert_eq!(lu.determinant(), 1.0);
    /// ```
    ///
    /// # Mathematical Background
    ///
    /// Every permutation can be expressed as a composition of transpositions (swaps).
    /// A permutation is called:
    /// - **Even** if it can be expressed as an even number of transpositions
    /// - **Odd** if it requires an odd number of transpositions
    ///
    /// The sign of a permutation σ, denoted sgn(σ), is:
    /// - sgn(σ) = +1 for even permutations
    /// - sgn(σ) = -1 for odd permutations
    ///
    /// This sign is precisely the determinant of the permutation matrix.
    ///
    /// # See Also
    ///
    /// - [`len`](Self::len): Get the number of swaps (determines even/odd)
    /// - [`LU::determinant`]: Compute full matrix determinant using LU decomposition
    #[inline]
    #[must_use]
    pub fn determinant<T: One + ClosedNeg>(&self) -> T {
        if self.len % 2 == 0 {
            T::one()
        } else {
            -T::one()
        }
    }
}
