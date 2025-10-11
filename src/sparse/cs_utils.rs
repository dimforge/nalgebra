use crate::allocator::Allocator;
use crate::{DefaultAllocator, Dim, OVector};

/// Computes cumulative sum (prefix sum) of vector elements, storing both forms.
///
/// This utility function computes the cumulative sum (also called prefix sum or scan)
/// of a vector's elements. It's a fundamental operation used in sparse matrix algorithms,
/// particularly for converting from one sparse format to another.
///
/// # What It Does
///
/// Given an input vector `a = [a₀, a₁, a₂, ..., aₙ₋₁]`, this function:
/// 1. Computes prefix sums: `b[i] = a₀ + a₁ + ... + aᵢ₋₁` (sum of all elements before position i)
/// 2. Overwrites `a` with the same prefix sums (so `a[i] = b[i]` after the call)
/// 3. Returns the total sum of all original elements
///
/// # Parameters
///
/// - `a`: Input vector, will be overwritten with prefix sums
/// - `b`: Output vector, will contain the prefix sums
///
/// Both vectors must have the same length (panics otherwise).
///
/// # Returns
///
/// The total sum of all original elements in `a` (equivalent to `b[b.len()] if b had one more element`).
///
/// # Mathematical Definition
///
/// ```text
/// Before: a = [a₀, a₁, a₂, ..., aₙ₋₁]
/// After:  a = [0, a₀, a₀+a₁, ..., sum(a₀..aₙ₋₂)]
///         b = [0, a₀, a₀+a₁, ..., sum(a₀..aₙ₋₂)]
/// Return: sum(a₀..aₙ₋₁)
/// ```
///
/// More precisely:
/// - `b[0] = 0`
/// - `b[i] = a₀ + a₁ + ... + aᵢ₋₁` for `i > 0`
/// - Return value = `a₀ + a₁ + ... + aₙ₋₁`
///
/// # Why This Is Useful for Sparse Matrices
///
/// In Compressed Sparse Column (CSC) format, you often need to convert counts
/// per column into column pointers (offsets). The cumulative sum does exactly this:
/// - Input: count of non-zeros in each column
/// - Output: starting index for each column's data
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use nalgebra::DVector;
///
/// // Column counts: column 0 has 2 elements, column 1 has 0, column 2 has 3
/// let mut counts = DVector::from_vec(vec![2, 0, 3]);
/// let mut pointers = DVector::zeros(3);
///
/// // Note: cumsum is not public, but this shows the concept
/// // let total = nalgebra::sparse::cs_utils::cumsum(&mut counts, &mut pointers);
///
/// // After cumsum:
/// // pointers = [0, 2, 2]  (column 0 starts at 0, column 1 at 2, column 2 at 2)
/// // counts = [0, 2, 2]    (same as pointers, overwritten)
/// // total = 5             (total number of elements)
/// ```
///
/// ## Understanding the Transformation
///
/// ```
/// use nalgebra::DVector;
///
/// let mut a = DVector::from_vec(vec![3, 1, 4, 1, 5]);
/// let mut b = DVector::zeros(5);
///
/// // Conceptually (cumsum is internal):
/// // let sum = cumsum(&mut a, &mut b);
///
/// // Results:
/// // b = [0, 3, 4, 8, 9]   (cumulative sums: 0, 0+3, 0+3+1, 0+3+1+4, etc.)
/// // a = [0, 3, 4, 8, 9]   (same as b)
/// // sum = 14              (3 + 1 + 4 + 1 + 5)
/// ```
///
/// ## Sparse Matrix Column Pointer Construction
///
/// This is the typical use case within nalgebra's sparse matrix code:
///
/// ```
/// use nalgebra::DVector;
///
/// // We've counted non-zeros per column during matrix construction
/// let mut column_counts = DVector::from_vec(vec![2, 3, 0, 1, 4]);
/// let mut column_pointers = DVector::zeros(5);
///
/// // cumsum converts counts to starting indices
/// // let total_nnz = cumsum(&mut column_counts, &mut column_pointers);
///
/// // Now we can use column_pointers to know where each column's data starts:
/// // Column 0: indices [0..2)   (2 elements starting at index 0)
/// // Column 1: indices [2..5)   (3 elements starting at index 2)
/// // Column 2: indices [5..5)   (0 elements starting at index 5)
/// // Column 3: indices [5..6)   (1 element starting at index 5)
/// // Column 4: indices [6..10)  (4 elements starting at index 6)
/// // total_nnz = 10
/// ```
///
/// ## From Counts to Offsets
///
/// ```
/// use nalgebra::DVector;
///
/// // Real-world scenario: building a sparse matrix
/// // Step 1: Count non-zeros in each column
/// let mut counts = DVector::from_vec(vec![
///     3,  // column 0 has 3 non-zeros
///     0,  // column 1 has 0 non-zeros
///     2,  // column 2 has 2 non-zeros
///     1,  // column 3 has 1 non-zero
/// ]);
///
/// let mut offsets = DVector::zeros(4);
///
/// // Convert to offset format
/// // let total = cumsum(&mut counts, &mut offsets);
///
/// // Result:
/// // offsets = [0, 3, 3, 5]
/// // This means:
/// //   - Column 0 data: values[0..3]
/// //   - Column 1 data: values[3..3] (empty)
/// //   - Column 2 data: values[3..5]
/// //   - Column 3 data: values[5..6]
/// // total = 6 (need array of length 6 for all values)
/// ```
///
/// # Algorithm Details
///
/// This is a simple sequential scan with O(n) time complexity and O(1) extra space.
/// The implementation is cache-friendly and very fast for the sizes typically encountered.
///
/// The reason both `a` and `b` are modified to contain the same result is historical:
/// - `b` is the natural output (prefix sums)
/// - `a` is overwritten to save memory in sparse matrix construction algorithms
/// - This avoids needing a temporary vector
///
/// # Implementation Note
///
/// This is an internal utility function (marked `pub(crate)` in the module) used by
/// sparse matrix construction routines. It's not part of the public API, but understanding
/// it helps when working with sparse matrices or reading nalgebra's source code.
///
/// # Use Cases Within Nalgebra
///
/// - **CSC Matrix Construction**: Converting column counts to column pointers
/// - **Triplet to CSC Conversion**: Building the column pointer array
/// - **Matrix Transpose**: Computing new column pointers from row counts
/// - **Format Conversions**: Any sparse format transformation needing offset arrays
///
/// # Performance
///
/// - Time: O(n) where n is the length of the vectors
/// - Space: O(1) additional space (modifies inputs in-place)
/// - Cache-friendly: sequential memory access pattern
///
/// For typical sparse matrices with 100-10,000 columns, this operation is extremely fast
/// (microseconds) and not a performance bottleneck.
///
/// # See Also
///
/// - [`CsMatrix::from_triplet`] - Uses cumsum internally for column pointer construction
/// - [`CsMatrix::transpose`] - Uses cumsum to build transposed column pointers
pub fn cumsum<D: Dim>(a: &mut OVector<usize, D>, b: &mut OVector<usize, D>) -> usize
where
    DefaultAllocator: Allocator<D>,
{
    assert!(a.len() == b.len());
    let mut sum = 0;

    for i in 0..a.len() {
        b[i] = sum;
        sum += a[i];
        a[i] = b[i];
    }

    sum
}
