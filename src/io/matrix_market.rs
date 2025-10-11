use std::fs;
use std::path::Path;

use crate::RealField;
use crate::sparse::CsMatrix;
use pest::Parser;

#[derive(Parser)]
#[grammar = "io/matrix_market.pest"]
struct MatrixMarketParser;

// TODO: return an Error instead of an Option.
/// Loads a sparse matrix from a Matrix Market format file.
///
/// **What is Matrix Market format?**
///
/// Matrix Market is a widely-used standard text format for storing sparse matrices.
/// It's particularly popular in scientific computing because it's human-readable and
/// supported by many numerical software packages (MATLAB, SciPy, Julia, etc.).
///
/// The format consists of:
/// - A header line starting with `%%MatrixMarket` that describes the matrix type
/// - Optional comment lines starting with `%`
/// - A shape line with: `rows columns non_zero_entries`
/// - Data lines with: `row_index column_index value` (indices are 1-based)
///
/// # Parameters
///
/// * `path` - Path to the Matrix Market file (`.mtx` extension is common but not required)
///
/// # Returns
///
/// * `Some(CsMatrix<T>)` - A compressed sparse column matrix if parsing succeeds
/// * `None` - If the file cannot be read or contains invalid data
///
/// # Type Parameters
///
/// * `T` - The scalar type for matrix elements. Must implement [`RealField`] (e.g., `f32`, `f64`)
/// * `P` - Any type that can be converted to a file path (e.g., `&str`, `String`, `Path`)
///
/// # Matrix Market Format Example
///
/// ```text
/// %%MatrixMarket matrix coordinate real general
/// % This is a comment
/// 3 3 4
/// 1 1 1.0
/// 2 2 2.5
/// 3 1 3.0
/// 3 3 4.0
/// ```
///
/// This represents a 3×3 matrix with 4 non-zero entries. Note that row and column
/// indices are 1-based in the file but will be converted to 0-based internally.
///
/// # Examples
///
/// ## Loading a matrix from a file
///
/// ```no_run
/// use nalgebra::io::cs_matrix_from_matrix_market;
/// use nalgebra::DMatrix;
///
/// // Load a sparse matrix from a Matrix Market file
/// let sparse_matrix = cs_matrix_from_matrix_market::<f64, _>("data/matrix.mtx")
///     .expect("Failed to load matrix");
///
/// println!("Loaded {}×{} sparse matrix with {} non-zero entries",
///          sparse_matrix.nrows(),
///          sparse_matrix.ncols(),
///          sparse_matrix.len());
///
/// // Convert to dense format if needed
/// let dense_matrix: DMatrix<f64> = sparse_matrix.into();
/// ```
///
/// ## Loading with explicit type annotation
///
/// ```no_run
/// use nalgebra::io::cs_matrix_from_matrix_market;
/// use std::path::Path;
///
/// // Load as f32 instead of f64
/// let matrix = cs_matrix_from_matrix_market::<f32, _>("physics_simulation.mtx");
///
/// match matrix {
///     Some(m) => println!("Successfully loaded matrix with {} entries", m.len()),
///     None => eprintln!("Failed to load or parse matrix file"),
/// }
/// ```
///
/// ## Working with different path types
///
/// ```no_run
/// use nalgebra::io::cs_matrix_from_matrix_market;
/// use std::path::{Path, PathBuf};
///
/// // Using &str
/// let matrix1 = cs_matrix_from_matrix_market::<f64, _>("matrix1.mtx");
///
/// // Using String
/// let filename = String::from("matrix2.mtx");
/// let matrix2 = cs_matrix_from_matrix_market::<f64, _>(filename);
///
/// // Using Path
/// let path = Path::new("matrix3.mtx");
/// let matrix3 = cs_matrix_from_matrix_market::<f64, _>(path);
///
/// // Using PathBuf
/// let mut pathbuf = PathBuf::from("data");
/// pathbuf.push("matrix4.mtx");
/// let matrix4 = cs_matrix_from_matrix_market::<f64, _>(pathbuf);
/// ```
///
/// # Common Use Cases
///
/// - **Loading test matrices**: Many standard test matrices for numerical algorithms
///   are available in Matrix Market format from repositories like SuiteSparse Matrix Collection
/// - **Data interchange**: Sharing sparse matrix data between different software packages
/// - **Reproducible research**: Storing matrices in a portable, human-readable format
/// - **Large-scale computations**: Matrix Market format is efficient for sparse matrices
///   where most elements are zero
///
/// # Performance Notes
///
/// Matrix Market files are text-based and parsing can be slower than binary formats.
/// For performance-critical applications with large matrices, consider:
/// - Using binary formats for repeated loading
/// - Pre-processing matrices and saving in a more efficient format
/// - Loading matrices once and caching them in memory
///
/// # See Also
///
/// * [`cs_matrix_from_matrix_market_str`] - Parse Matrix Market data from a string instead of a file
/// * [`CsMatrix`] - The compressed sparse column matrix type returned by this function
///
/// [`RealField`]: crate::RealField
/// [`CsMatrix`]: crate::sparse::CsMatrix
pub fn cs_matrix_from_matrix_market<T: RealField, P: AsRef<Path>>(path: P) -> Option<CsMatrix<T>> {
    let file = fs::read_to_string(path).ok()?;
    cs_matrix_from_matrix_market_str(&file)
}

// TODO: return an Error instead of an Option.
/// Parses a sparse matrix from a string containing Matrix Market format data.
///
/// This function is similar to [`cs_matrix_from_matrix_market`] but operates on in-memory
/// string data instead of reading from a file. This is useful when the Matrix Market data
/// is embedded in your code, received over a network, or generated dynamically.
///
/// **What is Matrix Market format?**
///
/// Matrix Market is a standard text format for exchanging sparse matrices between different
/// software packages. The format is:
/// 1. **Header line**: `%%MatrixMarket matrix coordinate real general`
/// 2. **Comments** (optional): Lines starting with `%`
/// 3. **Shape line**: `rows columns non_zero_count`
/// 4. **Data entries**: One per line as `row_index column_index value`
///
/// Important: Row and column indices in Matrix Market files are **1-based** (start at 1),
/// but they are automatically converted to 0-based indices used internally by nalgebra.
///
/// # Parameters
///
/// * `data` - A string slice containing Matrix Market formatted data
///
/// # Returns
///
/// * `Some(CsMatrix<T>)` - A compressed sparse column matrix if parsing succeeds
/// * `None` - If the string contains invalid Matrix Market data
///
/// # Type Parameters
///
/// * `T` - The scalar type for matrix elements. Must implement [`RealField`] (e.g., `f32`, `f64`)
///
/// # Examples
///
/// ## Basic usage with embedded data
///
/// ```
/// use nalgebra::io::cs_matrix_from_matrix_market_str;
/// use nalgebra::DMatrix;
///
/// // Define a small sparse matrix inline
/// let matrix_data = r#"
/// %%MatrixMarket matrix coordinate real general
/// 3 3 3
/// 1 1 1.0
/// 2 2 2.0
/// 3 3 3.0
/// "#;
///
/// let sparse = cs_matrix_from_matrix_market_str::<f64>(matrix_data)
///     .expect("Failed to parse matrix");
///
/// assert_eq!(sparse.nrows(), 3);
/// assert_eq!(sparse.ncols(), 3);
/// assert_eq!(sparse.len(), 3); // 3 non-zero entries
///
/// // Convert to dense to verify values
/// let dense: DMatrix<f64> = sparse.into();
/// assert_eq!(dense[(0, 0)], 1.0);
/// assert_eq!(dense[(1, 1)], 2.0);
/// assert_eq!(dense[(2, 2)], 3.0);
/// ```
///
/// ## Working with comments and formatting
///
/// ```
/// use nalgebra::io::cs_matrix_from_matrix_market_str;
///
/// // Matrix Market format supports comments
/// let data_with_comments = r#"
/// %%MatrixMarket matrix coordinate real general
/// % This is a 2x2 identity matrix
/// % Author: Example
/// % Date: 2024
/// 2 2 2
/// 1 1 1.0
/// 2 2 1.0
/// "#;
///
/// let matrix = cs_matrix_from_matrix_market_str::<f64>(data_with_comments)
///     .expect("Comments should be ignored");
///
/// assert_eq!(matrix.nrows(), 2);
/// assert_eq!(matrix.ncols(), 2);
/// ```
///
/// ## Creating a sparse matrix representation
///
/// ```
/// use nalgebra::io::cs_matrix_from_matrix_market_str;
///
/// // A 4x4 tridiagonal matrix
/// let tridiagonal = r#"
/// %%MatrixMarket matrix coordinate real general
/// 4 4 10
/// 1 1 2.0
/// 1 2 -1.0
/// 2 1 -1.0
/// 2 2 2.0
/// 2 3 -1.0
/// 3 2 -1.0
/// 3 3 2.0
/// 3 4 -1.0
/// 4 3 -1.0
/// 4 4 2.0
/// "#;
///
/// let matrix = cs_matrix_from_matrix_market_str::<f64>(tridiagonal)
///     .expect("Valid tridiagonal matrix");
///
/// println!("Created {}x{} tridiagonal matrix with {} non-zeros",
///          matrix.nrows(), matrix.ncols(), matrix.len());
/// ```
///
/// ## Testing with different data types
///
/// ```
/// use nalgebra::io::cs_matrix_from_matrix_market_str;
///
/// let data = r#"
/// %%MatrixMarket matrix coordinate real general
/// 2 2 2
/// 1 1 1.5
/// 2 2 2.5
/// "#;
///
/// // Parse as f64 (default)
/// let matrix_f64 = cs_matrix_from_matrix_market_str::<f64>(data).unwrap();
///
/// // Parse as f32 for lower precision
/// let matrix_f32 = cs_matrix_from_matrix_market_str::<f32>(data).unwrap();
/// ```
///
/// ## Handling parse failures
///
/// ```
/// use nalgebra::io::cs_matrix_from_matrix_market_str;
///
/// // Invalid data - missing header
/// let invalid_data = "1 1 1.0";
/// assert!(cs_matrix_from_matrix_market_str::<f64>(invalid_data).is_none());
///
/// // Invalid data - malformed entry
/// let malformed = r#"
/// %%MatrixMarket matrix coordinate real general
/// 2 2 1
/// 1 1 not_a_number
/// "#;
/// assert!(cs_matrix_from_matrix_market_str::<f64>(malformed).is_none());
/// ```
///
/// # Understanding Compressed Sparse Column Format
///
/// The returned [`CsMatrix`] uses Compressed Sparse Column (CSC) format, which stores:
/// - Only non-zero values (saves memory for sparse matrices)
/// - Column pointers indicating where each column starts
/// - Row indices for each non-zero value
///
/// This format is highly efficient for:
/// - Column-oriented operations
/// - Matrix-vector multiplication
/// - Linear system solving
/// - Sparse matrix storage (when most elements are zero)
///
/// # Common Use Cases
///
/// - **Unit Testing**: Embed test matrices directly in test code
/// - **Configuration Files**: Load matrix configurations from strings
/// - **Network Data**: Parse matrices received via HTTP, WebSocket, etc.
/// - **String Templates**: Generate matrices programmatically
/// - **Documentation Examples**: Include complete, runnable examples
///
/// # Differences from `cs_matrix_from_matrix_market`
///
/// | Feature | `cs_matrix_from_matrix_market_str` | `cs_matrix_from_matrix_market` |
/// |---------|-----------------------------------|-------------------------------|
/// | Input   | String data in memory             | File path                     |
/// | I/O     | No file system access             | Reads from disk               |
/// | Use case| Embedded data, testing, networking| Loading saved matrices        |
/// | Error handling | Returns None on parse error | Returns None on I/O or parse error |
///
/// # Performance Notes
///
/// Parsing Matrix Market format is relatively fast for moderate-sized matrices.
/// For matrices with millions of entries:
/// - Parsing may take noticeable time (proportional to entry count)
/// - Consider caching the parsed result
/// - For repeated use, convert once and store the `CsMatrix`
///
/// # See Also
///
/// * [`cs_matrix_from_matrix_market`] - Load Matrix Market data from a file instead
/// * [`CsMatrix`] - The compressed sparse column matrix type returned by this function
/// * [`CsMatrix::from_triplet`] - Alternative way to construct sparse matrices from coordinate format
///
/// [`RealField`]: crate::RealField
/// [`CsMatrix`]: crate::sparse::CsMatrix
/// [`CsMatrix::from_triplet`]: crate::sparse::CsMatrix::from_triplet
pub fn cs_matrix_from_matrix_market_str<T: RealField>(data: &str) -> Option<CsMatrix<T>> {
    let file = MatrixMarketParser::parse(Rule::Document, data)
        .unwrap()
        .next()?;
    let mut shape = (0, 0, 0);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut data: Vec<T> = Vec::new();

    for line in file.into_inner() {
        match line.as_rule() {
            Rule::Header => {}
            Rule::Shape => {
                let mut inner = line.into_inner();
                shape.0 = inner.next()?.as_str().parse::<usize>().ok()?;
                shape.1 = inner.next()?.as_str().parse::<usize>().ok()?;
                shape.2 = inner.next()?.as_str().parse::<usize>().ok()?;
            }
            Rule::Entry => {
                let mut inner = line.into_inner();
                // NOTE: indices are 1-based.
                rows.push(inner.next()?.as_str().parse::<usize>().ok()? - 1);
                cols.push(inner.next()?.as_str().parse::<usize>().ok()? - 1);
                data.push(crate::convert(inner.next()?.as_str().parse::<f64>().ok()?));
            }
            _ => return None, // TODO: return an Err instead.
        }
    }

    Some(CsMatrix::from_triplet(
        shape.0, shape.1, &rows, &cols, &data,
    ))
}
