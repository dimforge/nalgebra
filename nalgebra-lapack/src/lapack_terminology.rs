#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// Indicates the side from which a matrix multiplication is to be performed.
pub enum Side {
    /// perform multiplication from the left
    Left,
    /// perform multiplication from the right
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// Indicates whether or not to transpose a matrix during a matrix
/// operation.
// @note(geo-ant) once we add complex, we can refactor this
// to conjugate transpose (or hermitian transpose).
pub enum Transposition {
    /// don't transpose, i.e. leave the matrix as is
    No,
    /// transpose the matrix.
    Transpose,
}

/// describes the type of a triangular matrix
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TriangularStructure {
    /// upper triangular
    Upper,
    /// lower triangular
    Lower,
}

/// property of the diagonal of a triangular matrix
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DiagonalKind {
    /// diagonal entries all have value of 1
    Unit,
    /// diagonal elements are arbitrary
    NonUnit,
}
