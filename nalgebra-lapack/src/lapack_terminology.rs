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

impl Side {
    pub(crate) fn into_lapack_side_character(self) -> u8 {
        match self {
            Side::Left => b'L',
            Side::Right => b'R',
        }
    }
}

impl TriangularStructure {
    pub(crate) fn into_lapack_uplo_character(self) -> u8 {
        match self {
            Self::Upper => b'U',
            Self::Lower => b'L',
        }
    }
}

impl DiagonalKind {
    pub(crate) fn into_lapack_diag_character(self) -> u8 {
        match self {
            DiagonalKind::Unit => b'U',
            DiagonalKind::NonUnit => b'N',
        }
    }
}
