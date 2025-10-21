/// Indicates the side from which a matrix multiplication is to be performed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Side {
    /// Perform multiplication from the left.
    Left,
    /// Perform multiplication from the right.
    Right,
}

/// Indicates whether or not to transpose a matrix during a matrix
/// operation.
// @note(geo-ant) once we add complex, we can refactor this
// to conjugate transpose (or hermitian transpose).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Transposition {
    /// Don't transpose, i.e. leave the matrix as is.
    No,
    /// Transpose the matrix.
    Transpose,
}

/// Describes the type of a triangular matrix.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TriangularStructure {
    /// Upper triangular.
    Upper,
    /// Lower triangular.
    Lower,
}

/// Property of the diagonal of a triangular matrix.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DiagonalKind {
    /// Diagonal entries all have value of 1.
    Unit,
    /// Diagonal elements are arbitrary.
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
