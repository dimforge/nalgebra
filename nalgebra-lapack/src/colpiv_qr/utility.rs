use super::DiagonalKind;
use super::Side;
use super::TriangularStructure;

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
