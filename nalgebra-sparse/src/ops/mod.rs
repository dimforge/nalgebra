//! TODO

pub mod serial;

/// TODO
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transposition {
    /// TODO
    Transpose,
    /// TODO
    NoTranspose,
}

impl Transposition {
    /// TODO
    pub fn is_transpose(&self) -> bool {
        self == &Self::Transpose
    }

    /// TODO
    pub fn from_bool(transpose: bool) -> Self {
        if transpose { Self::Transpose } else { Self::NoTranspose }
    }
}

/// TODO
pub fn transpose() -> Transposition {
    Transposition::Transpose
}

/// TODO
pub fn no_transpose() -> Transposition {
    Transposition::NoTranspose
}