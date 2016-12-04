use core::MatrixArray;
use core::dimension::{U1, U2, U3};

use geometry::TranslationBase;

/// A D-dimensional translation.
pub type Translation<N, D> = TranslationBase<N, D, MatrixArray<N, D, U1>>;

/// A 2-dimensional translation.
pub type Translation2<N> = Translation<N, U2>;

/// A 3-dimensional translation.
pub type Translation3<N> = Translation<N, U3>;
