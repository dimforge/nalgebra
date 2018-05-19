use base::dimension::{U2, U3};

use geometry::Translation;

/// A 2-dimensional translation.
pub type Translation2<N> = Translation<N, U2>;

/// A 3-dimensional translation.
pub type Translation3<N> = Translation<N, U3>;
