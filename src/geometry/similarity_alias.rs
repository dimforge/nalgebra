use core::dimension::{U2, U3};

use geometry::{Similarity, UnitQuaternion, UnitComplex, Rotation2, Rotation3};

/// A 2-dimensional similarity.
pub type Similarity2<N> = Similarity<N, U2, UnitComplex<N>>;

/// A 3-dimensional similarity.
pub type Similarity3<N> = Similarity<N, U3, UnitQuaternion<N>>;

/// A 2-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix2<N> = Similarity<N, U2, Rotation2<N>>;

/// A 3-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix3<N> = Similarity<N, U3, Rotation3<N>>;
