use crate::geometry::{Rotation2, Rotation3, Similarity, UnitComplex, UnitQuaternion};

/// A 2-dimensional similarity.
pub type Similarity2<N> = Similarity<N, UnitComplex<N>, 2>;

/// A 3-dimensional similarity.
pub type Similarity3<N> = Similarity<N, UnitQuaternion<N>, 3>;

/// A 2-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix2<N> = Similarity<N, Rotation2<N>, 2>;

/// A 3-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix3<N> = Similarity<N, Rotation3<N>, 3>;
