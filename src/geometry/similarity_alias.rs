use core::MatrixArray;
use core::dimension::{U1, U2, U3};

use geometry::{Rotation, SimilarityBase, UnitQuaternion, UnitComplex};

/// A D-dimensional similarity.
pub type Similarity<N, D> = SimilarityBase<N, D, MatrixArray<N, D, U1>, Rotation<N, D>>;

/// A 2-dimensional similarity.
pub type Similarity2<N> = SimilarityBase<N, U2, MatrixArray<N, U2, U1>, UnitComplex<N>>;

/// A 3-dimensional similarity.
pub type Similarity3<N> = SimilarityBase<N, U3, MatrixArray<N, U3, U1>, UnitQuaternion<N>>;

/// A 2-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix2<N> = Similarity<N, U2>;

/// A 3-dimensional similarity using a rotation matrix for its rotation part.
pub type SimilarityMatrix3<N> = Similarity<N, U3>;
