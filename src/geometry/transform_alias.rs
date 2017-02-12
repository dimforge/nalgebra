use core::MatrixArray;
use core::dimension::{U1, U2, U3, DimNameSum};

use geometry::{TransformBase, TGeneral, TProjective, TAffine};

/// A `D`-dimensional general transformation that may not be inversible. Stored as an homogeneous
/// `(D + 1) × (D + 1)` matrix.
pub type Transform<N, D> = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TGeneral>;

/// An inversible `D`-dimensional general transformation. Stored as an homogeneous
/// `(D + 1) × (D + 1)` matrix.
pub type Projective<N, D> = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TProjective>;

/// A `D`-dimensional affine transformation. Stored as an homogeneous `(D + 1) × (D + 1)` matrix.
pub type Affine<N, D> = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TAffine>;

/// A 2D general transformation that may not be inversible. Stored as an homogeneous 3x3 matrix.
pub type Transform2<N> = Transform<N, U2>;
/// An inversible 2D general transformation. Stored as an homogeneous 3x3 matrix.
pub type Projective2<N> = Projective<N, U2>;
/// A 2D affine transformation. Stored as an homogeneous 3x3 matrix.
pub type Affine2<N> = Affine<N, U2>;

/// A 3D general transformation that may not be inversible. Stored as an homogeneous 4x4 matrix.
pub type Transform3<N> = Transform<N, U3>;
/// An inversible 3D general transformation. Stored as an homogeneous 4x4 matrix.
pub type Projective3<N> = Projective<N, U3>;
/// A 3D affine transformation. Stored as an homogeneous 4x4 matrix.
pub type Affine3<N> = Affine<N, U3>;
