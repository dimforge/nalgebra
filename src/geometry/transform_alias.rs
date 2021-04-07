use crate::geometry::{TAffine, TGeneral, TProjective, Transform};

/// A 2D general transformation that may not be invertible. Stored as a homogeneous 3x3 matrix.
pub type Transform2<N> = Transform<N, TGeneral, 2>;
/// An invertible 2D general transformation. Stored as a homogeneous 3x3 matrix.
pub type Projective2<N> = Transform<N, TProjective, 2>;
/// A 2D affine transformation. Stored as a homogeneous 3x3 matrix.
pub type Affine2<N> = Transform<N, TAffine, 2>;

/// A 3D general transformation that may not be inversible. Stored as a homogeneous 4x4 matrix.
pub type Transform3<N> = Transform<N, TGeneral, 3>;
/// An invertible 3D general transformation. Stored as a homogeneous 4x4 matrix.
pub type Projective3<N> = Transform<N, TProjective, 3>;
/// A 3D affine transformation. Stored as a homogeneous 4x4 matrix.
pub type Affine3<N> = Transform<N, TAffine, 3>;
