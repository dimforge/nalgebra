use core::MatrixArray;
use core::dimension::{U1, U2, U3, DimNameSum};

use geometry::{TransformBase, TGeneral, TProjective, TAffine};

pub type Transform<N, D>  = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TGeneral>;
pub type Projective<N, D> = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TProjective>;
pub type Affine<N, D>     = TransformBase<N, D, MatrixArray<N, DimNameSum<D, U1>, DimNameSum<D, U1>>, TAffine>;

pub type Transform2<N>  = Transform<N, U2>;
pub type Projective2<N> = Projective<N, U2>;
pub type Affine2<N>     = Affine<N, U2>;

pub type Transform3<N>  = Transform<N, U3>;
pub type Projective3<N> = Projective<N, U3>;
pub type Affine3<N>     = Affine<N, U3>;
