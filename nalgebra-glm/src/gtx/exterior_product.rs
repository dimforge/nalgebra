use crate::aliases::TVec2;
use crate::traits::Number;

/// The 2D perpendicular product between two vectors.
pub fn cross2d<N: Number>(v: &TVec2<N>, u: &TVec2<N>) -> N {
    v.perp(u)
}
