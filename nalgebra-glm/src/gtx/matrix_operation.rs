use traits::Number;
use aliases::{TVec2, TVec3, TVec4, TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3};

/// Builds a 2x2 diagonal matrix.
pub fn diagonal2x2<N: Number>(v: &TVec2<N>) -> TMat2<N> {
    TMat2::from_diagonal(v)
}

/// Builds a 2x3 diagonal matrix.
pub fn diagonal2x3<N: Number>(v: &TVec2<N>) -> TMat2x3<N> {
    TMat2x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 2x4 diagonal matrix.
pub fn diagonal2x4<N: Number>(v: &TVec2<N>) -> TMat2x4<N> {
    TMat2x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x2 diagonal matrix.
pub fn diagonal3x2<N: Number>(v: &TVec2<N>) -> TMat3x2<N> {
    TMat3x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x3 diagonal matrix.
pub fn diagonal3x3<N: Number>(v: &TVec3<N>) -> TMat3<N> {
    TMat3::from_diagonal(v)
}

/// Builds a 3x4 diagonal matrix.
pub fn diagonal3x4<N: Number>(v: &TVec3<N>) -> TMat3x4<N> {
    TMat3x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x2 diagonal matrix.
pub fn diagonal4x2<N: Number>(v: &TVec2<N>) -> TMat4x2<N> {
    TMat4x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x3 diagonal matrix.
pub fn diagonal4x3<N: Number>(v: &TVec3<N>) -> TMat4x3<N> {
    TMat4x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x4 diagonal matrix.
pub fn diagonal4x4<N: Number>(v: &TVec4<N>) -> TMat4<N> {
    TMat4::from_diagonal(v)
}
