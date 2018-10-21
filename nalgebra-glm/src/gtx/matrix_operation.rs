use aliases::{
    TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3, TVec2, TVec3, TVec4,
};
use traits::Number;

/// Builds a 2x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal2x2<N: Number>(v: &TVec2<N>) -> TMat2<N> {
    TMat2::from_diagonal(v)
}

/// Builds a 2x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal2x3<N: Number>(v: &TVec2<N>) -> TMat2x3<N> {
    TMat2x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 2x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal2x4<N: Number>(v: &TVec2<N>) -> TMat2x4<N> {
    TMat2x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal3x2<N: Number>(v: &TVec2<N>) -> TMat3x2<N> {
    TMat3x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal3x3<N: Number>(v: &TVec3<N>) -> TMat3<N> {
    TMat3::from_diagonal(v)
}

/// Builds a 3x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal3x4<N: Number>(v: &TVec3<N>) -> TMat3x4<N> {
    TMat3x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal4x2<N: Number>(v: &TVec2<N>) -> TMat4x2<N> {
    TMat4x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x4`](fn.diagonal4x4.html)
pub fn diagonal4x3<N: Number>(v: &TVec3<N>) -> TMat4x3<N> {
    TMat4x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2`](fn.diagonal2x2.html)
/// * [`diagonal2x3`](fn.diagonal2x3.html)
/// * [`diagonal2x4`](fn.diagonal2x4.html)
/// * [`diagonal3x2`](fn.diagonal3x2.html)
/// * [`diagonal3x3`](fn.diagonal3x3.html)
/// * [`diagonal3x4`](fn.diagonal3x4.html)
/// * [`diagonal4x2`](fn.diagonal4x2.html)
/// * [`diagonal4x3`](fn.diagonal4x3.html)
pub fn diagonal4x4<N: Number>(v: &TVec4<N>) -> TMat4<N> {
    TMat4::from_diagonal(v)
}
