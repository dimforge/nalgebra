use crate::aliases::{TMat4, TVec2, TVec3, TVec4};
use crate::RealNumber;

/// Define a picking region.
///
/// # Parameters:
///
/// * `center` - Specify the center of a picking region in window coordinates.
/// * `delta` - Specify the width and height, respectively, of the picking region in window coordinates.
/// * `viewport` - Rendering viewport.
pub fn pick_matrix<T: RealNumber>(
    center: &TVec2<T>,
    delta: &TVec2<T>,
    viewport: &TVec4<T>,
) -> TMat4<T> {
    let shift = TVec3::new(
        (viewport.z - (center.x - viewport.x) * na::convert(2.0)) / delta.x,
        (viewport.w - (center.y - viewport.y) * na::convert(2.0)) / delta.y,
        T::zero(),
    );

    let result = TMat4::new_translation(&shift);
    result.prepend_nonuniform_scaling(&TVec3::new(
        viewport.z / delta.x,
        viewport.w / delta.y,
        T::one(),
    ))
}

/// Map the specified object coordinates `(obj.x, obj.y, obj.z)` into window coordinates with a
/// depth range of -1 to 1
///
/// # Parameters:
///
/// * `obj` - Specify the object coordinates.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project_no()`]
/// * [`project_zo()`]
/// * [`unproject()`]
/// * [`unproject_no()`]
/// * [`unproject_zo()`]
pub fn project<T: RealNumber>(
    obj: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    project_no(obj, model, proj, viewport)
}

/// Map the specified object coordinates (obj.x, obj.y, obj.z) into window coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively. (OpenGL clip volume definition)
///
/// # Parameters:
///
/// * `obj` - Specify the object coordinates.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project()`]
/// * [`project_zo()`]
/// * [`unproject()`]
/// * [`unproject_no()`]
/// * [`unproject_zo()`]
pub fn project_no<T: RealNumber>(
    obj: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    let proj = project_zo(obj, model, proj, viewport);
    TVec3::new(proj.x, proj.y, proj.z * na::convert(0.5) + na::convert(0.5))
}

/// Map the specified object coordinates (obj.x, obj.y, obj.z) into window coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of 0 and +1 respectively. (Direct3D clip volume definition)
///
/// # Parameters:
///
/// * `obj` - Specify the object coordinates.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project()`]
/// * [`project_no()`]
/// * [`unproject()`]
/// * [`unproject_no()`]
/// * [`unproject_zo()`]
pub fn project_zo<T: RealNumber>(
    obj: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    let normalized = proj * model * TVec4::new(obj.x, obj.y, obj.z, T::one());
    let scale = T::one() / normalized.w;

    TVec3::new(
        viewport.x + (viewport.z * (normalized.x * scale + T::one()) * na::convert(0.5)),
        viewport.y + (viewport.w * (normalized.y * scale + T::one()) * na::convert(0.5)),
        normalized.z * scale,
    )
}

/// Map the specified window coordinates (win.x, win.y, win.z) into object coordinates using a
/// depth range of -1 to 1
///
/// # Parameters:
///
/// * `obj` - Specify the window coordinates to be mapped.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project()`]
/// * [`project_no()`]
/// * [`project_zo()`]
/// * [`unproject_no()`]
/// * [`unproject_zo()`]
pub fn unproject<T: RealNumber>(
    win: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    unproject_no(win, model, proj, viewport)
}

/// Map the specified window coordinates (win.x, win.y, win.z) into object coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively. (OpenGL clip volume definition)
///
/// # Parameters:
///
/// * `obj` - Specify the window coordinates to be mapped.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project()`]
/// * [`project_no()`]
/// * [`project_zo()`]
/// * [`unproject()`]
/// * [`unproject_zo()`]
pub fn unproject_no<T: RealNumber>(
    win: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    let _2: T = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or_else(TMat4::zeros);
    let pt = TVec4::new(
        _2 * (win.x - viewport.x) / viewport.z - T::one(),
        _2 * (win.y - viewport.y) / viewport.w - T::one(),
        _2 * win.z - T::one(),
        T::one(),
    );

    let result = transform * pt;
    result.fixed_rows::<3>(0) / result.w
}

/// Map the specified window coordinates (win.x, win.y, win.z) into object coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of 0 and +1 respectively. (Direct3D clip volume definition)
///
/// # Parameters:
///
/// * `obj` - Specify the window coordinates to be mapped.
/// * `model` - Specifies the current modelview matrix.
/// * `proj` - Specifies the current projection matrix.
/// * `viewport` - Specifies the current viewport.
///
/// # See also:
///
/// * [`project()`]
/// * [`project_no()`]
/// * [`project_zo()`]
/// * [`unproject()`]
/// * [`unproject_no()`]
pub fn unproject_zo<T: RealNumber>(
    win: &TVec3<T>,
    model: &TMat4<T>,
    proj: &TMat4<T>,
    viewport: TVec4<T>,
) -> TVec3<T> {
    let _2: T = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or_else(TMat4::zeros);
    let pt = TVec4::new(
        _2 * (win.x - viewport.x) / viewport.z - T::one(),
        _2 * (win.y - viewport.y) / viewport.w - T::one(),
        win.z,
        T::one(),
    );

    let result = transform * pt;
    result.fixed_rows::<3>(0) / result.w
}
