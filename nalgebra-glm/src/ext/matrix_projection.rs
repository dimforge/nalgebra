use na::{self, Real, U3};

use aliases::{TMat4, TVec2, TVec3, TVec4};

/// Define a picking region.
///
/// # Parameters:
///
/// * `center` - Specify the center of a picking region in window coordinates.
/// * `delta` - Specify the width and height, respectively, of the picking region in window coordinates.
/// * `viewport` - Rendering viewport.
pub fn pick_matrix<N: Real>(center: &TVec2<N>, delta: &TVec2<N>, viewport: &TVec4<N>) -> TMat4<N> {
    let shift = TVec3::new(
        (viewport.z - (center.x - viewport.x) * na::convert(2.0)) / delta.x,
        (viewport.w - (center.y - viewport.y) * na::convert(2.0)) / delta.y,
        N::zero(),
    );

    let result = TMat4::new_translation(&shift);
    result.prepend_nonuniform_scaling(&TVec3::new(
        viewport.z / delta.x,
        viewport.w / delta.y,
        N::one(),
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
/// * [`project_no`](fn.project_no.html)
/// * [`project_zo`](fn.project_zo.html)
/// * [`unproject`](fn.unproject.html)
/// * [`unproject_no`](fn.unproject_no.html)
/// * [`unproject_zo`](fn.unproject_zo.html)
pub fn project<N: Real>(
    obj: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
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
/// * [`project`](fn.project.html)
/// * [`project_zo`](fn.project_zo.html)
/// * [`unproject`](fn.unproject.html)
/// * [`unproject_no`](fn.unproject_no.html)
/// * [`unproject_zo`](fn.unproject_zo.html)
pub fn project_no<N: Real>(
    obj: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
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
/// * [`project`](fn.project.html)
/// * [`project_no`](fn.project_no.html)
/// * [`unproject`](fn.unproject.html)
/// * [`unproject_no`](fn.unproject_no.html)
/// * [`unproject_zo`](fn.unproject_zo.html)
pub fn project_zo<N: Real>(
    obj: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
    let normalized = proj * model * TVec4::new(obj.x, obj.y, obj.z, N::one());
    let scale = N::one() / normalized.w;

    TVec3::new(
        viewport.x + (viewport.z * (normalized.x * scale + N::one()) * na::convert(0.5)),
        viewport.y + (viewport.w * (normalized.y * scale + N::one()) * na::convert(0.5)),
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
/// * [`project`](fn.project.html)
/// * [`project_no`](fn.project_no.html)
/// * [`project_zo`](fn.project_zo.html)
/// * [`unproject_no`](fn.unproject_no.html)
/// * [`unproject_zo`](fn.unproject_zo.html)
pub fn unproject<N: Real>(
    win: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
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
/// * [`project`](fn.project.html)
/// * [`project_no`](fn.project_no.html)
/// * [`project_zo`](fn.project_zo.html)
/// * [`unproject`](fn.unproject.html)
/// * [`unproject_zo`](fn.unproject_zo.html)
pub fn unproject_no<N: Real>(
    win: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
    let _2: N = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or_else(TMat4::zeros);
    let pt = TVec4::new(
        _2 * (win.x - viewport.x) / viewport.z - N::one(),
        _2 * (win.y - viewport.y) / viewport.w - N::one(),
        _2 * win.z - N::one(),
        N::one(),
    );

    let result = transform * pt;
    result.fixed_rows::<U3>(0) / result.w
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
/// * [`project`](fn.project.html)
/// * [`project_no`](fn.project_no.html)
/// * [`project_zo`](fn.project_zo.html)
/// * [`unproject`](fn.unproject.html)
/// * [`unproject_no`](fn.unproject_no.html)
pub fn unproject_zo<N: Real>(
    win: &TVec3<N>,
    model: &TMat4<N>,
    proj: &TMat4<N>,
    viewport: TVec4<N>,
) -> TVec3<N>
{
    let _2: N = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or_else(TMat4::zeros);
    let pt = TVec4::new(
        _2 * (win.x - viewport.x) / viewport.z - N::one(),
        _2 * (win.y - viewport.y) / viewport.w - N::one(),
        win.z,
        N::one(),
    );

    let result = transform * pt;
    result.fixed_rows::<U3>(0) / result.w
}
