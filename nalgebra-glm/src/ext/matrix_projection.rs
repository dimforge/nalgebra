use na::{self, Real, U2, U3, U4, Vector3, Vector4, Matrix4};

use aliases::{Mat, Vec};

/// Define a picking region.
///
/// # Parameters
///     * `center`: Specify the center of a picking region in window coordinates.
//      * `delta`: Specify the width and height, respectively, of the picking region in window coordinates.
//      * `viewport`: Rendering viewport
pub fn pick_matrix<N: Real>(center: &Vec<N, U2>, delta: &Vec<N, U2>, viewport: &Vec<N, U4>) -> Mat<N, U4, U4> {
    let shift = Vector3::new(
        (viewport.z - (center.x - viewport.x) * na::convert(2.0)) / delta.x,
        (viewport.w - (center.y - viewport.y) * na::convert(2.0)) / delta.y,
        N::zero()
    );

    let result = Matrix4::new_translation(&shift);
    result.prepend_nonuniform_scaling(&Vector3::new(viewport.z / delta.x, viewport.w / delta.y, N::one()))
}

/// Map the specified object coordinates `(obj.x, obj.y, obj.z)` into window coordinates using OpenGL near and far clip planes definition.
///
/// # Parameters
///     * `obj`: Specify the object coordinates.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn project<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    project_no(obj, model, proj, viewport)
}

/// Map the specified object coordinates (obj.x, obj.y, obj.z) into window coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively. (OpenGL clip volume definition)
///
/// # Parameters
///     * `obj`: Specify the object coordinates.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn project_no<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    let proj = project_zo(obj, model, proj, viewport);
    Vector3::new(proj.x, proj.y, proj.z * na::convert(0.5) + na::convert(0.5))
}

/// Map the specified object coordinates (obj.x, obj.y, obj.z) into window coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of 0 and +1 respectively. (Direct3D clip volume definition)
///
/// # Parameters
///     * `obj`: Specify the object coordinates.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn project_zo<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    let normalized = proj * model * Vector4::new(obj.x, obj.y, obj.z, N::one());
    let scale = N::one() / normalized.w;

    Vector3::new(
        viewport.x + (viewport.z * (normalized.x * scale + N::one()) * na::convert(0.5)),
        viewport.y + (viewport.w * (normalized.y * scale + N::one()) * na::convert(0.5)),
        normalized.z * scale,
    )
}

/// Map the specified window coordinates (win.x, win.y, win.z) into object coordinates using OpengGL near and far clip planes definition.
///
/// # Parameters
///     * `obj`: Specify the window coordinates to be mapped.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn unproject<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unproject_no(win, model, proj, viewport)
}

/// Map the specified window coordinates (win.x, win.y, win.z) into object coordinates.
///
/// The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively. (OpenGL clip volume definition)
///
/// # Parameters
///     * `obj`: Specify the window coordinates to be mapped.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn unproject_no<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    let _2: N = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or(Matrix4::zeros());
    let pt = Vector4::new(
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
/// # Parameters
///     * `obj`: Specify the window coordinates to be mapped.
///     * `model`: Specifies the current modelview matrix.
///     * `proj`: Specifies the current projection matrix.
///     * `viewport`: Specifies the current viewport.
pub fn unproject_zo<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    let _2: N = na::convert(2.0);
    let transform = (proj * model).try_inverse().unwrap_or(Matrix4::zeros());
    let pt = Vector4::new(
        _2 * (win.x - viewport.x) / viewport.z - N::one(),
        _2 * (win.y - viewport.y) / viewport.w - N::one(),
        win.z,
        N::one(),
    );

    let result = transform * pt;
    result.fixed_rows::<U3>(0) / result.w
}