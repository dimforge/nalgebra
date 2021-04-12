use crate::aliases::TMat4;
use na::RealField;

//pub fn frustum<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}

//pub fn frustum_lh<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_lr_no<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_lh_zo<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_no<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_rh<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_no<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_zo<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn frustum_zo<T: RealField>(left: T, right: T, bottom: T, top: T, near: T, far: T) -> TMat4<T> {
//    unimplemented!()
//}

//pub fn infinite_perspective<T: RealField>(fovy: T, aspect: T, near: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn infinite_perspective_lh<T: RealField>(fovy: T, aspect: T, near: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn infinite_ortho<T: RealField>(left: T, right: T, bottom: T, top: T) -> TMat4<T> {
//    unimplemented!()
//}

/// Creates a matrix for a right hand orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho<T: RealField>(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> TMat4<T> {
    ortho_rh_no(left, right, bottom, top, znear, zfar)
}

/// Creates a left hand matrix for a orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_lh<T: RealField>(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> TMat4<T> {
    ortho_lh_no(left, right, bottom, top, znear, zfar)
}

/// Creates a left hand matrix for a orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_lh_no<T: RealField>(
    left: T,
    right: T,
    bottom: T,
    top: T,
    znear: T,
    zfar: T,
) -> TMat4<T> {
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::<T>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = -(right + left) / (right - left);
    mat[(1, 1)] = two / (top - bottom);
    mat[(1, 3)] = -(top + bottom) / (top - bottom);
    mat[(2, 2)] = two / (zfar - znear);
    mat[(2, 3)] = -(zfar + znear) / (zfar - znear);

    mat
}

/// Creates a matrix for a left hand orthographic-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_lh_zo<T: RealField>(
    left: T,
    right: T,
    bottom: T,
    top: T,
    znear: T,
    zfar: T,
) -> TMat4<T> {
    let one: T = T::one();
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::<T>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = -(right + left) / (right - left);
    mat[(1, 1)] = two / (top - bottom);
    mat[(1, 3)] = -(top + bottom) / (top - bottom);
    mat[(2, 2)] = one / (zfar - znear);
    mat[(2, 3)] = -znear / (zfar - znear);

    mat
}

/// Creates a matrix for a right hand orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_no<T: RealField>(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> TMat4<T> {
    ortho_rh_no(left, right, bottom, top, znear, zfar)
}

/// Creates a matrix for a right hand orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_rh<T: RealField>(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> TMat4<T> {
    ortho_rh_no(left, right, bottom, top, znear, zfar)
}

/// Creates a matrix for a right hand orthographic-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_rh_no<T: RealField>(
    left: T,
    right: T,
    bottom: T,
    top: T,
    znear: T,
    zfar: T,
) -> TMat4<T> {
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::<T>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = -(right + left) / (right - left);
    mat[(1, 1)] = two / (top - bottom);
    mat[(1, 3)] = -(top + bottom) / (top - bottom);
    mat[(2, 2)] = -two / (zfar - znear);
    mat[(2, 3)] = -(zfar + znear) / (zfar - znear);

    mat
}

/// Creates a right hand matrix for a orthographic-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_rh_zo<T: RealField>(
    left: T,
    right: T,
    bottom: T,
    top: T,
    znear: T,
    zfar: T,
) -> TMat4<T> {
    let one: T = T::one();
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::<T>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = -(right + left) / (right - left);
    mat[(1, 1)] = two / (top - bottom);
    mat[(1, 3)] = -(top + bottom) / (top - bottom);
    mat[(2, 2)] = -one / (zfar - znear);
    mat[(2, 3)] = -znear / (zfar - znear);

    mat
}

/// Creates a right hand matrix for a orthographic-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `left` - Coordinate for left bound of matrix
/// * `right` - Coordinate for right bound of matrix
/// * `bottom` - Coordinate for bottom bound of matrix
/// * `top` - Coordinate for top bound of matrix
/// * `znear` - Distance from the viewer to the near clipping plane
/// * `zfar` - Distance from the viewer to the far clipping plane
///
pub fn ortho_zo<T: RealField>(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> TMat4<T> {
    ortho_rh_zo(left, right, bottom, top, znear, zfar)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov<T: RealField>(fov: T, width: T, height: T, near: T, far: T) -> TMat4<T> {
    perspective_fov_rh_no(fov, width, height, near, far)
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_lh<T: RealField>(fov: T, width: T, height: T, near: T, far: T) -> TMat4<T> {
    perspective_fov_lh_no(fov, width, height, near, far)
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_lh_no<T: RealField>(
    fov: T,
    width: T,
    height: T,
    near: T,
    far: T,
) -> TMat4<T> {
    assert!(width > T::zero(), "The width must be greater than zero");
    assert!(height > T::zero(), "The height must be greater than zero.");
    assert!(fov > T::zero(), "The fov must be greater than zero");

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = (far + near) / (far - near);
    mat[(2, 3)] = -(far * near * crate::convert(2.0)) / (far - near);
    mat[(3, 2)] = T::one();

    mat
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_lh_zo<T: RealField>(
    fov: T,
    width: T,
    height: T,
    near: T,
    far: T,
) -> TMat4<T> {
    assert!(width > T::zero(), "The width must be greater than zero");
    assert!(height > T::zero(), "The height must be greater than zero.");
    assert!(fov > T::zero(), "The fov must be greater than zero");

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = far / (far - near);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = T::one();

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_no<T: RealField>(fov: T, width: T, height: T, near: T, far: T) -> TMat4<T> {
    perspective_fov_rh_no(fov, width, height, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_rh<T: RealField>(fov: T, width: T, height: T, near: T, far: T) -> TMat4<T> {
    perspective_fov_rh_no(fov, width, height, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_rh_no<T: RealField>(
    fov: T,
    width: T,
    height: T,
    near: T,
    far: T,
) -> TMat4<T> {
    assert!(width > T::zero(), "The width must be greater than zero");
    assert!(height > T::zero(), "The height must be greater than zero.");
    assert!(fov > T::zero(), "The fov must be greater than zero");

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = -(far + near) / (far - near);
    mat[(2, 3)] = -(far * near * crate::convert(2.0)) / (far - near);
    mat[(3, 2)] = -T::one();

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_rh_zo<T: RealField>(
    fov: T,
    width: T,
    height: T,
    near: T,
    far: T,
) -> TMat4<T> {
    assert!(width > T::zero(), "The width must be greater than zero");
    assert!(height > T::zero(), "The height must be greater than zero.");
    assert!(fov > T::zero(), "The fov must be greater than zero");

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = far / (near - far);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = -T::one();

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
pub fn perspective_fov_zo<T: RealField>(fov: T, width: T, height: T, near: T, far: T) -> TMat4<T> {
    perspective_fov_rh_zo(fov, width, height, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    // TODO: Breaking change - revert back to proper glm conventions?
    //
    //       Prior to changes to support configuring the behaviour of this function it was simply
    //       a wrapper around Perspective3::new(). The argument order for that function is different
    //       than the glm convention, but reordering the arguments would've caused pointlessly
    //       un-optimal code to be generated so they were rearranged so the function would just call
    //       straight through.
    //
    //       Now this call to Perspective3::new() is no longer made so the functions can have their
    //       arguments reordered to the glm convention. Unfortunately this is a breaking change so
    //       can't be cleanly integrated into the existing library version without breaking other
    //       people's code. Reordering to glm isn't a huge deal but if it is done it will have to be
    //       in a major API breaking update.
    //
    perspective_rh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    perspective_lh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh_no<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    assert!(
        !relative_eq!(far - near, T::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, T::zero()),
        "The aspect ratio must not be zero."
    );

    let one = T::one();
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = (far + near) / (far - near);
    mat[(2, 3)] = -(two * far * near) / (far - near);
    mat[(3, 2)] = one;

    mat
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh_zo<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    assert!(
        !relative_eq!(far - near, T::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, T::zero()),
        "The aspect ratio must not be zero."
    );

    let one = T::one();
    let two: T = crate::convert(2.0);
    let mut mat: TMat4<T> = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = far / (far - near);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = one;

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_no<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    perspective_rh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    perspective_rh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh_no<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    assert!(
        !relative_eq!(far - near, T::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, T::zero()),
        "The aspect ratio must not be zero."
    );

    let negone = -T::one();
    let one = T::one();
    let two: T = crate::convert(2.0);
    let mut mat = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = -(far + near) / (far - near);
    mat[(2, 3)] = -(two * far * near) / (far - near);
    mat[(3, 2)] = negone;

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh_zo<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    assert!(
        !relative_eq!(far - near, T::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, T::zero()),
        "The aspect ratio must not be zero."
    );

    let negone = -T::one();
    let one = T::one();
    let two = crate::convert(2.0);
    let mut mat = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = far / (near - far);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = negone;

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_zo<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    perspective_rh_zo(aspect, fovy, near, far)
}

/// Build infinite right-handed perspective projection matrix with [-1,1] depth range.
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn infinite_perspective_rh_no<T: RealField>(aspect: T, fovy: T, near: T) -> TMat4<T> {
    let f = T::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 2)] = -T::one();
    mat[(2, 3)] = -near * na::convert(2.0);
    mat[(3, 2)] = -T::one();

    mat
}

/// Build infinite right-handed perspective projection matrix with [0,1] depth range.
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
///
// https://discourse.nphysics.org/t/reversed-z-and-infinite-zfar-in-projections/341/2
pub fn infinite_perspective_rh_zo<T: RealField>(aspect: T, fovy: T, near: T) -> TMat4<T> {
    let f = T::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 2)] = -T::one();
    mat[(2, 3)] = -near;
    mat[(3, 2)] = -T::one();

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a reversed depth range of 0 to 1.
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
// NOTE: The variants `_no` of reversed perspective are not useful.
pub fn reversed_perspective_rh_zo<T: RealField>(aspect: T, fovy: T, near: T, far: T) -> TMat4<T> {
    let one = T::one();
    let two = crate::convert(2.0);
    let mut mat = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = -far / (near - far) - one;
    mat[(2, 3)] = (far * near) / (far - near);
    mat[(3, 2)] = -one;

    mat
}

/// Build an infinite perspective projection matrix with a reversed [0, 1] depth range.
///
/// # Parameters
///
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `fovy` - Field of view, in radians
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
// Credit: https://discourse.nphysics.org/t/reversed-z-and-infinite-zfar-in-projections/341/2
// NOTE: The variants `_no` of reversed perspective are not useful.
pub fn reversed_infinite_perspective_rh_zo<T: RealField>(aspect: T, fovy: T, near: T) -> TMat4<T> {
    let f = T::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 3)] = near;
    mat[(3, 2)] = -T::one();

    mat
}

//pub fn tweaked_infinite_perspective<T: RealField>(fovy: T, aspect: T, near: T) -> TMat4<T> {
//    unimplemented!()
//}
//
//pub fn tweaked_infinite_perspective_ep<T: RealField>(fovy: T, aspect: T, near: T, ep: T) -> TMat4<T> {
//    unimplemented!()
//}
