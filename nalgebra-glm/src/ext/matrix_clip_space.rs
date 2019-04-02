use crate::aliases::TMat4;
use na::{RealField};

//pub fn frustum<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}

//pub fn frustum_lh<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_lr_no<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_lh_zo<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_no<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_no<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_zo<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_zo<N: RealField>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}

//pub fn infinite_perspective<N: RealField>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn infinite_perspective_lh<N: RealField>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn infinite_ortho<N: RealField>(left: N, right: N, bottom: N, top: N) -> TMat4<N> {
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
pub fn ortho<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
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
pub fn ortho_lh<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
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
pub fn ortho_lh_no<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let two    : N =  crate::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = -(right + left) / (right - left);
    mat[(1, 1)] = two / (top-bottom);
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
pub fn ortho_lh_zo<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let one    : N = N::one();
    let two    : N = crate::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = - (right + left) / (right - left);
    mat[(1, 1)] = two / (top - bottom);
    mat[(1, 3)] = - (top + bottom) / (top - bottom);
    mat[(2, 2)] = one / (zfar - znear);
    mat[(2, 3)] = - znear / (zfar  - znear);

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
pub fn ortho_no<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
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
pub fn ortho_rh<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
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
pub fn ortho_rh_no<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let two    : N =  crate::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = - (right + left) / (right - left);
    mat[(1, 1)] = two/(top-bottom);
    mat[(1, 3)] = - (top + bottom) / (top - bottom);
    mat[(2, 2)] = - two / (zfar - znear);
    mat[(2, 3)] = - (zfar + znear) / (zfar  - znear);

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
pub fn ortho_rh_zo<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let one    : N = N::one();
    let two    : N =  crate::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::identity();

    mat[(0, 0)] = two / (right - left);
    mat[(0, 3)] = - (right + left) / (right - left);
    mat[(1, 1)] = two/(top-bottom);
    mat[(1, 3)] = - (top + bottom) / (top - bottom);
    mat[(2, 2)] = - one / (zfar - znear);
    mat[(2, 3)] = - znear / (zfar  - znear);

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
pub fn ortho_zo<N: RealField>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
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
pub fn perspective_fov<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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
pub fn perspective_fov_lh<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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
pub fn perspective_fov_lh_no<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    assert!(
        width > N::zero(),
        "The width must be greater than zero"
    );
    assert!(
        height > N::zero(),
        "The height must be greater than zero."
    );
    assert!(
        fov > N::zero(),
        "The fov must be greater than zero"
    );

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = (far + near) / (far - near);
    mat[(2, 3)] = - (far * near * crate::convert(2.0)) / (far - near);
    mat[(3, 2)] = N::one();

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
pub fn perspective_fov_lh_zo<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    assert!(
        width > N::zero(),
        "The width must be greater than zero"
    );
    assert!(
        height > N::zero(),
        "The height must be greater than zero."
    );
    assert!(
        fov > N::zero(),
        "The fov must be greater than zero"
    );

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = far / (far - near);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = N::one();

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
pub fn perspective_fov_no<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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
pub fn perspective_fov_rh<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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
pub fn perspective_fov_rh_no<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    assert!(
        width > N::zero(),
        "The width must be greater than zero"
    );
    assert!(
        height > N::zero(),
        "The height must be greater than zero."
    );
    assert!(
        fov > N::zero(),
        "The fov must be greater than zero"
    );

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = - (far + near) / (far - near);
    mat[(2, 3)] = - (far * near * crate::convert(2.0)) / (far - near);
    mat[(3, 2)] = -N::one();

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
pub fn perspective_fov_rh_zo<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    assert!(
        width > N::zero(),
        "The width must be greater than zero"
    );
    assert!(
        height > N::zero(),
        "The height must be greater than zero."
    );
    assert!(
        fov > N::zero(),
        "The fov must be greater than zero"
    );

    let mut mat = TMat4::zeros();

    let rad = fov;
    let h = (rad * crate::convert(0.5)).cos() / (rad * crate::convert(0.5)).sin();
    let w = h * height / width;

    mat[(0, 0)] = w;
    mat[(1, 1)] = h;
    mat[(2, 2)] = far / (near - far);
    mat[(2, 3)] = -(far * near) / (far - near);
    mat[(3, 2)] = -N::one();

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
pub fn perspective_fov_zo<N: RealField>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    perspective_fov_rh_zo(fov, width, height, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
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
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    perspective_lh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a left hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh_no<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let one = N::one();
    let two: N = crate::convert( 2.0);
    let mut mat : TMat4<N> = TMat4::zeros();

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
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_lh_zo<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let one = N::one();
    let two: N = crate::convert( 2.0);
    let mut mat: TMat4<N> = TMat4::zeros();

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
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_no<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    perspective_rh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    perspective_rh_no(aspect, fovy, near, far)
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of -1 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh_no<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let negone = -N::one();
    let one =  N::one();
    let two: N =   crate::convert( 2.0);
    let mut mat = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = - (far + near) / (far - near);
    mat[(2, 3)] = -(two * far * near) / (far - near);
    mat[(3, 2)] = negone;

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a depth range of 0 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_rh_zo<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let negone = -N::one();
    let one =  N::one();
    let two =   crate::convert( 2.0);
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
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn perspective_zo<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    perspective_rh_zo(aspect, fovy, near, far)
}

/// Build infinite right-handed perspective projection matrix with [-1,1] depth range.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn infinite_perspective_rh_no<N: RealField>(aspect: N, fovy: N, near: N) -> TMat4<N> {
    let f = N::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 2)] = -N::one();
    mat[(2, 3)] = -near * na::convert(2.0);
    mat[(3, 2)] = -N::one();

    mat
}

/// Build infinite right-handed perspective projection matrix with [0,1] depth range.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
///
// https://discourse.nphysics.org/t/reversed-z-and-infinite-zfar-in-projections/341/2
pub fn infinite_perspective_rh_zo<N: RealField>(aspect: N, fovy: N, near: N) -> TMat4<N> {
    let f = N::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 2)] = -N::one();
    mat[(2, 3)] = -near;
    mat[(3, 2)] = -N::one();

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a reversed depth range of -1 to 1.
///
/// Note that when using reversed perspective, it is best to use a depth buffer based on floating points.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn reversed_perspective_rh_no<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    let one =  N::one();
    let two: N = crate::convert(2.0);
    let mut mat = TMat4::zeros();

    let tan_half_fovy = (fovy / two).tan();

    mat[(0, 0)] = one / (aspect * tan_half_fovy);
    mat[(1, 1)] = one / tan_half_fovy;
    mat[(2, 2)] = (far + near) / (far - near) - one;
    mat[(2, 3)] = (two * far * near) / (far - near);
    mat[(3, 2)] = -one;

    mat
}

/// Creates a matrix for a right hand perspective-view frustum with a reversed depth range of 0 to 1.
///
/// Note that when using reversed perspective, it is best to use a depth buffer based on floating points.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn reversed_perspective_rh_zo<N: RealField>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    let one = N::one();
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

/// Build an infinite perspective projection matrix with a reversed [-1, 1] depth range.
///
/// Note that when using reversed perspective, it is best to use a depth buffer based on floating points.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
pub fn reversed_infinite_perspective_rh_no<N: RealField>(aspect: N, fovy: N, near: N) -> TMat4<N> {
    let f = N::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 3)] = near * crate::convert(2.0);
    mat[(3, 2)] = -N::one();

    mat
}

/// Build an infinite perspective projection matrix with a reversed [0, 1] depth range.
///
/// Note that when using reversed perspective, it is best to use a depth buffer based on floating points.
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane.
///
/// # Important note
/// The `aspect` and `fovy` argument are interchanged compared to the original GLM API.
// Credit: https://discourse.nphysics.org/t/reversed-z-and-infinite-zfar-in-projections/341/2
pub fn reversed_infinite_perspective_rh_zo<N: RealField>(aspect: N, fovy: N, near: N) -> TMat4<N> {
    let f = N::one() / (fovy * na::convert(0.5)).tan();
    let mut mat = TMat4::zeros();

    mat[(0, 0)] = f / aspect;
    mat[(1, 1)] = f;
    mat[(2, 3)] = near;
    mat[(3, 2)] = -N::one();

    mat
}

//pub fn tweaked_infinite_perspective<N: RealField>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn tweaked_infinite_perspective_ep<N: RealField>(fovy: N, aspect: N, near: N, ep: N) -> TMat4<N> {
//    unimplemented!()
//}