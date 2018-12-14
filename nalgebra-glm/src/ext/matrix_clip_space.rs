use aliases::TMat4;
use na::{Real};

//pub fn frustum<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}

//pub fn frustum_lh<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_lr_no<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_lh_zo<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_no<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_no<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_rh_zo<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn frustum_zo<N: Real>(left: N, right: N, bottom: N, top: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}

//pub fn infinite_perspective<N: Real>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn infinite_perspective_lh<N: Real>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn infinite_perspective_rh<N: Real>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn infinite_ortho<N: Real>(left: N, right: N, bottom: N, top: N) -> TMat4<N> {
//    unimplemented!()
//}

/// Creates a matrix for a orthographic-view frustum with a handedness and depth range defined by
/// the defaults configured for the library at build time
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
/// # Compile Options
///
/// There are 3 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
/// 3. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand orthographic matrix.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a orthographic matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn ortho<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        ortho_zo(left, right, bottom, top, znear, zfar)
    } else if cfg!(feature="negone_to_one_clip_default") {
        ortho_no(left, right, bottom, top, znear, zfar)
    } else {
        unimplemented!()
    }
}

/// Creates a left hand matrix for a orthographic-view frustum with a depth range defined by the
/// default configured for the library at build time
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
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a orthographic matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn ortho_lh<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        ortho_zo(left, right, bottom, top, znear, zfar)
    } else if cfg!(feature="negone_to_one_clip_default") {
        ortho_no(left, right, bottom, top, znear, zfar)
    } else {
        unimplemented!()
    }
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
/// # Compile Options
///
/// There are 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn ortho_lh_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let zero   : N = ::convert(0.0);
    let one    : N = ::convert(1.0);
    let two    : N = ::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(one ,zero,zero,zero,
                                             zero,one ,zero,zero,
                                             zero,zero,one ,zero,
                                             zero,zero,zero,one );

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (two/(top-bottom)) * ::convert(-1.0)
        } else {
            (two/(top-bottom))
        };

    mat[(0,0)] = two / (right - left);
    mat[(0,3)] = -(right + left) / (right - left);
    mat[(1,1)] = m11;
    mat[(1,3)] = -(top + bottom) / (top - bottom);
    mat[(2,2)] = two / (zfar - znear);
    mat[(2,3)] = -(zfar + znear) / (zfar - znear);

    mat
}

/// Creates a left hand matrix for a orthographic-view frustum with a depth range of 0 to 1
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
/// # Compile Options
///
/// There are 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn ortho_lh_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let zero   : N = ::convert(0.0);
    let one    : N = ::convert(1.0);
    let two    : N = ::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(one ,zero,zero,zero,
                                             zero,one ,zero,zero,
                                             zero,zero,one ,zero,
                                             zero,zero,zero,one );

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (two/(top-bottom)) * ::convert(-1.0)
        } else {
            (two/(top-bottom))
        };

    mat[(0,0)] = two / (right - left);
    mat[(0,3)] = - (right + left) / (right - left);
    mat[(1,1)] = m11;
    mat[(1,3)] = - (top + bottom) / (top - bottom);
    mat[(2,2)] = one / (zfar - znear);
    mat[(2,3)] = - znear / (zfar  - znear);

    mat
}

/// Creates a matrix for a orthographic-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of -1 to 1
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
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand orthographic matrix.
///
pub fn ortho_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        ortho_lh_no(left, right, bottom, top, znear, zfar)
    } else if cfg!(feature="right_hand_default") {
        ortho_rh_no(left, right, bottom, top, znear, zfar)
    } else {
        unimplemented!()
    }
}

/// Creates a right hand matrix for a orthographic-view frustum with a depth range defined by the
/// default configured for the library at build time
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
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a orthographic matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn ortho_rh<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        ortho_rh_zo(left, right, bottom, top, znear, zfar)
    } else if cfg!(feature="negone_to_one_clip_default") {
        ortho_rh_no(left, right, bottom, top, znear, zfar)
    } else {
        unimplemented!()
    }
}

/// Creates a right hand matrix for a orthographic-view frustum with a depth range of -1 to 1
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
/// # Compile Options
///
/// There are 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn ortho_rh_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let zero   : N = ::convert(0.0);
    let one    : N = ::convert(1.0);
    let two    : N = ::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(one ,zero,zero,zero,
                                             zero,one ,zero,zero,
                                             zero,zero,one ,zero,
                                             zero,zero,zero,one );

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (two/(top-bottom)) * ::convert(-1.0)
        } else {
            (two/(top-bottom))
        };

    mat[(0,0)] = two / (right - left);
    mat[(0,3)] = - (right + left) / (right - left);
    mat[(1,1)] = m11;
    mat[(1,3)] = - (top + bottom) / (top - bottom);
    mat[(2,2)] = - two / (zfar - znear);
    mat[(2,3)] = - (zfar + znear) / (zfar  - znear);

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
/// # Compile Options
///
/// There are 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn ortho_rh_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    let zero   : N = ::convert(0.0);
    let one    : N = ::convert(1.0);
    let two    : N = ::convert(2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(one ,zero,zero,zero,
                                             zero,one ,zero,zero,
                                             zero,zero,one ,zero,
                                             zero,zero,zero,one );

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (two/(top-bottom)) * ::convert(-1.0)
        } else {
            (two/(top-bottom))
        };

    mat[(0,0)] = two / (right - left);
    mat[(0,3)] = - (right + left) / (right - left);
    mat[(1,1)] = m11;
    mat[(1,3)] = - (top + bottom) / (top - bottom);
    mat[(2,2)] = - one / (zfar - znear);
    mat[(2,3)] = - znear / (zfar  - znear);

    mat
}

/// Creates a matrix for a orthographic-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of 0 to 1
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
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand orthographic matrix.
///
pub fn ortho_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        ortho_lh_zo(left, right, bottom, top, znear, zfar)
    } else if cfg!(feature="right_hand_default") {
        ortho_rh_zo(left, right, bottom, top, znear, zfar)
    } else {
        unimplemented!()
    }
}

/// Creates a matrix for a perspective-view frustum with a handedness and depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 3 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
/// 3. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective_fov<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_fov_zo(fov, width, height, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_fov_no(fov, width, height, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a left handed matrix for a perspective-view frustum with a depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective_fov_lh<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_fov_lh_zo(fov, width, height, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_fov_lh_no(fov, width, height, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a left hand matrix for a perspective-view frustum with a -1 to 1 depth range
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_fov_lh_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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

    let zero   : N = ::convert( 0.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let rad = fov;
    let h = (rad * ::convert(0.5)).cos() / (rad * ::convert(0.5)).sin();
    let w = h * height / width;

    let m11 =
        if cfg!(feature="projection_y_flip") {
            h * ::convert(-1.0)
        } else {
            h
        };

    mat[(0,0)] = w;
    mat[(1,1)] = m11;
    mat[(2,2)] = (far + near) / (far - near);
    mat[(2,3)] = - (far * near * ::convert(2.0)) / (far - near);
    mat[(3,2)] = ::convert(1.0);

    mat
}

/// Creates a left hand matrix for a perspective-view frustum with a 0 to 1 depth range
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_fov_lh_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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

    let zero   : N = ::convert( 0.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let rad = fov;
    let h = (rad * ::convert(0.5)).cos() / (rad * ::convert(0.5)).sin();
    let w = h * height / width;

    let m11 =
        if cfg!(feature="projection_y_flip") {
            h * ::convert(-1.0)
        } else {
            h
        };

    mat[(0,0)] = w;
    mat[(1,1)] = m11;
    mat[(2,2)] = far / (far - near);
    mat[(2,3)] = -(far * near) / (far - near);
    mat[(3,2)] = ::convert(1.0);

    mat
}

/// Creates a matrix for a perspective-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of -1 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
pub fn perspective_fov_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        perspective_fov_lh_no(fov, width, height, near, far)
    } else if cfg!(feature="right_hand_default") {
        perspective_fov_rh_no(fov, width, height, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a right handed matrix for a perspective-view frustum with a depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective_fov_rh<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_fov_rh_zo(fov, width, height, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_fov_rh_no(fov, width, height, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a right hand matrix for a perspective-view frustum with a -1 to 1 depth range
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_fov_rh_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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

    let negone : N = ::convert(-1.0);
    let zero   : N = ::convert( 0.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let rad = fov;
    let h = (rad * ::convert(0.5)).cos() / (rad * ::convert(0.5)).sin();
    let w = h * height / width;

    let m11 =
        if cfg!(feature="projection_y_flip") {
            h * ::convert(-1.0)
        } else {
            h
        };

    mat[(0,0)] = w;
    mat[(1,1)] = m11;
    mat[(2,2)] = - (far + near) / (far - near);
    mat[(2,3)] = - (far * near * ::convert(2.0)) / (far - near);
    mat[(3,2)] =   negone;

    mat
}

/// Creates a right hand matrix for a perspective-view frustum with a 0 to 1 depth range
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_fov_rh_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
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

    let negone : N = ::convert(-1.0);
    let zero   : N = ::convert( 0.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let rad = fov;
    let h = (rad * ::convert(0.5)).cos() / (rad * ::convert(0.5)).sin();
    let w = h * height / width;

    let m11 =
        if cfg!(feature="projection_y_flip") {
            h * ::convert(-1.0)
        } else {
            h
        };

    mat[(0,0)] = w;
    mat[(1,1)] = m11;
    mat[(2,2)] = far / (near - far);
    mat[(2,3)] = -(far * near) / (far - near);
    mat[(3,2)] = negone;

    mat
}

/// Creates a matrix for a perspective-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of 0 to 1
///
/// # Parameters
///
/// * `fov` - Field of view, in radians
/// * `width` - Width of the viewport
/// * `height` - Height of the viewport
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
pub fn perspective_fov_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        perspective_fov_lh_zo(fov, width, height, near, far)
    } else if cfg!(feature="right_hand_default") {
        perspective_fov_rh_zo(fov, width, height, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a matrix for a perspective-view frustum with a handedness and depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 3 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
/// 3. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    // TODO: Breaking change: the arguments can be reversed back to proper glm conventions
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_zo(aspect, fovy, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_no(aspect, fovy, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a left handed matrix for a perspective-view frustum with a depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective_lh<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_lh_zo(fovy, aspect, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_lh_no(fovy, aspect, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a left hand matrix for a perspective-view frustum with a -1 to 1 depth range
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_lh_no<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let zero   : N = ::convert( 0.0);
    let one    : N = ::convert( 1.0);
    let two    : N = ::convert( 2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let tan_half_fovy = (fovy / two).tan();

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (one / tan_half_fovy) * ::convert(-1.0)
        } else {
            (one / tan_half_fovy)
        };

    mat[(0,0)] = one / (aspect * tan_half_fovy);
    mat[(1,1)] = m11;
    mat[(2,2)] = (far + near) / (far - near);
    mat[(2,3)] = -(two * far * near) / (far - near);
    mat[(3,2)] = one;

    mat
}

/// Creates a left hand matrix for a perspective-view frustum with a 0 to 1 depth range
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_lh_zo<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let zero   : N = ::convert( 0.0);
    let one    : N = ::convert( 1.0);
    let two    : N = ::convert( 2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let tan_half_fovy = (fovy / two).tan();

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (one / tan_half_fovy) * ::convert(-1.0)
        } else {
            (one / tan_half_fovy)
        };

    mat[(0,0)] = one / (aspect * tan_half_fovy);
    mat[(1,1)] = m11;
    mat[(2,2)] = far / (far - near);
    mat[(2,3)] = -(far * near) / (far - near);
    mat[(3,2)] = one;

    mat
}

/// Creates a matrix for a perspective-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of -1 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
pub fn perspective_no<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        perspective_lh_no(aspect, fovy, near, far)
    } else if cfg!(feature="right_hand_default") {
        perspective_rh_no(aspect, fovy, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a right handed matrix for a perspective-view frustum with a depth range defined by the
/// defaults configured for the library at build time
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. zero_to_one_clip_default/negone_to_one_clip_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### zero_to_one_clip_default/negone_to_one_clip_default
/// Depending on which option is set the function will return a perspective matrix meant for a
/// 0 to 1 depth clip space or a -1 to 1 depth clip space.
///
pub fn perspective_rh<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="zero_to_one_clip_default") {
        perspective_rh_zo(aspect, fovy, near, far)
    } else if cfg!(feature="negone_to_one_clip_default") {
        perspective_rh_no(aspect, fovy, near, far)
    } else {
        unimplemented!()
    }
}

/// Creates a right hand matrix for a perspective-view frustum with a -1 to 1 depth range
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_rh_no<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let negone : N = ::convert(-1.0);
    let zero   : N = ::convert( 0.0);
    let one    : N = ::convert( 1.0);
    let two    : N = ::convert( 2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let tan_half_fovy = (fovy / two).tan();

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (one / tan_half_fovy) * ::convert(-1.0)
        } else {
            (one / tan_half_fovy)
        };

    mat[(0,0)] = one / (aspect * tan_half_fovy);
    mat[(1,1)] = m11;
    mat[(2,2)] = - (far + near) / (far - near);
    mat[(2,3)] = -(two * far * near) / (far - near);
    mat[(3,2)] = negone;

    mat
}

/// Creates a right hand matrix for a perspective-view frustum with a 0 to 1 depth range
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. projection_y_flip
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
pub fn perspective_rh_zo<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    assert!(
        !relative_eq!(far - near, N::zero()),
        "The near-plane and far-plane must not be superimposed."
    );
    assert!(
        !relative_eq!(aspect, N::zero()),
        "The apsect ratio must not be zero."
    );

    let negone : N = ::convert(-1.0);
    let zero   : N = ::convert( 0.0);
    let one    : N = ::convert( 1.0);
    let two    : N = ::convert( 2.0);
    let mut mat : TMat4<N> = TMat4::<N>::new(zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero, 
                                             zero,zero,zero,zero);

    let tan_half_fovy = (fovy / two).tan();

    let m11 =
        if cfg!(feature="projection_y_flip") {
            (one / tan_half_fovy) * ::convert(-1.0)
        } else {
            (one / tan_half_fovy)
        };

    mat[(0,0)] = one / (aspect * tan_half_fovy);
    mat[(1,1)] = m11;
    mat[(2,2)] = far / (near - far);
    mat[(2,3)] = -(far * near) / (far - near);
    mat[(3,2)] = negone;

    mat
}

/// Creates a matrix for a perspective-view frustum with a handedness defined by the defaults
/// configured for the library at build time and a depth range of 0 to 1
///
/// # Parameters
///
/// * `fovy` - Field of view, in radians
/// * `aspect` - Ratio of viewport width to height (width/height)
/// * `near` - Distance from the viewer to the near clipping plane
/// * `far` - Distance from the viewer to the far clipping plane
///
/// # Compile Options
///
/// There are 2 compile options that change the behaviour of the function:
/// 1. projection_y_flip
/// 2. left_hand_default/right_hand_default
///
/// ##### projection_y_flip
/// When enabled will perform a `matrix[(1,1)] *= 1` implicitly. Primary use case is for Vulkan
/// where the viewport coordinate origin is the top left, unlike OpenGL which is the bottom left.
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand perspective matrix.
///
pub fn perspective_zo<N: Real>(aspect: N, fovy: N, near: N, far: N) -> TMat4<N> {
    if cfg!(feature="left_hand_default") {
        perspective_lh_zo(aspect, fovy, near, far)
    } else if cfg!(feature="right_hand_default") {
        perspective_rh_zo(aspect, fovy, near, far)
    } else {
        unimplemented!()
    }
}

//pub fn tweaked_infinite_perspective<N: Real>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn tweaked_infinite_perspective_ep<N: Real>(fovy: N, aspect: N, near: N, ep: N) -> TMat4<N> {
//    unimplemented!()
//}
