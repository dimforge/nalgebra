use na::{Real, Orthographic3, Perspective3};
use aliases::TMat4;

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

/// Creates a matrix for an orthographic parallel viewing volume, using the right handedness and OpenGL near and far clip planes definition.
pub fn ortho<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
    Orthographic3::new(left, right, bottom, top, znear, zfar).unwrap()
}

//pub fn ortho_lh<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_lh_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_lh_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_rh<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_rh_no<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_rh_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn ortho_zo<N: Real>(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> TMat4<N> {
//    unimplemented!()
//}

/// Creates a matrix for a symmetric perspective-view frustum based on the right handedness and OpenGL near and far clip planes definition.
pub fn perspective<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
    Perspective3::new(fovy, aspect, near, far).unwrap()
}

//pub fn perspective_fov<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_lh<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_lh_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_lh_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_rh<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_rh_no<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_rh_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_fov_zo<N: Real>(fov: N, width: N, height: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_lh<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_lh_no<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_lh_zo<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_no<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_rh<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_rh_no<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_rh_zo<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn perspective_zo<N: Real>(fovy: N, aspect: N, near: N, far: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn tweaked_infinite_perspective<N: Real>(fovy: N, aspect: N, near: N) -> TMat4<N> {
//    unimplemented!()
//}
//
//pub fn tweaked_infinite_perspective_ep<N: Real>(fovy: N, aspect: N, near: N, ep: N) -> TMat4<N> {
//    unimplemented!()
//}
