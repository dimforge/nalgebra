extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use na::Perspective3;
use na::Orthographic3;
use glm::Mat4;
use glm::Vec4;

#[test]
pub fn orthographic_glm_nalgebra_same()
{
    let na_mat : Mat4 = Orthographic3::new(-100.0f32,100.0f32, -50.0f32, 50.0f32, 0.1f32, 100.0f32).into_inner();
    let gl_mat : Mat4 = glm::ortho(-100.0f32,100.0f32, -50.0f32, 50.0f32, 0.1f32, 100.0f32);

    assert_eq!(na_mat, gl_mat);
}

#[test]
pub fn perspective_glm_nalgebra_same()
{
    let na_mat : Mat4 = Perspective3::new(16.0f32/9.0f32, 3.14f32/2.0f32, 0.1f32, 100.0f32).into_inner();
    let gl_mat : Mat4 = glm::perspective(16.0f32/9.0f32, 3.14f32/2.0f32, 0.1f32, 100.0f32);

    assert_eq!(na_mat, gl_mat);
}

#[test]
pub fn orthographic_glm_nalgebra_project_same()
{
    let point = Vec4::new(1.0,0.0,-20.0,1.0);

    let na_mat : Mat4 = Orthographic3::new(-100.0f32,100.0f32, -50.0f32, 50.0f32, 0.1f32, 100.0f32).into_inner();
    let gl_mat : Mat4 = glm::ortho(-100.0f32,100.0f32, -50.0f32, 50.0f32, 0.1f32, 100.0f32);

    let na_pt = na_mat * point;
    let gl_pt = gl_mat * point;

    assert_eq!(na_mat, gl_mat);
    assert_eq!(na_pt, gl_pt);
}

#[test]
pub fn perspective_glm_nalgebra_project_same()
{
    let point = Vec4::new(1.0,0.0,-20.0,1.0);

    let na_mat : Mat4 = Perspective3::new(16.0f32/9.0f32, 3.14f32/2.0f32, 0.1f32, 100.0f32).into_inner();
    let gl_mat : Mat4 = glm::perspective(16.0f32/9.0f32, 3.14f32/2.0f32, 0.1f32, 100.0f32);

    let na_pt = na_mat * point;
    let gl_pt = gl_mat * point;

    assert_eq!(na_mat, gl_mat);
    assert_eq!(na_pt, gl_pt);
}
