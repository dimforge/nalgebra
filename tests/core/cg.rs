use na::{Matrix3, Matrix4, Orthographic3, Perspective3, Point2, Point3, Vector2, Vector3};

#[cfg(feature = "wide-tests")]
use na::{Rotation3, SimdValue, Translation3};
#[cfg(feature = "wide-tests")]
use simba::simd::WideF32x4;

/// See Example 3.4 of "Graphics and Visualization: Principles & Algorithms"
/// by Theoharis, Papaioannou, Platis, Patrikalakis.
#[test]
fn test_scaling_wrt_point_1() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 1.0);
    let c = Point2::new(5.0, 2.0);

    let scaling = Vector2::new(2.0, 2.0);
    let scale_about = Matrix3::new_nonuniform_scaling_wrt_point(&scaling, &c);

    let expected_a = Point2::new(-5.0, -2.0);
    let expected_b = Point2::new(-3.0, 0.0);
    let result_a = scale_about.transform_point(&a);
    let result_b = scale_about.transform_point(&b);
    let result_c = scale_about.transform_point(&c);

    assert!(expected_a == result_a);
    assert!(expected_b == result_b);
    assert!(c == result_c);
}

/// Based on the same example as the test above.
#[test]
fn test_scaling_wrt_point_2() {
    let a = Point3::new(0.0, 0.0, 1.0);
    let b = Point3::new(1.0, 1.0, 1.0);
    let c = Point3::new(5.0, 2.0, 1.0);

    let scaling = Vector3::new(2.0, 2.0, 1.0);
    let scale_about = Matrix4::new_nonuniform_scaling_wrt_point(&scaling, &c);

    let expected_a = Point3::new(-5.0, -2.0, 1.0);
    let expected_b = Point3::new(-3.0, 0.0, 1.0);

    let result_a = scale_about.transform_point(&a);
    let result_b = scale_about.transform_point(&b);
    let result_c = scale_about.transform_point(&c);

    assert!(expected_a == result_a);
    assert!(expected_b == result_b);
    assert!(c == result_c);
}

/// Based on https://github.com/emlowry/AiE/blob/50bae4068edb686cf8ffacdf6fab8e7cb22e7eb1/Year%201%20Classwork/MathTest/Matrix4x4TestGroup.cpp#L145
#[test]
fn test_scaling_wrt_point_3() {
    let about = Point3::new(2.0, 1.0, -2.0);
    let scale = Vector3::new(2.0, 0.5, -1.0);
    let pt = Point3::new(1.0, 2.0, 3.0);
    let scale_about = Matrix4::new_nonuniform_scaling_wrt_point(&scale, &about);

    let expected = Point3::new(0.0, 1.5, -7.0);
    let result = scale_about.transform_point(&pt);

    assert!(result == expected);
}

#[test]
fn test_perspective_transform_vector() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    let perspective = Perspective3::new(2.0, 45.0, 1.0, 1000.0);

    let transformed = perspective.as_matrix().transform_vector(&vector);

    let multiplied = perspective.as_matrix() * vector.to_homogeneous();
    let multiplied = multiplied / multiplied.w;

    assert_relative_eq!(transformed, perspective.project_vector(&vector));
    assert_relative_eq!(transformed.push(1.0), multiplied);
}

#[test]
fn test_perspective_simd_transform_vector() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    let perspective = Perspective3::new(2.0, 45.0, 1.0, 1000.0);

    assert_relative_eq!(
        perspective.as_matrix().simd_transform_vector(&vector),
        perspective.as_matrix().transform_vector(&vector)
    );
}

#[test]
fn test_perspective_transform_point3() {
    let point = Point3::new(1.0, 2.0, 3.0);
    let perspective = Perspective3::new(2.0, 45.0, 1.0, 1000.0);

    let transformed = perspective.as_matrix().transform_point(&point);

    let multiplied = perspective.as_matrix() * point.to_homogeneous();
    let multiplied = multiplied / multiplied.w;

    assert_relative_eq!(transformed, perspective.project_point(&point));
    assert_relative_eq!(transformed.coords.push(1.0), multiplied);
}

#[test]
fn test_perspective_simd_transform_point3() {
    let point = Point3::new(1.0, 2.0, 3.0);
    let perspective = Perspective3::new(2.0, 45.0, 1.0, 1000.0);

    assert_relative_eq!(
        perspective.as_matrix().simd_transform_point(&point),
        perspective.as_matrix().transform_point(&point)
    );
}

#[test]
fn test_orthographic_transform_vector() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    let orthographic = Orthographic3::from_fov(2.0, 45.0, 1.0, 1000.0);

    let transformed = orthographic.as_matrix().transform_vector(&vector);

    let multiplied = orthographic.as_matrix() * vector.to_homogeneous();

    assert_relative_eq!(transformed, orthographic.project_vector(&vector));
    assert_relative_eq!(transformed.push(0.0), multiplied);
}

#[test]
fn test_orthographic_simd_transform_vector() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    let orthographic = Orthographic3::from_fov(2.0, 45.0, 1.0, 1000.0);

    assert_relative_eq!(
        orthographic.as_matrix().simd_transform_vector(&vector),
        orthographic.as_matrix().transform_vector(&vector)
    );
}

#[test]
fn test_orthographic_transform_point3() {
    let point = Point3::new(1.0, 2.0, 3.0);
    let orthographic = Orthographic3::from_fov(2.0, 45.0, 1.0, 1000.0);

    let transformed = orthographic.as_matrix().transform_point(&point);

    let multiplied = orthographic.as_matrix() * point.to_homogeneous();
    let multiplied = multiplied / multiplied.w;

    assert_relative_eq!(transformed, orthographic.project_point(&point));
    assert_relative_eq!(transformed.coords.push(1.0), multiplied);
}

#[test]
fn test_orthographic_simd_transform_point3() {
    let point = Point3::new(1.0, 2.0, 3.0);
    let orthographic = Orthographic3::from_fov(2.0, 45.0, 1.0, 1000.0);

    assert_relative_eq!(
        orthographic.as_matrix().simd_transform_point(&point),
        orthographic.as_matrix().transform_point(&point)
    );
}

#[cfg(feature = "wide-tests")]
#[test]
fn test_transform_vector_x4wide() {
    let v1 = Vector3::new(0.0, 1.0, 2.0);
    let v2 = Vector3::new(3.0, 4.0, 5.0);
    let v3 = Vector3::new(6.0, 7.0, 8.0);
    let v4 = Vector3::new(9.0, 10.0, 11.0);

    let wide_v = Vector3::new(
        WideF32x4::from_arr([v1.x, v2.x, v3.x, v4.x]),
        WideF32x4::from_arr([v1.y, v2.y, v3.y, v4.y]),
        WideF32x4::from_arr([v1.z, v2.z, v3.z, v4.z]),
    );

    let m1 = Perspective3::new(2.0, 45.0, 1.0, 1000.0).to_homogeneous();
    let m2 = Orthographic3::from_fov(2.0, 45.0, 1.0, 1000.0).to_homogeneous();
    let m3 = Rotation3::from_axis_angle(&Vector3::y_axis(), 2.5).to_homogeneous();
    let m4 = Translation3::new(1.0, 2.0, 3.0).to_homogeneous();

    let wide_m = Matrix4::new(
        WideF32x4::from_arr([m1.m11, m2.m11, m3.m11, m4.m11]),
        WideF32x4::from_arr([m1.m12, m2.m12, m3.m12, m4.m12]),
        WideF32x4::from_arr([m1.m13, m2.m13, m3.m13, m4.m13]),
        WideF32x4::from_arr([m1.m14, m2.m14, m3.m14, m4.m14]),
        WideF32x4::from_arr([m1.m21, m2.m21, m3.m21, m4.m21]),
        WideF32x4::from_arr([m1.m22, m2.m22, m3.m22, m4.m22]),
        WideF32x4::from_arr([m1.m23, m2.m23, m3.m23, m4.m23]),
        WideF32x4::from_arr([m1.m24, m2.m24, m3.m24, m4.m24]),
        WideF32x4::from_arr([m1.m31, m2.m31, m3.m31, m4.m31]),
        WideF32x4::from_arr([m1.m32, m2.m32, m3.m32, m4.m32]),
        WideF32x4::from_arr([m1.m33, m2.m33, m3.m33, m4.m33]),
        WideF32x4::from_arr([m1.m34, m2.m34, m3.m34, m4.m34]),
        WideF32x4::from_arr([m1.m41, m2.m41, m3.m41, m4.m41]),
        WideF32x4::from_arr([m1.m42, m2.m42, m3.m42, m4.m42]),
        WideF32x4::from_arr([m1.m43, m2.m43, m3.m43, m4.m43]),
        WideF32x4::from_arr([m1.m44, m2.m44, m3.m44, m4.m44]),
    );

    let wide_v = wide_m.simd_transform_vector(&wide_v);

    assert_eq!(wide_v.extract(0), m1.transform_vector(&v1));
    assert_eq!(wide_v.extract(1), m2.transform_vector(&v2));
    assert_eq!(wide_v.extract(2), m3.transform_vector(&v3));
    assert_eq!(wide_v.extract(3), m4.transform_vector(&v4));
}
