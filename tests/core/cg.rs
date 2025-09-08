use na::{Matrix3, Matrix4, Orthographic3, Perspective3, Point2, Point3, Vector2, Vector3};

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
