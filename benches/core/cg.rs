use na::{Matrix3, Matrix4, Orthographic3, Point2, Point3, Vector2, Vector3};
use rand::{Rng, SeedableRng};
use rand_isaac::IsaacRng;
use simba::simd::WideF32x4;

#[path = "../common/macros.rs"]
mod macros;

bench_binop_ref!(
    mat3_transform_vector2,
    Matrix3<f32>,
    Vector2<f32>,
    transform_vector
);
bench_binop_ref!(
    mat4_transform_vector3,
    Matrix4<f32>,
    Vector3<f32>,
    transform_vector
);
bench_binop_ref!(
    mat3_transform_point2,
    Matrix3<f32>,
    Point2<f32>,
    transform_point
);
bench_binop_ref!(
    mat4_transform_point3,
    Matrix4<f32>,
    Point3<f32>,
    transform_point
);

bench_binop_ref!(
    mat3_transform_vector2_x4wide,
    Matrix3<WideF32x4>,
    Vector2<WideF32x4>,
    simd_transform_vector
);
bench_binop_ref!(
    mat4_transform_vector3_x4wide,
    Matrix4<WideF32x4>,
    Vector3<WideF32x4>,
    simd_transform_vector
);
bench_binop_ref!(
    mat3_transform_point2_x4wide,
    Matrix3<WideF32x4>,
    Point2<WideF32x4>,
    simd_transform_point
);
bench_binop_ref!(
    mat4_transform_point3_x4wide,
    Matrix4<WideF32x4>,
    Point3<WideF32x4>,
    simd_transform_point
);

fn mat4_transform_vector3_no_division(bench: &mut criterion::Criterion) {
    let mut rng = IsaacRng::seed_from_u64(0);
    let orthographic = Orthographic3::from_fov(
        rng.random_range(0.5..2.0),
        rng.random_range(30.0..90.0),
        rng.random_range(0.5..1.5),
        rng.random_range(2.0..1000.0),
    )
    .to_homogeneous();
    let vector = rng.random();
    bench.bench_function("mat4_transform_vector3_no_division", move |bh| {
        bh.iter(|| orthographic.transform_vector(&vector))
    });
}

criterion_group!(
    cg,
    mat3_transform_vector2,
    mat4_transform_vector3,
    mat3_transform_point2,
    mat4_transform_point3,
    mat3_transform_vector2_x4wide,
    mat4_transform_vector3_x4wide,
    mat3_transform_point2_x4wide,
    mat4_transform_point3_x4wide,
    mat4_transform_vector3_no_division,
);
