#![cfg(feature = "serde-serialize")]

use na::{
    DMatrix, Isometry2, Isometry3, IsometryMatrix2, IsometryMatrix3, Matrix3x4, Point2, Point3,
    Quaternion, Rotation2, Rotation3, Similarity2, Similarity3, SimilarityMatrix2,
    SimilarityMatrix3, Translation2, Translation3, Unit,
};
use rand;
use serde_json;

macro_rules! test_serde(
    ($($test: ident, $ty: ident);* $(;)*) => {$(
        #[test]
        fn $test() {
            let v: $ty<f32> = rand::random();
            let serialized = serde_json::to_string(&v).unwrap();
            let deserialized: $ty<f32> = serde_json::from_str(&serialized).unwrap();
            assert_eq!(v, deserialized);
        }
    )*}
);

#[test]
fn serde_dmatrix() {
    let v: DMatrix<f32> = DMatrix::new_random(3, 4);
    let serialized = serde_json::to_string(&v).unwrap();
    let deserialized: DMatrix<f32> = serde_json::from_str(&serialized).unwrap();
    assert_eq!(v, deserialized);
}

test_serde!(
    serde_matrix3x4,          Matrix3x4;
    serde_point3,             Point3;
    serde_translation3,       Translation3;
    serde_rotation3,          Rotation3;
    serde_isometry3,          Isometry3;
    serde_isometry_matrix3,   IsometryMatrix3;
    serde_similarity3,        Similarity3;
    serde_similarity_matrix3, SimilarityMatrix3;
    serde_quaternion,         Quaternion;
    serde_point2,             Point2;
    serde_translation2,       Translation2;
    serde_rotation2,          Rotation2;
    serde_isometry2,          Isometry2;
    serde_isometry_matrix2,   IsometryMatrix2;
    serde_similarity2,        Similarity2;
    serde_similarity_matrix2, SimilarityMatrix2;
);

#[test]
fn serde_flat() {
    // The actual storage is hidden behind three layers of wrapper types that shouldn't appear in serialized form.
    let v = Unit::new_normalize(Quaternion::new(0., 0., 0., 1.));
    let serialized = serde_json::to_string(&v).unwrap();
    assert_eq!(serialized, "[0.0,0.0,1.0,0.0]");
}
