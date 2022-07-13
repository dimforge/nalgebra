#![cfg(feature = "rkyv-serialize")]

use na::{
    DMatrix, Isometry2, Isometry3, IsometryMatrix2, IsometryMatrix3, Matrix2x3, Matrix3x4, Point2,
    Point3, Quaternion, Rotation2, Rotation3, Similarity2, Similarity3, SimilarityMatrix2,
    SimilarityMatrix3, Translation2, Translation3, Unit, Vector2,
};
use rand;
use rkyv::{Archive, Serialize, Deserialize};

macro_rules! test_rkyv(
    ($($test: ident, $ty: ident);* $(;)*) => {$(
        #[test]
        fn $test() {
            let value: $ty<f32> = rand::random();
			let bytes = rkyv::to_bytes::<_, 256>(&value).unwrap();

			let archived = rkyv::check_archived_root::<$ty<f32>>(&bytes[..]).unwrap();
			assert_eq!(archived, &value);

			assert_eq!(format!("{:?}", value), format!("{:?}", archived));
        }
    )*}
);

test_rkyv!(
    rkyv_matrix3x4,          Matrix3x4;
    // rkyv_point3,             Point3;
   /*  rkyv_translation3,       Translation3;
    rkyv_rotation3,          Rotation3;
    rkyv_isometry3,          Isometry3;
    rkyv_isometry_matrix3,   IsometryMatrix3;
    rkyv_similarity3,        Similarity3;
    rkyv_similarity_matrix3, SimilarityMatrix3;
    rkyv_quaternion,         Quaternion;
    rkyv_point2,             Point2;
    rkyv_translation2,       Translation2;
    rkyv_rotation2,          Rotation2;
    rkyv_isometry2,          Isometry2;
    rkyv_isometry_matrix2,   IsometryMatrix2;
    rkyv_similarity2,        Similarity2;
    rkyv_similarity_matrix2, SimilarityMatrix2; */
);