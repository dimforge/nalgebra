use na::{
    Isometry3, IsometryMatrix2, IsometryMatrix3, Matrix3x4, Point2, Point3, Quaternion, Rotation2,
    Rotation3, Similarity3, SimilarityMatrix2, SimilarityMatrix3, Translation2, Translation3,
};
use rkyv::ser::{serializers::AllocSerializer, Serializer};

macro_rules! test_rkyv(
    ($($test: ident, $ty: ident);* $(;)*) => {$(
        #[test]
        fn $test() {
            let v: $ty<f32> = rand::random();
            // serialize
            let mut serializer = AllocSerializer::<0>::default();
            serializer.serialize_value(&v).unwrap();
            let serialized = serializer.into_serializer().into_inner();

            let deserialized: $ty<f32> = unsafe { rkyv::from_bytes_unchecked(&serialized).unwrap() };
            assert_eq!(v, deserialized);

            #[cfg(feature = "rkyv-safe-deser")]
            {
                let deserialized: $ty<f32> = rkyv::from_bytes(&serialized).unwrap();
                assert_eq!(v, deserialized);
            }
        }
    )*}
);

test_rkyv!(
    rkyv_matrix3x4,          Matrix3x4;
    rkyv_point3,             Point3;
    rkyv_translation3,       Translation3;
    rkyv_rotation3,          Rotation3;
    rkyv_isometry3,          Isometry3;
    rkyv_isometry_matrix3,   IsometryMatrix3;
    rkyv_similarity3,        Similarity3;
    rkyv_similarity_matrix3, SimilarityMatrix3;
    rkyv_quaternion,         Quaternion;
    rkyv_point2,             Point2;
    rkyv_translation2,       Translation2;
    rkyv_rotation2,          Rotation2;
    rkyv_isometry_matrix2,   IsometryMatrix2;
    rkyv_similarity_matrix2, SimilarityMatrix2;
);
