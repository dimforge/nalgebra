use rand::random;
use abomonation::{Abomonation, encode, decode};
use na::{
    DMatrix, Matrix3x4, Point3, Translation3, Rotation3, Isometry3, Quaternion,
    IsometryMatrix3, Similarity3, SimilarityMatrix3
};

#[test]
fn abomonate_dmatrix() {
    assert_encode_and_decode(DMatrix::<f32>::new_random(3, 5));
}

macro_rules! test_abomonation(
    ($($test: ident, $ty: ty);* $(;)*) => {$(
        #[test]
        fn $test() {
            assert_encode_and_decode(random::<$ty>());
        }
    )*}
);

test_abomonation! {
    abomonate_matrix3x4, Matrix3x4<f32>;
    abomonate_point3, Point3<f32>;
    abomonate_translation3, Translation3<f64>;
    abomonate_rotation3, Rotation3<f64>;
    abomonate_isometry3, Isometry3<f32>;
    abomonate_isometry_matrix3, IsometryMatrix3<f64>;
    abomonate_similarity3, Similarity3<f32>;
    abomonate_similarity_matrix3, SimilarityMatrix3<f32>;
    abomonate_quaternion, Quaternion<f32>;
}

fn assert_encode_and_decode<T: Abomonation + PartialEq + Clone>(original_data: T) {
    use std::mem::drop;

    // Hold on to a clone for later comparison
    let data = original_data.clone();

    // Encode
    let mut bytes = Vec::new();
    unsafe { encode(&original_data, &mut bytes); }

    // Drop the original, so that dangling pointers are revealed by the test
    drop(original_data);

    if let Some((result, rest)) = unsafe { decode::<T>(&mut bytes) } {
        assert!(result == &data);
        assert!(rest.len() == 0, "binary data was not decoded completely");
    }
}
