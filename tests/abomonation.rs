extern crate rand;
extern crate nalgebra;
extern crate abomonation;

use rand::random;
use abomonation::{Abomonation, encode, decode};
use nalgebra::{DMatrix, Matrix3x4, Point3, Translation3};

#[test]
fn abomonate_matrix3x4() {
    assert_encode_and_decode(&random::<Matrix3x4<f32>>());
}

#[test]
fn abomonate_point3() {
    assert_encode_and_decode(&random::<Point3<f64>>());
}

#[test]
fn abomonate_dmatrix() {
    assert_encode_and_decode(&DMatrix::<f32>::new_random(3, 5));
}

#[test]
fn abomonate_translation3() {
    assert_encode_and_decode(&random::<Translation3<f32>>());
}

fn assert_encode_and_decode<T: Abomonation + PartialEq>(data: &T) {
    let mut bytes = Vec::new();
    unsafe { encode(data, &mut bytes); }

    if let Some((result, rest)) = unsafe { decode::<T>(&mut bytes) } {
        assert!(result == data);
        assert!(rest.len() == 0, "binary data was not decoded completely");
    }
}
