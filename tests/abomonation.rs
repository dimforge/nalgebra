extern crate rand;
extern crate nalgebra;
extern crate abomonation;

use rand::random;
use abomonation::{Abomonation, encode, decode};
use nalgebra::Matrix3x4;

#[test]
fn abomonate_matrix3x4() {
    assert_encode_and_decode(&random::<Matrix3x4<f32>>());
}

fn assert_encode_and_decode<T: Abomonation + PartialEq>(data: &T) {
    let mut bytes = Vec::new();
    unsafe { encode(data, &mut bytes); }

    if let Some((result, rest)) = unsafe { decode::<T>(&mut bytes) } {
        assert!(result == data);
        assert!(rest.len() == 0, "binary data was not decoded completely");
    }
}
