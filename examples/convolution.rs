extern crate nalgebra;
use nalgebra::{Vector2,Vector3,Vector4,Vector5,convolve_full,convolve_same,convolve_valid};

fn main(){
    let vec = Vector4::new(1.0,2.0,3.0,4.0);
    let ker = Vector3::new(1.0,2.0,2.1);

    let actual =  Vector5::from_vec(vec![1.0,4.0,7.0,10.0,8.0]);

    let expected = convolve_full(vec,ker);
    let expected2 = convolve_same(vec,ker);
    // let expected3 = convolve_valid(vec,ker);
    println!("{}", actual);
    println!("{}", expected);
    println!("{}", expected2);
    // println!("{}", expected3);
}