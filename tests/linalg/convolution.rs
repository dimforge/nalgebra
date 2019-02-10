use na::linalg::{convolve_full,convolve_valid,convolve_same};
use na::{Vector2,Vector4,DVector};

//
// Should mimic calculations in Python's scipy library
// >>>from scipy.signal import convolve
//

// >>> convolve([1,2,3,4],[1,2],"same")
// array([ 1,  4,  7, 10])
#[test]
fn convolve_same_check(){
    let vec = Vector4::new(1.0,2.0,3.0,4.0);
    let ker = Vector2::new(1.0,2.0);

    let actual =  DVector::from_vec(4, vec![1.0,4.0,7.0,10.0]);

    let expected = convolve_same(vec,ker);

    assert!(relative_eq!(actual, expected, epsilon = 1.0e-7));
}

// >>> convolve([1,2,3,4],[1,2],"valid")
// array([ 1, 4,  7, 10, 8])
#[test]
fn convolve_full_check(){
    let vec = Vector4::new(1.0,2.0,3.0,4.0);
    let ker = Vector2::new(1.0,2.0);

    let actual =  DVector::from_vec(5, vec![1.0,4.0,7.0,10.0,8.0]);

    let expected = convolve_full(vec,ker);

    assert!(relative_eq!(actual, expected, epsilon = 1.0e-7));
}

// >>> convolve([1,2,3,4],[1,2],"valid")
// array([ 4,  7, 10])
#[test]
fn convolve_valid_check(){
    let vec = Vector4::new(1.0,2.0,3.0,4.0);
    let ker = Vector2::new(1.0,2.0);

    let actual =  DVector::from_vec(3, vec![4.0,7.0,10.0]);

    let expected = convolve_valid(vec,ker);

    assert!(relative_eq!(actual, expected, epsilon = 1.0e-7));
}