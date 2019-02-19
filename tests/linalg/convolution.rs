#[allow(unused_imports)] // remove after fixing unit test
use na::linalg::{convolve_full,convolve_valid,convolve_same};
#[allow(unused_imports)]
use na::{Vector2,Vector3,Vector4,Vector5,DVector};

//
// Should mimic calculations in Python's scipy library
// >>>from scipy.signal import convolve
//

// >>> convolve([1,2,3,4],[1,2],"same")
// array([ 1,  4,  7, 10])
#[test]
fn convolve_same_check(){
    let vec_s = Vector4::new(1.0,2.0,3.0,4.0);
    let ker_s = Vector2::new(1.0,2.0);

    let actual_s =  Vector4::from_vec(vec![1.0,4.0,7.0,10.0]);

    let expected_s = convolve_same(vec_s,ker_s);
    let expected_s_r = convolve_same(ker_s,vec_s);

    assert!(relative_eq!(actual_s, expected_s, epsilon = 1.0e-7));
    assert!(relative_eq!(actual_s, expected_s_r, epsilon = 1.0e-7));
       
    let vec_d = DVector::from_vec(4,vec![1.0,2.0,3.0,4.0]);
    let ker_d = DVector::from_vec(2,vec![1.0,2.0]);

    let actual_d =  DVector::from_vec(4,vec![1.0,4.0,7.0,10.0]);

    let expected_d = convolve_same(vec_d.clone(),ker_d.clone());
    let expected_d_r = convolve_same(ker_d,vec_d);

    assert!(relative_eq!(actual_d, expected_d, epsilon = 1.0e-7));
    assert!(relative_eq!(actual_d, expected_d_r, epsilon = 1.0e-7));
}

// >>> convolve([1,2,3,4],[1,2],"full")
// array([ 1, 4,  7, 10, 8])
#[test]
fn convolve_full_check(){
    let vec_s = Vector4::new(1.0,2.0,3.0,4.0);
    let ker_s = Vector2::new(1.0,2.0);

    let actual_s =  Vector5::new(1.0,4.0,7.0,10.0,8.0);

    let expected_s = convolve_full(vec_s,ker_s);
    let expected_s_r = convolve_full(ker_s,vec_s);

    assert!(relative_eq!(actual_s, expected_s, epsilon = 1.0e-7));
    assert!(relative_eq!(actual_s, expected_s_r, epsilon = 1.0e-7));
    
    let vec_d = DVector::from_vec(4,vec![1.0,2.0,3.0,4.0]);
    let ker_d = DVector::from_vec(2,vec![1.0,2.0]);

    let actual_d =  DVector::from_vec(5,vec![1.0,4.0,7.0,10.0,8.0]);

    let expected_d = convolve_full(vec_d.clone(),ker_d.clone());
    let expected_d_r = convolve_full(ker_d,vec_d);

    assert!(relative_eq!(actual_d, expected_d, epsilon = 1.0e-7));
    assert!(relative_eq!(actual_d, expected_d_r, epsilon = 1.0e-7));
}

// >>> convolve([1,2,3,4],[1,2],"valid")
// array([ 4,  7, 10])
// #[test]
// fn convolve_valid_check(){
//     let vec = Vector4::new(1.0,2.0,3.0,4.0);
//     let ker = Vector2::new(1.0,2.0);

//     let actual =  Vector3::from_vec(vec![4.0,7.0,10.0]);

//     let expected1 = convolve_valid(vec, ker);
//     let expected2 = convolve_valid(ker, vec);


//     assert!(relative_eq!(actual, expected1, epsilon = 1.0e-7));
//     assert!(relative_eq!(actual, expected2, epsilon = 1.0e-7));

// }