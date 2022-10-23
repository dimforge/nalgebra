extern crate nalgebra as na;
extern crate nalgebra_lapack;
#[macro_use]
extern crate approx; // for assert_relative_eq

use na::Matrix3;
use nalgebra_lapack::Eigen;
use num_complex::Complex;

//Matrix taken from https://textbooks.math.gatech.edu/ila/1553/complex-eigenvalues.html
fn main() {
    let m = Matrix3::<f64>::new(4.0/5.0, -3.0/5.0, 0.0, 3.0/5.0, 4.0/5.0, 0.0, 1.0, 2.0, 2.0);
    let eigen = Eigen::new(m,true,true).expect("Eigen Creation Failed!");
    let (some_eigenvalues, some_left_vec, some_right_vec) = eigen.get_complex_elements();
    let eigenvalues = some_eigenvalues.expect("Eigenvalues Failed");
    let _left_eigenvectors = some_left_vec.expect("Left Eigenvectors Failed");
    let eigenvectors = some_right_vec.expect("Right Eigenvectors Failed");

    assert_relative_eq!(eigenvalues[0].re, Complex::<f64>::new(4.0/5.0,3.0/5.0).re);
    assert_relative_eq!(eigenvalues[0].im, Complex::<f64>::new(4.0/5.0,3.0/5.0).im);
    assert_relative_eq!(eigenvalues[1].re, Complex::<f64>::new(4.0/5.0,-3.0/5.0).re);
    assert_relative_eq!(eigenvalues[1].im, Complex::<f64>::new(4.0/5.0,-3.0/5.0).im);


    assert_relative_eq!(eigenvectors[0][0].re, -12.0/32.7871926215100059134410999);
    assert_relative_eq!(eigenvectors[0][0].im, -9.0/32.7871926215100059134410999);
    assert_relative_eq!(eigenvectors[0][1].re, -9.0/32.7871926215100059134410999);
    assert_relative_eq!(eigenvectors[0][1].im, 12.0/32.7871926215100059134410999);
    assert_relative_eq!(eigenvectors[0][2].re, 25.0/32.7871926215100059134410999);
    assert_relative_eq!(eigenvectors[0][2].im, 0.0);

    
}