extern crate nalgebra as na;
use na::storage::Storage;
#[allow(unused_imports)]
use na::{
    DMatrix, DVector, Dim, Dynamic, Matrix, Matrix2x3, Real, VecStorage, Vector, Vector2, Vector3,
    Vector4, U1,
};
use std::cmp;

enum ConvolveMode {
    Full,
    Valid,
    Same,
}

fn convolve_full<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();
    let newlen = vec + ker - 1;

    let mut conv = DVector::<R>::zeros(newlen);

    for i in 0..newlen {
        let u_i = if i > ker {i - ker} else {0};
        let u_f = cmp::min(i, vec - 1);

        if u_i == u_f {
            conv[i] += vector[u_i] * kernel[(i - u_i)];
        } else {
            for u in u_i..(u_f + 1) {
                if i - u < ker {
                    conv[i] += vector[u] * kernel[(i - u)];
                }
            }
        }
    }
    conv
}

fn convolve_valid<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();
    let newlen = vec - ker + 1;
    let mut conv = DVector::<R>::zeros(newlen);

    for i in 0..newlen {
        for j in 0..ker {
            conv[i] += vector[i + j] * kernel[ker - j - 1];
        }
    }
    conv
}

fn convolve_same<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    let vec = vector.len();
    let ker = kernel.len();
    let newlen = vec + ker - 1;

    let mut conv = DVector::<R>::zeros(newlen);

    for i in 0..newlen {
        // let u_i = cmp::max(0, i - k);
        // let u_f = cmp::min(i, v - 1);

        // if u_i == u_f {
        //     conv[i as usize] += vector[u_i as usize] * kernel[(i - u_i) as usize];
        // } else {
        //     for u in u_i..(u_f + 1) {
        //         if i - u < k {
        //             conv[i as usize] += vector[u as usize] * kernel[(i - u) as usize];
        //         }
        //     }
        // }
    }
    conv
}

fn convolve<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
    mode: Option<ConvolveMode>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
    //
    // vector is the vector, Kervel is the kervel
    // C is the returv vector
    //
    if kernel.len() > vector.len() {
        return convolve(kernel, vector, mode);
    }

    match mode.unwrap_or(ConvolveMode::Full) {
        ConvolveMode::Full => return convolve_full(vector, kernel),
        ConvolveMode::Valid => return convolve_valid(vector, kernel),
        ConvolveMode::Same => return convolve_same(vector, kernel),
    }
}

fn main() {
    let v1 = Vector2::new(3.0,1.0);
    let v2 = Vector4::new(1.0,2.0,5.0,9.0);
    let x = convolve(v1,v2,Some(ConvolveMode::Valid));
    println!("{:?}",x)

    // let m = Matrix2x3::from_anti_diagonal_element(5.0);
    // The two additional arguments represent the matrix dimensions.
    // let dm = DMatrix::from_anti_diagonal_element(2, 3, 5.0);
    let mut m = Matrix2x3::new(1.1, 1.2, 1.3,
                                2.1, 2.2, 2.3);

    // assert!(m.m11 == 0.0 && m.m12 == 0.0 && m.m13 == 5.0 &&
    //         m.m21 == 0.0 && m.m22 == 5.0 && m.m23 == 0.0);
    // assert!(dm[(0, 0)] == 0.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 5.0 &&
    //         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 5.0 && dm[(1, 2)] == 0.0);
    println!("m={:?}",m);
    for i in 0..std::cmp::min(m.nrows(),m.ncols()) {
        // for j in 0..3 {
            println!("m({:?},{:?})={:?}",i,3-i-1,m[(i,3-i-1)]);
unsafe { println!("m({:?},{:?})={:?}",i,3-i-1,*m.get_unchecked_mut((i, 3-i-1))) }
        // }
    }

    
}
