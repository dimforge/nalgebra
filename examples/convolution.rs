extern crate nalgebra as na;
use na::storage::Storage;
use na::{
    convert, zero, DMatrix, DVector, Dim, Dynamic, Matrix, Matrix2x3, Real, VecStorage, Vector,
    Vector2, Vector3, Vector4, Vector5, U1,
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
        let u_i = if i > ker { i - ker } else { 0 };
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

    let mut conv = DVector::<R>::zeros(vec);

    for i in 0..vec {
        for j in 0..ker {
            let val = if i + j < 1 || i + j >= vec + 1 {
                zero::<R>()
            } else {
                vector[i + j - 1]
            };
            conv[i] += val * kernel[ker - j - 1];
        }
    }
    conv
}

fn convolve<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector: Vector<R, D, S>,
    kernel: Vector<R, E, Q>,
    mode: Option<ConvolveMode>,
) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>> {
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
    let v1 = Vector4::new(1.0, 2.0, 1.0, 0.0);
    let v2 = Vector4::new(1.0, 2.0, 5.0, 9.0);
    let x = convolve(v1, v2, Some(ConvolveMode::Same));
    println!("{:?}", x);
}
