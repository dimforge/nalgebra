extern crate nalgebra as na;
#[allow(unused_imports)]
use na::{Vector,Dim,Real,Vector4,Vector3,Vector2,U1,Matrix,DVector,Dynamic,VecStorage};
use na::storage::{Storage};
use std::cmp;



enum ConvolveMode{
    Full,
    Valid,
    Same
}

fn convolve_full<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector : Vector<R,D,S>,
    kernel : Vector<R,E,Q>
    ) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>>
    {

        let vec = vector.len();
        let ker = kernel.len();
        let len = vec + ker - 1;
        let v = vec as i8;
        let k = ker as i8;
        let l = len as i8;
        let mut conv = DVector::<R>::zeros(len);

        for i in 0..l{
            let u_i = cmp::max(0, i - k);
            let u_f = cmp::min(i, v - 1);

            if u_i == u_f{
                conv[i as usize] += vector[u_i as usize] * kernel[(i - u_i) as usize];
            }
            else{
                for u in u_i..(u_f+1){
                    if i - u < k{
                        conv[i as usize] += vector[u as usize] * kernel[(i - u ) as usize];
                    }
                }
            }
        }
        conv
    }

fn convolve_valid<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector : Vector<R,D,S>,
    kernel : Vector<R,E,Q>
    ) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>>
    {
        let vec = vector.len();
        let ker = kernel.len();
        let len = vec - ker + 1;
        let mut conv = DVector::<R>::zeros(len);

        for i in 0..len {
            for j in 0..ker {
                conv[i] += vector[i + j] * kernel[ker - j - 1];
            }
        }
        
        conv
    }

fn convolve_same<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector : Vector<R,D,S>,
    kernel : Vector<R,E,Q>
    ) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>>
    {

        let vec = vector.len();
        let ker = kernel.len();
        let len = vec + ker - 1;
        let v = vec as i8;
        let k = ker as i8;
        let l = len as i8;
        let mut conv = DVector::<R>::zeros(len);

        for i in 0..l {
            let u_i = cmp::max(0, i - k);
            let u_f = cmp::min(i, v - 1);

            if u_i == u_f {
                conv[i as usize] += vector[u_i as usize] * kernel[(i - u_i) as usize];
            }
            else{
                for u in u_i..(u_f+1){
                    if i - u < k{
                        conv[i as usize] += vector[u as usize] * kernel[(i - u ) as usize];
                    }
                }
            }
        }
        conv
    }

fn convolve<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    vector : Vector<R,D,S>,
    kernel : Vector<R,E,Q>,
    mode : Option<ConvolveMode>
    ) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>>
    {
        //
        // vector is the vector, Kervel is the kervel
        // C is the returv vector
        //
        if kernel.len() > vector.len(){
            return convolve(kernel, vector, mode);
        }

        match mode.unwrap_or(ConvolveMode::Full) {
            ConvolveMode::Full => return convolve_full(vector,kernel),
            ConvolveMode::Valid => return convolve_valid(vector,kernel),
            ConvolveMode::Same => return convolve_same(vector,kernel)
        }       
    }


fn main() {
    let v1 = Vector2::new(3.0,1.0);
    let v2 = Vector4::new(1.0,2.0,5.0,9.0);
    let x = convolve(v1,v2,Some(ConvolveMode::Valid));
    println!("{:?}",x)
}