extern crate nalgebra as na;
#[allow(unused_imports)]
use na::{Vector,Dim,Real,Vector4,Vector3,Vector2,U1,Matrix,DVector,Dynamic,VecStorage};
use na::storage::{Storage};
use std::cmp;



// evum CovvolveMode{
//     Full,
//     Valid,
//     Same
// }



#[allow(non_snake_case)]
fn Convolve1D<R: Real, D: Dim, E: Dim, S: Storage<R, D>, Q: Storage<R, E>>(
    Vector : Vector<R,D,S>,
    Kernel : Vector<R,E,Q>
    ) -> Matrix<R, Dynamic, U1, VecStorage<R, Dynamic, U1>>
    {
        //
        // Vector is the vector, Kervel is the kervel
        // C is the returv vector
        //
        if Kernel.len() > Vector.len(){
            return Convolve1D(Kernel, Vector);
        }

        let V = Vector.len();
        let K = Kernel.len();
        let L = V + K - 1;
        let v = V as i8;
        let k = K as i8;
        let l = L as i8;
        let mut C = DVector::<R>::zeros(L);

        for i in 0..l{
            let u_i = cmp::max(0, i - k);
            let u_f = cmp::min(i, v - 1);
            if u_i == u_f{
                C[i as usize] += Vector[u_i as usize] * Kernel[(i - u_i) as usize];
            }
            else{
                for u in u_i..(u_f+1){
                    if i - u < k{
                        C[i as usize] += Vector[u as usize] * Kernel[(i - u ) as usize];
                    }
                }
            }
        }
        C
    }


fn main() {
    let v1 = Vector2::new(3.0,3.0);
    let v2 = Vector4::new(1.0,2.0,5.0,9.0);
    let x = Convolve1D(v1,v2);
    println!("{:?}",x)
}