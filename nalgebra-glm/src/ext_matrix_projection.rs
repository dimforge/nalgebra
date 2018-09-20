use na::{Real, U2, U3, U4};

use aliases::{Mat, Vec};


pub fn pickMatrix<N: Real>(center: &Vec<N, U2>, delta: &Vec<N, U2>, viewport: &Vec<N, U4>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn project<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn projectNO<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn projectZO<N: Real>(obj: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn unProject<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn unProjectNO<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn unProjectZO<N: Real>(win: &Vec<N, U3>, model: &Mat<N, U4, U4>, proj: &Mat<N, U4, U4>, viewport: Vec<N, U4>) -> Vec<N, U3> {
    unimplemented!()
}