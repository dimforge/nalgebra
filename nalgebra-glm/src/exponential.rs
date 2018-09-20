use na::{Real, DefaultAllocator};
use aliases::Vec;
use traits::{Alloc, Dimension};


pub fn exp<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp())
}

pub fn exp2<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp2())
}

pub fn inversesqrt<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| N::one() / x.sqrt())

}

pub fn log<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.ln())
}

pub fn log2<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.log2())
}

pub fn pow<N: Real, D: Dimension>(base: &Vec<N, D>, exponent: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    base.zip_map(exponent, |b, e| b.powf(e))
}

pub fn sqrt<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.sqrt())
}
