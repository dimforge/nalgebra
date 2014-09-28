#![feature(macro_rules)]

extern crate debug;
extern crate test;
extern crate "nalgebra" as na;

use std::rand::random;
use test::Bencher;
use na::{Vec2, Vec3, Vec4, Vec5, Vec6};

macro_rules! bench_dot_vec(
    ($bh: expr, $t: ty) => {
        {
            let a: $t = random();
            let b: $t = random();
            let mut d = 0.0;

            $bh.iter(|| {
                for _ in range(0u, 1000) {
                    d = d + na::dot(&a, &b);
                }
            })
        }
    }
)

#[bench]
fn bench_dot_vec2(bh: &mut Bencher) {
    bench_dot_vec!(bh, Vec2<f64>)
}

#[bench]
fn bench_dot_vec3(bh: &mut Bencher) {
    bench_dot_vec!(bh, Vec3<f64>)
}

#[bench]
fn bench_dot_vec4(bh: &mut Bencher) {
    bench_dot_vec!(bh, Vec4<f64>)
}

#[bench]
fn bench_dot_vec5(bh: &mut Bencher) {
    bench_dot_vec!(bh, Vec5<f64>)
}

#[bench]
fn bench_dot_vec6(bh: &mut Bencher) {
    bench_dot_vec!(bh, Vec6<f64>)
}
