use std::rand::random;
use extra::test::BenchHarness;
use vec::*;

macro_rules! bench_dot_vec(
    ($bh: expr, $t: ty) => {
        {
            let a: $t = random();
            let b: $t = random();
            let mut d = 0.0;

            do $bh.iter {
                do 1000.times {
                    d = d + a.dot(&b);
                }
            }
        }
    }
)

#[bench]
fn bench_dot_vec2(bh: &mut BenchHarness) {
    bench_dot_vec!(bh, Vec2<f64>)
}

#[bench]
fn bench_dot_vec3(bh: &mut BenchHarness) {
    bench_dot_vec!(bh, Vec3<f64>)
}

#[bench]
fn bench_dot_vec4(bh: &mut BenchHarness) {
    bench_dot_vec!(bh, Vec4<f64>)
}

#[bench]
fn bench_dot_vec5(bh: &mut BenchHarness) {
    bench_dot_vec!(bh, Vec5<f64>)
}

#[bench]
fn bench_dot_vec6(bh: &mut BenchHarness) {
    bench_dot_vec!(bh, Vec6<f64>)
}
