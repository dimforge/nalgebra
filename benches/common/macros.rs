#![macro_use]

macro_rules! bench_binop(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            use rand::SeedableRng;
            use std::hint::black_box;

            let mut rng = IsaacRng::seed_from_u64(0);
            let a = rng.random::<$t1>();
            let b = rng.random::<$t2>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                black_box(&a).$binop(black_box(b))
            }));
        }
    }
);

macro_rules! bench_binop_ref(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            use rand::SeedableRng;
            use std::hint::black_box;

            let mut rng = IsaacRng::seed_from_u64(0);
            let a = rng.random::<$t1>();
            let b = rng.random::<$t2>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                black_box(&a).$binop(black_box(&b))
            }));
        }
    }
);

macro_rules! bench_binop_fn(
    ($name: ident, $t1: ty, $t2: ty, $binop: path) => {
        fn $name(bh: &mut criterion::Criterion) {
            use rand::SeedableRng;
            use std::hint::black_box;

            let mut rng = IsaacRng::seed_from_u64(0);
            let a = rng.random::<$t1>();
            let b = rng.random::<$t2>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                $binop(black_box(&a), black_box(&b))
            }));
        }
    }
);

macro_rules! bench_unop(
    ($name: ident, $t: ty, $unop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            const LEN: usize = 1 << 13;

            use rand::SeedableRng;
            use std::hint::black_box;

            let mut rng = IsaacRng::seed_from_u64(0);

            let mut elems: Vec<$t> =  (0usize .. LEN).map(|_| rng.random::<$t>()).collect();
            let mut i = 0;

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    black_box(elems.get_unchecked_mut(i)).$unop()
                }
            }));
        }
    }
);
