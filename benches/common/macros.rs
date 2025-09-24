#![macro_use]

macro_rules! bench_binop(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            use rand::SeedableRng;
            let mut rng = IsaacRng::seed_from_u64(0);
            let a = rng.random::<$t1>();
            let b = rng.random::<$t2>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(b)
            }));
        }
    }
);

macro_rules! bench_binop_ref(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            use rand::SeedableRng;
            let mut rng = IsaacRng::seed_from_u64(0);
            let a = rng.random::<$t1>();
            let b = rng.random::<$t2>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(&b)
            }));
        }
    }
);

macro_rules! bench_unop(
    ($name: ident, $t: ty, $unop: ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            const LEN: usize = 1 << 13;

            use rand::SeedableRng;
            let mut rng = IsaacRng::seed_from_u64(0);

            let mut elems: Vec<$t> =  (0usize .. LEN).map(|_| rng.random::<$t>()).collect();
            let mut i = 0;

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    std::hint::black_box(elems.get_unchecked_mut(i).$unop())
                }
            }));
        }
    }
);
