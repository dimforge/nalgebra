#![macro_use]

macro_rules! bench_binop(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            let mut rng = IsaacRng::new_unseeded();
            let a = rng.gen::<$t1>();
            let b = rng.gen::<$t2>();

            bh.iter(|| {
                a.$binop(b)
            })
        }
    }
);

macro_rules! bench_binop_ref(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            let mut rng = IsaacRng::new_unseeded();
            let a = rng.gen::<$t1>();
            let b = rng.gen::<$t2>();

            bh.iter(|| {
                a.$binop(&b)
            })
        }
    }
);

macro_rules! bench_binop_fn(
    ($name: ident, $t1: ty, $t2: ty, $binop: path) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            let mut rng = IsaacRng::new_unseeded();
            let a = rng.gen::<$t1>();
            let b = rng.gen::<$t2>();

            bh.iter(|| {
                $binop(&a, &b)
            })
        }
    }
);

macro_rules! bench_unop_na(
    ($name: ident, $t: ty, $unop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems: Vec<$t> =  (0usize .. LEN).map(|_| rng.gen::<$t>()).collect();
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(na::$unop(elems.get_unchecked(i)))
                }
            })
        }
    }
);

macro_rules! bench_unop(
    ($name: ident, $t: ty, $unop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let mut elems: Vec<$t> =  (0usize .. LEN).map(|_| rng.gen::<$t>()).collect();
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(elems.get_unchecked_mut(i).$unop())
                }
            })
        }
    }
);

macro_rules! bench_construction(
    ($name: ident, $constructor: path, $( $args: ident: $types: ty),*) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            $(let $args: Vec<$types> = (0usize .. LEN).map(|_| rng.gen::<$types>()).collect();)*
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    let res = $constructor($(*$args.get_unchecked(i),)*);
                    test::black_box(res)
                }
            })
        }
    }
);
