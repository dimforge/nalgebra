#![macro_use]

macro_rules! bench_binop(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems1: Vec<$t1> = (0us .. LEN).map(|_| rng.gen::<$t1>()).collect();
            let elems2: Vec<$t2> = (0us .. LEN).map(|_| rng.gen::<$t2>()).collect();
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(elems1.get_unchecked(i).$binop(*elems2.get_unchecked(i)))
                }
            })
        }
    }
);

macro_rules! bench_binop_na(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems1: Vec<$t1> = (0us .. LEN).map(|_| rng.gen::<$t1>()).collect();
            let elems2: Vec<$t2> = (0us .. LEN).map(|_| rng.gen::<$t2>()).collect();
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(na::$binop(elems1.get_unchecked(i), elems2.get_unchecked(i)))
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

            let elems: Vec<$t> =  (0us .. LEN).map(|_| rng.gen::<$t>()).collect();
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

macro_rules! bench_unop_self(
    ($name: ident, $t: ty, $unop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: usize = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let mut elems: Vec<$t> =  (0us .. LEN).map(|_| rng.gen::<$t>()).collect();
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

            $(let $args: Vec<$types> = (0us .. LEN).map(|_| rng.gen::<$types>()).collect();)*
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
