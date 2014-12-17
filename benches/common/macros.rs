#![macro_escape]

macro_rules! bench_binop(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: uint = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems1 =  Vec::from_fn(LEN, |_| rng.gen::<$t1>());
            let elems2 =  Vec::from_fn(LEN, |_| rng.gen::<$t2>());
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(elems1.unsafe_get(i).$binop(*elems2.unsafe_get(i)))
                }
            })
        }
    }
)

macro_rules! bench_binop_na(
    ($name: ident, $t1: ty, $t2: ty, $binop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: uint = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems1 =  Vec::from_fn(LEN, |_| rng.gen::<$t1>());
            let elems2 =  Vec::from_fn(LEN, |_| rng.gen::<$t2>());
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(na::$binop(elems1.unsafe_get(i), elems2.unsafe_get(i)))
                }
            })
        }
    }
)

macro_rules! bench_unop(
    ($name: ident, $t: ty, $unop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: uint = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let elems =  Vec::from_fn(LEN, |_| rng.gen::<$t>());
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(na::$unop(elems.unsafe_get(i)))
                }
            })
        }
    }
)

macro_rules! bench_unop_self(
    ($name: ident, $t: ty, $unop: ident) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: uint = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            let mut elems =  Vec::from_fn(LEN, |_| rng.gen::<$t>());
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    test::black_box(elems.unsafe_mut(i).$unop())
                }
            })
        }
    }
)

macro_rules! bench_construction(
    ($name: ident, $constructor: path $(, $args: ident: $types: ty)*) => {
        #[bench]
        fn $name(bh: &mut Bencher) {
            const LEN: uint = 1 << 13;

            let mut rng = IsaacRng::new_unseeded();

            $(let $args = Vec::from_fn(LEN, |_| rng.gen::<$types>());)*
            let mut i = 0;

            bh.iter(|| {
                i = (i + 1) & (LEN - 1);

                unsafe {
                    let res = $constructor($(*$args.unsafe_get(i),)*);
                    test::black_box(res)
                }
            })
        }
    }
)
