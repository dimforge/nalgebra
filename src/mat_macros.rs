#[macro_escape];

macro_rules! mat_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N> $t<N> {
        #[inline]
        pub fn new($comp0: N $(, $compN: N )*) -> $t<N> {
            $t {
                $comp0: $comp0
                $(, $compN: $compN )*
            }
        }
    }
  )
)

macro_rules! mat_cast_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<Nin: NumCast + Clone, Nout: NumCast> MatCast<$t<Nout>> for $t<Nin> {
        #[inline]
        fn from(m: $t<Nin>) -> $t<Nout> {
            $t::new(NumCast::from(m.$comp0.clone()) $(, NumCast::from(m.$compN.clone()) )*)
        }
    }
  )
)

macro_rules! add_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Add<N, N>> Add<$t<N>, $t<N>> for $t<N> {
        #[inline]
        fn add(&self, other: &$t<N>) -> $t<N> {
            $t::new(self.$comp0 + other.$comp0 $(, self.$compN + other.$compN )*)
        }
    }
  )
)

macro_rules! sub_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Sub<N, N>> Sub<$t<N>, $t<N>> for $t<N> {
        #[inline]
        fn sub(&self, other: &$t<N>) -> $t<N> {
            $t::new(self.$comp0 - other.$comp0 $(, self.$compN - other.$compN )*)
        }
    }
  )
)

macro_rules! scalar_mul_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Mul<N, N>> $t<N> {
        #[inline]
        /// Scalar multiplication of each component of this matrix by a scalar.
        pub fn scalar_mul(&self, other: &N) -> $t<N> {
            $t::new(self.$comp0 * *other $(, self.$compN * *other )*)
        }
    }
  )
)

macro_rules! scalar_div_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Div<N, N>> $t<N> {
        #[inline]
        /// Scalar division of each component of this matrix by a scalar.
        pub fn scalar_div(&self, other: &N) -> $t<N> {
            $t::new(self.$comp0 / *other $(, self.$compN / *other )*)
        }
    }
  )
)

macro_rules! scalar_add_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Add<N, N>> ScalarAdd<N> for $t<N> {
        #[inline]
        fn scalar_add(&self, other: &N) -> $t<N> {
            $t::new(self.$comp0 + *other $(, self.$compN + *other )*)
        }

        #[inline]
        fn scalar_add_inplace(&mut self, other: &N) {
            self.$comp0 = self.$comp0 + *other;
            $(self.$compN = self.$compN + *other; )*
        }
    }
  )
)

macro_rules! scalar_sub_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Sub<N, N>> ScalarSub<N> for $t<N> {
        #[inline]
        fn scalar_sub(&self, other: &N) -> $t<N> {
            $t::new(self.$comp0 - *other $(, self.$compN - *other )*)
        }

        #[inline]
        fn scalar_sub_inplace(&mut self, other: &N) {
            self.$comp0 = self.$comp0 - *other;
            $(self.$compN = self.$compN - *other; )*
        }
    }
  )
)

macro_rules! iterable_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Iterable<N> for $t<N> {
        #[inline]
        fn iter<'l>(&'l self) -> VecIterator<'l, N> {
            unsafe {
                cast::transmute::<&'l $t<N>, &'l [N, ..$dim * $dim]>(self).iter()
            }
        }
    }
  )
)

macro_rules! iterable_mut_impl(
  ($t: ident, $dim: expr) => (
    impl<N> IterableMut<N> for $t<N> {
        #[inline]
        fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N> {
            unsafe {
                cast::transmute::<&'l mut $t<N>, &'l mut [N, ..$dim * $dim]>(self).mut_iter()
            }
        }
    }
  )
)

macro_rules! one_impl(
  ($t: ident, $value0: ident $(, $valueN: ident)* ) => (
    impl<N: Clone + One + Zero> One for $t<N> {
        #[inline]
        fn one() -> $t<N> {
            let (_0, _1): (N, N) = (Zero::zero(), One::one());
            return $t::new($value0.clone() $(, $valueN.clone() )*)
        }
    }
  )
)

macro_rules! dim_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Dim for $t<N> {
        #[inline]
        fn dim(_: Option<$t<N>>) -> uint {
            $dim
        }
    }
  )
)

macro_rules! indexable_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone> Indexable<(uint, uint), N> for $t<N> {
        #[inline]
        fn at(&self, (i, j): (uint, uint)) -> N {
            unsafe {
                cast::transmute::<&$t<N>, &[N, ..$dim * $dim]>(self)[i * $dim + j].clone()
            }
        }

        #[inline]
        fn set(&mut self, (i, j): (uint, uint), val: N) {
            unsafe {
                cast::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)[i * $dim + j] = val
            }
        }

        #[inline]
        fn swap(&mut self, (i1, j1): (uint, uint), (i2, j2): (uint, uint)) {
            unsafe {
              cast::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)
                .swap(i1 * $dim + j1, i2 * $dim + j2)
            }
        }
    }
  )
)

macro_rules! column_impl(
  ($t: ident, $tv: ident, $dim: expr) => (
    impl<N: Clone + Zero> Column<$tv<N>> for $t<N> {
        #[inline]
        fn set_column(&mut self, col: uint, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self.set((i, col), e.clone());
            }
        }

        #[inline]
        fn column(&self, col: uint) -> $tv<N> {
            let mut res: $tv<N> = Zero::zero();

            for (i, e) in res.mut_iter().enumerate() {
                *e = self.at((i, col));
            }

            res
        }
    }
  )
)

macro_rules! row_impl(
  ($t: ident, $tv: ident, $dim: expr) => (
    impl<N: Clone + Zero> Row<$tv<N>> for $t<N> {
        #[inline]
        fn set_row(&mut self, row: uint, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self.set((row, i), e.clone());
            }
        }

        #[inline]
        fn row(&self, row: uint) -> $tv<N> {
            let mut res: $tv<N> = Zero::zero();

            for (i, e) in res.mut_iter().enumerate() {
                *e = self.at((row, i));
            }

            res
        }
    }
  )
)

macro_rules! mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Num> Mul<$t<N>, $t<N>> for $t<N> {
        #[inline]
        fn mul(&self, other: &$t<N>) -> $t<N> {
            let mut res: $t<N> = Zero::zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    let mut acc: N = Zero::zero();

                    for k in range(0u, $dim) {
                        acc = acc + self.at((i, k)) * other.at((k, j));
                    }

                    res.set((i, j), acc);
                }
            }

            res
        }
    }
  )
)

macro_rules! rmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Clone + Num> RMul<$v<N>> for $t<N> {
        #[inline]
        fn rmul(&self, other: &$v<N>) -> $v<N> {
            let mut res : $v<N> = Zero::zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    let val = res.at(i) + other.at(j) * self.at((i, j));
                    res.set(i, val)
                }
            }

            res
        }
    }
  )
)

macro_rules! lmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Clone + Num> LMul<$v<N>> for $t<N> {
        #[inline]
        fn lmul(&self, other: &$v<N>) -> $v<N> {
            let mut res : $v<N> = Zero::zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                  let val = res.at(i) + other.at(j) * self.at((j, i));
                  res.set(i, val)
                }
            }

            res
        }
    }
  )
)

macro_rules! transform_impl(
  ($t: ident, $v: ident) => (
    impl<N: Clone + Num + Eq>
    Transform<$v<N>> for $t<N> {
        #[inline]
        fn transform(&self, v: &$v<N>) -> $v<N> {
            self.rmul(v)
        }

        #[inline]
        fn inv_transform(&self, v: &$v<N>) -> $v<N> {
            match self.inverse() {
                Some(t) => t.transform(v),
                None    => fail!("Cannot use inv_transform on a non-inversible matrix.")
            }
        }
    }
  )
)

macro_rules! inv_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Eq + Num>
    Inv for $t<N> {
        #[inline]
        fn inverse(&self) -> Option<$t<N>> {
            let mut res : $t<N> = self.clone();

            if res.inplace_inverse() {
                Some(res)
            }
            else {
                None
            }
        }

        fn inplace_inverse(&mut self) -> bool {
            let mut res: $t<N> = One::one();
            let     _0N: N     = Zero::zero();

            // inversion using Gauss-Jordan elimination
            for k in range(0u, $dim) {
                // search a non-zero value on the k-th column
                // FIXME: would it be worth it to spend some more time searching for the
                // max instead?

                let mut n0 = k; // index of a non-zero entry

                while (n0 != $dim) {
                    if self.at((n0, k)) != _0N {
                        break;
                    }

                    n0 = n0 + 1;
                }

                if n0 == $dim {
                    return false
                }

                // swap pivot line
                if n0 != k {
                    for j in range(0u, $dim) {
                        self.swap((n0, j), (k, j));
                        res.swap((n0, j), (k, j));
                    }
                }

                let pivot = self.at((k, k));

                for j in range(k, $dim) {
                    let selfval = self.at((k, j)) / pivot;
                    self.set((k, j), selfval);
                }

                for j in range(0u, $dim) {
                    let resval = res.at((k, j)) / pivot;
                    res.set((k, j), resval);
                }

                for l in range(0u, $dim) {
                    if l != k {
                        let normalizer = self.at((l, k));

                        for j in range(k, $dim) {
                            let selfval = self.at((l, j)) - self.at((k, j)) * normalizer;
                            self.set((l, j), selfval);
                        }

                        for j in range(0u, $dim) {
                            let resval  = res.at((l, j)) - res.at((k, j)) * normalizer;
                            res.set((l, j), resval);
                        }
                    }
                }
            }

            *self = res;

            true
        }
    }
  )
)

macro_rules! transpose_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone> Transpose for $t<N> {
        #[inline]
        fn transposed(&self) -> $t<N> {
            let mut res = self.clone();

            res.transpose();

            res
        }

        #[inline]
        fn transpose(&mut self) {
            for i in range(1u, $dim) {
                for j in range(0u, i) {
                    self.swap((i, j), (j, i))
                }
            }
        }
    }
  )
)

macro_rules! approx_eq_impl(
  ($t: ident) => (
    impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
        #[inline]
        fn approx_epsilon() -> N {
            fail!("approx_epsilon is broken since rust revision 8693943676487c01fa09f5f3daf0df6a1f71e24d.")
            // ApproxEq::<N>::approx_epsilon()
        }

        #[inline]
        fn approx_eq(&self, other: &$t<N>) -> bool {
            let mut zip = self.iter().zip(other.iter());

            do zip.all |(a, b)| {
                a.approx_eq(b)
            }
        }

        #[inline]
        fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
            let mut zip = self.iter().zip(other.iter());

            do zip.all |(a, b)| {
                a.approx_eq_eps(b, epsilon)
            }
        }
    }
  )
)

macro_rules! to_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: One + Zero + Clone> ToHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn to_homogeneous(&self) -> $t2<N> {
            let mut res: $t2<N> = One::one();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    res.set((i, j), self.at((i, j)))
                }
            }

            res
        }
    }
  )
)

macro_rules! from_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: One + Zero + Clone> FromHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn from(m: &$t2<N>) -> $t<N> {
            let mut res: $t<N> = One::one();

            for i in range(0u, $dim2) {
                for j in range(0u, $dim2) {
                    res.set((i, j), m.at((i, j)))
                }
            }

            // FIXME: do we have to deal the lost components
            // (like if the 1 is not a 1… do we have to divide?)

            res
        }
    }
  )
)

macro_rules! outer_impl(
    ($t: ident, $m: ident) => (
        impl<N: Mul<N, N> + Zero + Clone> Outer<$t<N>, $m<N>> for $t<N> {
            #[inline]
            fn outer(&self, other: &$t<N>) -> $m<N> {
                let mut res: $m<N> = Zero::zero();

                let _dim: Option<$t<N>> = None;
                for i in range(0u, Dim::dim(_dim)) {
                    let _dim: Option<$t<N>> = None;
                    for j in range(0u, Dim::dim(_dim)) {
                        res.set((i, j), self.at(i) * other.at(j))
                    }
                }

                res
            }
        }
    )
)
