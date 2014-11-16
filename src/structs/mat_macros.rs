#![macro_escape]

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

macro_rules! at_fast_impl(
    ($t: ident, $dim: expr) => (
        impl<N: Clone> $t<N> {
            #[inline]
            pub unsafe fn at_fast(&self, (i, j): (uint, uint)) -> N {
                (*mem::transmute::<&$t<N>, &[N, ..$dim * $dim]>(self)
                 .unsafe_get(i + j * $dim)).clone()
            }

            #[inline]
            pub unsafe fn set_fast(&mut self, (i, j): (uint, uint), val: N) {
                (*mem::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)
                 .unsafe_mut(i + j * $dim)) = val
            }
        }
    )
)

macro_rules! mat_cast_impl(
    ($t: ident, $tcast: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<Nin: Clone, Nout: Clone + Cast<Nin>> $tcast<Nout> for $t<Nin> {
            #[inline]
            fn to(v: $t<Nin>) -> $t<Nout> {
                $t::new(Cast::from(v.$comp0.clone()) $(, Cast::from(v.$compN.clone()))*)
            }
        }
    )
)

macro_rules! add_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Add<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 + right.$comp0 $(, left.$compN + right.$compN)*)
            }
        }
    )
)

macro_rules! sub_impl(
    ($t: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl<N: Sub<N, N>> $trhs<N, $t<N>> for $t<N> {
            #[inline]
            fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
                $t::new(left.$comp0 - right.$comp0 $(, left.$compN - right.$compN)*)
            }
        }
    )
)

macro_rules! mat_mul_scalar_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 * *right $(, left.$compN * *right)*)
            }
        }
    )
)

macro_rules! mat_div_scalar_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 / *right $(, left.$compN / *right)*)
            }
        }
    )
)

macro_rules! mat_add_scalar_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 + *right $(, left.$compN + *right)*)
            }
        }
    )
)


macro_rules! eye_impl(
    ($t: ident, $ndim: expr, $($comp_diagN: ident),+) => (
        impl<N: Zero + One> Eye for $t<N> {
            fn new_identity(dim: uint) -> $t<N> {
                assert!(dim == $ndim);
                let mut eye: $t<N> = ::zero();
                $(eye.$comp_diagN = ::one();)+
                eye
            }
        }
    )
)

macro_rules! mat_sub_scalar_impl(
    ($t: ident, $n: ident, $trhs: ident, $comp0: ident $(,$compN: ident)*) => (
        impl $trhs<$n, $t<$n>> for $n {
            #[inline]
            fn binop(left: &$t<$n>, right: &$n) -> $t<$n> {
                $t::new(left.$comp0 - *right $(, left.$compN - *right)*)
            }
        }
    )
)

macro_rules! absolute_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N: Absolute<N>> Absolute<$t<N>> for $t<N> {
        #[inline]
        fn abs(m: &$t<N>) -> $t<N> {
            $t::new(::abs(&m.$comp0) $(, ::abs(&m.$compN) )*)
        }
    }
  )
)

macro_rules! iterable_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Iterable<N> for $t<N> {
        #[inline]
        fn iter<'l>(&'l self) -> Items<'l, N> {
            unsafe {
                mem::transmute::<&'l $t<N>, &'l [N, ..$dim * $dim]>(self).iter()
            }
        }
    }
  )
)

macro_rules! iterable_mut_impl(
  ($t: ident, $dim: expr) => (
    impl<N> IterableMut<N> for $t<N> {
        #[inline]
        fn iter_mut<'l>(&'l mut self) -> MutItems<'l, N> {
            unsafe {
                mem::transmute::<&'l mut $t<N>, &'l mut [N, ..$dim * $dim]>(self).iter_mut()
            }
        }
    }
  )
)

macro_rules! one_impl(
  ($t: ident, $value0: expr $(, $valueN: expr)* ) => (
    impl<N: Clone + BaseNum> One for $t<N> {
        #[inline]
        fn one() -> $t<N> {
            $t::new($value0() $(, $valueN() )*)
        }
    }
  )
)

macro_rules! zero_impl(
  ($t: ident, $comp0: ident $(, $compN: ident)* ) => (
    impl<N: Zero> Zero for $t<N> {
        #[inline]
        fn zero() -> $t<N> {
            $t {
                $comp0: ::zero()
                $(, $compN: ::zero() )*
            }
        }

        #[inline]
        fn is_zero(&self) -> bool {
            ::is_zero(&self.$comp0) $(&& ::is_zero(&self.$compN) )*
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
    impl<N> Shape<(uint, uint), N> for $t<N> {
        #[inline]
        fn shape(&self) -> (uint, uint) {
            ($dim, $dim)
        }
    }

    impl<N: Clone> Indexable<(uint, uint), N> for $t<N> {
        #[inline]
        fn at(&self, (i, j): (uint, uint)) -> N {
            unsafe {
                mem::transmute::<&$t<N>, &[N, ..$dim * $dim]>(self)[i + j * $dim].clone()
            }
        }

        #[inline]
        fn set(&mut self, (i, j): (uint, uint), val: N) {
            unsafe {
                mem::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)[i + j * $dim] = val
            }
        }

        #[inline]
        fn swap(&mut self, (i1, j1): (uint, uint), (i2, j2): (uint, uint)) {
            unsafe {
              mem::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)
                .swap(i1 + j1 * $dim, i2 + j2 * $dim)
            }
        }

        #[inline]
        unsafe fn unsafe_at(&self, (i, j): (uint, uint)) -> N {
            (*mem::transmute::<&$t<N>, &[N, ..$dim * $dim]>(self).unsafe_get(i + j * $dim)).clone()
        }

        #[inline]
        unsafe fn unsafe_set(&mut self, (i, j): (uint, uint), val: N) {
            (*mem::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self).unsafe_mut(i + j * $dim)) = val
        }
    }
  )
)

macro_rules! index_impl(
    ($t: ident, $dim: expr) => (
        impl<N> Index<(uint, uint), N> for $t<N> {
            fn index(&self, &(i, j): &(uint, uint)) -> &N {
                unsafe {
                    &mem::transmute::<&$t<N>, &mut [N, ..$dim * $dim]>(self)[i + j * $dim]
                }
            }
        }

        impl<N> IndexMut<(uint, uint), N> for $t<N> {
            fn index_mut(&mut self, &(i, j): &(uint, uint)) -> &mut N {
                unsafe {
                    &mut mem::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)[i + j * $dim]
                }
            }
        }
    )
)

macro_rules! col_slice_impl(
    ($t: ident, $tv: ident, $slice: ident, $dim: expr) => (
        impl<N: Clone + Zero> ColSlice<$slice<N>> for $t<N> {
            fn col_slice(&self, cid: uint, rstart: uint, rend: uint) -> $slice<N> {
                let col = self.col(cid);

                $slice::from_slice(rend - rstart, col.as_slice().slice(rstart, rend))
            }
        }
    )
)

macro_rules! row_impl(
  ($t: ident, $tv: ident, $dim: expr) => (
    impl<N: Clone + Zero> Row<$tv<N>> for $t<N> {
        #[inline]
        fn nrows(&self) -> uint {
            Dim::dim(None::<$t<N>>)
        }

        #[inline]
        fn set_row(&mut self, row: uint, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self.set((row, i), e.clone());
            }
        }

        #[inline]
        fn row(&self, row: uint) -> $tv<N> {
            let mut res: $tv<N> = ::zero();

            for (i, e) in res.iter_mut().enumerate() {
                *e = self.at((row, i));
            }

            res
        }
    }
  )
)

macro_rules! row_slice_impl(
    ($t: ident, $tv: ident, $slice: ident, $dim: expr) => (
        impl<N: Clone + Zero> RowSlice<$slice<N>> for $t<N> {
            fn row_slice(&self, rid: uint, cstart: uint, cend: uint) -> $slice<N> {
                let row = self.row(rid);

                $slice::from_slice(cend - cstart, row.as_slice().slice(cstart, cend))
            }
        }
    )
)

macro_rules! col_impl(
  ($t: ident, $tv: ident, $dim: expr) => (
    impl<N: Clone + Zero> Col<$tv<N>> for $t<N> {
        #[inline]
        fn ncols(&self) -> uint {
            Dim::dim(None::<$t<N>>)
        }

        #[inline]
        fn set_col(&mut self, col: uint, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self.set((i, col), e.clone());
            }
        }

        #[inline]
        fn col(&self, col: uint) -> $tv<N> {
            let mut res: $tv<N> = ::zero();

            for (i, e) in res.iter_mut().enumerate() {
                *e = self.at((i, col));
            }

            res
        }
    }
  )
)

macro_rules! diag_impl(
    ($t: ident, $tv: ident, $dim: expr) => (
        impl<N: Clone + Zero> Diag<$tv<N>> for $t<N> {
            #[inline]
            fn from_diag(diag: &$tv<N>) -> $t<N> {
                let mut res: $t<N> = ::zero();

                res.set_diag(diag);

                res
            }

            #[inline]
            fn set_diag(&mut self, diag: &$tv<N>) {
                for i in range(0, $dim) {
                    unsafe { self.unsafe_set((i, i), diag.unsafe_at(i)) }
                }
            }

            #[inline]
            fn diag(&self) -> $tv<N> {
                let mut diag: $tv<N> = ::zero();

                for i in range(0, $dim) {
                    unsafe { diag.unsafe_set(i, self.unsafe_at((i, i))) }
                }

                diag
            }
        }
    )
)

macro_rules! mat_mul_mat_impl(
  ($t: ident, $trhs: ident, $dim: expr) => (
    impl<N: Clone + BaseNum> $trhs<N, $t<N>> for $t<N> {
        #[inline]
        fn binop(left: &$t<N>, right: &$t<N>) -> $t<N> {
            // careful! we need to comute other * self here (self is the rhs).
            let mut res: $t<N> = ::zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    let mut acc: N = ::zero();

                    unsafe {
                        for k in range(0u, $dim) {
                            acc = acc + left.at_fast((i, k)) * right.at_fast((k, j));
                        }

                        res.set_fast((i, j), acc);
                    }
                }
            }

            res
        }
    }
  )
)

macro_rules! vec_mul_mat_impl(
  ($t: ident, $v: ident, $trhs: ident, $dim: expr, $zero: expr) => (
    impl<N: Clone + BaseNum> $trhs<N, $v<N>> for $t<N> {
        #[inline]
        fn binop(left: &$v<N>, right: &$t<N>) -> $v<N> {
            let mut res : $v<N> = $zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    unsafe {
                        let val = res.at_fast(i) + left.at_fast(j) * right.at_fast((j, i));
                        res.set_fast(i, val)
                    }
                }
            }

            res
        }
    }
  )
)

macro_rules! mat_mul_vec_impl(
  ($t: ident, $v: ident, $trhs: ident, $dim: expr, $zero: expr) => (
    impl<N: Clone + BaseNum> $trhs<N, $v<N>> for $v<N> {
        #[inline]
        fn binop(left: &$t<N>, right: &$v<N>) -> $v<N> {
            let mut res : $v<N> = $zero();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    unsafe {
                        let val = res.at_fast(i) + left.at_fast((i, j)) * right.at_fast(j);
                        res.set_fast(i, val)
                    }
                }
            }

            res
        }
    }
  )
)

macro_rules! pnt_mul_mat_impl(
  ($t: ident, $v: ident, $trhs: ident, $dim: expr, $zero: expr) => (
      vec_mul_mat_impl!($t, $v, $trhs, $dim, $zero)
  )
)

macro_rules! mat_mul_pnt_impl(
  ($t: ident, $v: ident, $trhs: ident, $dim: expr, $zero: expr) => (
      mat_mul_vec_impl!($t, $v, $trhs, $dim, $zero)
  )
)

macro_rules! inv_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + BaseNum>
    Inv for $t<N> {
        #[inline]
        fn inv_cpy(m: &$t<N>) -> Option<$t<N>> {
            let mut res : $t<N> = m.clone();

            if res.inv() {
                Some(res)
            }
            else {
                None
            }
        }

        fn inv(&mut self) -> bool {
            let mut res: $t<N> = ::one();

            // inversion using Gauss-Jordan elimination
            for k in range(0u, $dim) {
                // search a non-zero value on the k-th column
                // FIXME: would it be worth it to spend some more time searching for the
                // max instead?

                let mut n0 = k; // index of a non-zero entry

                while n0 != $dim {
                    if self.at((n0, k)) != ::zero() {
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
        fn transpose_cpy(m: &$t<N>) -> $t<N> {
            let mut res = m.clone();

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
        fn approx_epsilon(_: Option<$t<N>>) -> N {
            ApproxEq::approx_epsilon(None::<N>)
        }

        #[inline]
        fn approx_eq(a: &$t<N>, b: &$t<N>) -> bool {
            let mut zip = a.iter().zip(b.iter());

            zip.all(|(a, b)| ApproxEq::approx_eq(a, b))
        }

        #[inline]
        fn approx_eq_eps(a: &$t<N>, b: &$t<N>, epsilon: &N) -> bool {
            let mut zip = a.iter().zip(b.iter());

            zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
        }
    }
  )
)

macro_rules! to_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: BaseNum + Clone> ToHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn to_homogeneous(m: &$t<N>) -> $t2<N> {
            let mut res: $t2<N> = ::one();

            for i in range(0u, $dim) {
                for j in range(0u, $dim) {
                    res.set((i, j), m.at((i, j)))
                }
            }

            res
        }
    }
  )
)

macro_rules! from_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: BaseNum + Clone> FromHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn from(m: &$t2<N>) -> $t<N> {
            let mut res: $t<N> = ::one();

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
        impl<N: Clone + Mul<N, N> + Zero> Outer<$m<N>> for $t<N> {
            #[inline]
            fn outer(a: &$t<N>, b: &$t<N>) -> $m<N> {
                let mut res: $m<N> = ::zero();

                for i in range(0u, Dim::dim(None::<$t<N>>)) {
                    for j in range(0u, Dim::dim(None::<$t<N>>)) {
                        res.set((i, j), a.at(i) * b.at(j))
                    }
                }

                res
            }
        }
    )
)

macro_rules! eigen_qr_impl(
    ($t: ident, $v: ident) => (
        impl<N> EigenQR<N, $v<N>> for $t<N>
            where N: BaseNum + One + Zero + BaseFloat + ApproxEq<N> + Clone {
            fn eigen_qr(m: &$t<N>, eps: &N, niter: uint) -> ($t<N>, $v<N>) {
                linalg::eigen_qr(m, eps, niter)
            }
        }
    )
)
