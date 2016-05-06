#![macro_use]

macro_rules! mat_impl(
  ($t: ident, $($compN: ident),+) => (
    impl<N> $t<N> {
        #[inline]
        pub fn new($($compN: N ),+) -> $t<N> {
            $t {
                $($compN: $compN ),+
            }
        }
    }
  )
);

macro_rules! conversion_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> AsRef<[[N; $dimension]; $dimension]> for $t<N> {
            #[inline]
            fn as_ref(&self) -> &[[N; $dimension]; $dimension] {
                unsafe {
                    mem::transmute(self)
                }
            }
        }

        impl<N> AsMut<[[N; $dimension]; $dimension]> for $t<N> {
            #[inline]
            fn as_mut(&mut self) -> &mut [[N; $dimension]; $dimension] {
                unsafe {
                    mem::transmute(self)
                }
            }
        }

        impl<'a, N> From<&'a [[N; $dimension]; $dimension]> for &'a $t<N> {
            #[inline]
            fn from(arr: &'a [[N; $dimension]; $dimension]) -> &'a $t<N> {
                unsafe {
                    mem::transmute(arr)
                }
            }
        }

        impl<'a, N> From<&'a mut [[N; $dimension]; $dimension]> for &'a mut $t<N> {
            #[inline]
            fn from(arr: &'a mut [[N; $dimension]; $dimension]) -> &'a mut $t<N> {
                unsafe {
                    mem::transmute(arr)
                }
            }
        }

        impl<'a, N: Clone> From<&'a [[N; $dimension]; $dimension]> for $t<N> {
            #[inline]
            fn from(arr: &'a [[N; $dimension]; $dimension]) -> $t<N> {
                let tref: &$t<N> = From::from(arr);
                tref.clone()
            }
        }
    )
);

macro_rules! at_fast_impl(
    ($t: ident, $dimension: expr) => (
        impl<N: Copy> $t<N> {
            #[inline]
            pub unsafe fn at_fast(&self, (i, j): (usize, usize)) -> N {
                (*mem::transmute::<&$t<N>, &[N; $dimension * $dimension]>(self)
                 .get_unchecked(i + j * $dimension))
            }

            #[inline]
            pub unsafe fn set_fast(&mut self, (i, j): (usize, usize), val: N) {
                (*mem::transmute::<&mut $t<N>, &mut [N; $dimension * $dimension]>(self)
                 .get_unchecked_mut(i + j * $dimension)) = val
            }
        }
    )
);

macro_rules! mat_cast_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<Nin: Copy, Nout: Copy + Cast<Nin>> Cast<$t<Nin>> for $t<Nout> {
            #[inline]
            fn from(v: $t<Nin>) -> $t<Nout> {
                $t::new($(Cast::from(v.$compN)),+)
            }
        }
    )
);

macro_rules! add_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Add<N, Output = N>> Add<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN + right.$compN),+)
            }
        }

        impl<N: AddAssign<N>> AddAssign<$t<N>> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: $t<N>) {
                $( self.$compN += right.$compN; )+
            }
        }
    )
);

macro_rules! sub_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Sub<N, Output = N>> Sub<$t<N>> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: $t<N>) -> $t<N> {
                $t::new($(self.$compN - right.$compN),+)
            }
        }


        impl<N: SubAssign<N>> SubAssign<$t<N>> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: $t<N>) {
                $( self.$compN -= right.$compN; )+
            }
        }
    )
);

macro_rules! mat_mul_scalar_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Mul<N, Output = N>> Mul<N> for N {
            type Output = $t<N>;

            #[inline]
            fn mul(self, right: N) -> $t<N> {
                $t::new($(self.$compN * *right),+)
            }
        }

        impl<N: MulAssign<N>> MulAssign<N> for $t<N> {
            #[inline]
            fn mul_assign(&mut self, right: N) {
                $( self.$compN *= *right; )+
            }
        }

        impl Mul<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn mul(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self * right.$compN),+)
            }
        }

        impl Mul<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn mul(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self * right.$compN),+)
            }
        }
    )
);

macro_rules! mat_div_scalar_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Div<N, Output = N>> Div<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn div(self, right: N) -> $t<N> {
                $t::new($(self.$compN / *right),+)
            }
        }

        impl<N: DivAssign<N>> DivAssign<N> for $t<N> {
            #[inline]
            fn div_assign(&mut self, right: N) {
                $( self.$compN /= *right; )+
            }
        }
    )
);

macro_rules! mat_add_scalar_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Add<N, Output = N>> Add<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn add(self, right: N) -> $t<N> {
                $t::new($(self.$compN + *right),+)
            }
        }

        impl<N: AddAssign<N>> AddAssign<N> for $t<N> {
            #[inline]
            fn add_assign(&mut self, right: N) {
                $( self.$compN += *right; )+
            }
        }

        impl Add<$t<f32>> for f32 {
            type Output = $t<f32>;

            #[inline]
            fn add(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self + right.$compN),+)
            }
        }

        impl Add<$t<f64>> for f64 {
            type Output = $t<f64>;

            #[inline]
            fn add(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self + right.$compN),+)
            }
        }
    )
);

macro_rules! mat_sub_scalar_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Sub<N, Output = N>> Sub<N> for $t<N> {
            type Output = $t<N>;

            #[inline]
            fn sub(self, right: &N) -> $t<N> {
                $t::new($(self.$compN - *right),+)
            }
        }

        impl<N: SubAssign<N>> SubAssign<N> for $t<N> {
            #[inline]
            fn sub_assign(&mut self, right: N) {
                $( self.$compN -= *right; )+
            }
        }

        impl Sub<f32> for $t<f32> {
            type Output = $t<f32>;

            #[inline]
            fn sub(self, right: $t<f32>) -> $t<f32> {
                $t::new($(self - right.$compN),+)
            }
        }

        impl Sub<f64> for $t<f64> {
            type Output = $t<f64>;

            #[inline]
            fn sub(self, right: $t<f64>) -> $t<f64> {
                $t::new($(self - right.$compN),+)
            }
        }
    )
);


macro_rules! eye_impl(
    ($t: ident, $dimension: expr, $($comp_diagN: ident),+) => (
        impl<N: Zero + One> Eye for $t<N> {
            fn new_identity(dimension: usize) -> $t<N> {
                assert!(dimension == $dimension);
                let mut eye: $t<N> = ::zero();
                $(eye.$comp_diagN = ::one();)+
                eye
            }
        }
    )
);

macro_rules! repeat_impl(
    ($t: ident, $($compN: ident),+) => (
        impl<N: Copy> Repeat<N> for $t<N> {
            fn repeat(val: N) -> $t<N> {
                $t {
                    $($compN: val ),+
                }
            }
        }
    )
);

macro_rules! absolute_impl(
  ($t: ident, $($compN: ident),+) => (
    impl<N: Absolute<N>> Absolute<$t<N>> for $t<N> {
        #[inline]
        fn abs(m: &$t<N>) -> $t<N> {
            $t::new($(::abs(&m.$compN) ),+)
        }
    }
  )
);

macro_rules! iterable_impl(
  ($t: ident, $dimension: expr) => (
    impl<N> Iterable<N> for $t<N> {
        #[inline]
        fn iter(&self) -> Iter<N> {
            unsafe {
                mem::transmute::<&$t<N>, &[N; $dimension * $dimension]>(self).iter()
            }
        }
    }
  )
);

macro_rules! iterable_mut_impl(
  ($t: ident, $dimension: expr) => (
    impl<N> IterableMut<N> for $t<N> {
        #[inline]
        fn iter_mut(& mut self) -> IterMut<N> {
            unsafe {
                mem::transmute::<&mut $t<N>, &mut [N; $dimension * $dimension]>(self).iter_mut()
            }
        }
    }
  )
);

macro_rules! one_impl(
  ($t: ident, $($valueN: expr),+ ) => (
    impl<N: Copy + BaseNum> One for $t<N> {
        #[inline]
        fn one() -> $t<N> {
            $t::new($($valueN() ),+)
        }
    }
  )
);

macro_rules! zero_impl(
  ($t: ident, $($compN: ident),+ ) => (
    impl<N: Zero> Zero for $t<N> {
        #[inline]
        fn zero() -> $t<N> {
            $t {
                $($compN: ::zero() ),+
            }
        }

        #[inline]
        fn is_zero(&self) -> bool {
            $(::is_zero(&self.$compN) )&&+
        }
    }
  )
);

macro_rules! dim_impl(
  ($t: ident, $dimension: expr) => (
    impl<N> Dimension for $t<N> {
        #[inline]
        fn dimension(_: Option<$t<N>>) -> usize {
            $dimension
        }
    }
  )
);

macro_rules! indexable_impl(
  ($t: ident, $dimension: expr) => (
    impl<N> Shape<(usize, usize)> for $t<N> {
        #[inline]
        fn shape(&self) -> (usize, usize) {
            ($dimension, $dimension)
        }
    }

    impl<N: Copy> Indexable<(usize, usize), N> for $t<N> {
        #[inline]
        fn swap(&mut self, (i1, j1): (usize, usize), (i2, j2): (usize, usize)) {
            unsafe {
              mem::transmute::<&mut $t<N>, &mut [N; $dimension * $dimension]>(self)
                .swap(i1 + j1 * $dimension, i2 + j2 * $dimension)
            }
        }

        #[inline]
        unsafe fn unsafe_at(&self, (i, j): (usize, usize)) -> N {
            (*mem::transmute::<&$t<N>, &[N; $dimension * $dimension]>(self).get_unchecked(i + j * $dimension))
        }

        #[inline]
        unsafe fn unsafe_set(&mut self, (i, j): (usize, usize), val: N) {
            (*mem::transmute::<&mut $t<N>, &mut [N; $dimension * $dimension]>(self).get_unchecked_mut(i + j * $dimension)) = val
        }
    }
  )
);

macro_rules! index_impl(
    ($t: ident, $dimension: expr) => (
        impl<N> Index<(usize, usize)> for $t<N> {
            type Output = N;

            fn index(&self, (i, j): (usize, usize)) -> &N {
                unsafe {
                    &mem::transmute::<&$t<N>, & [N; $dimension * $dimension]>(self)[i + j * $dimension]
                }
            }
        }

        impl<N> IndexMut<(usize, usize)> for $t<N> {
            fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut N {
                unsafe {
                    &mut mem::transmute::<&mut $t<N>, &mut [N; $dimension * $dimension]>(self)[i + j * $dimension]
                }
            }
        }
    )
);

macro_rules! column_slice_impl(
    ($t: ident, $tv: ident, $slice: ident, $dimension: expr) => (
        impl<N: Clone + Copy + Zero> ColumnSlice<$slice<N>> for $t<N> {
            fn column_slice(&self, cid: usize, rstart: usize, rend: usize) -> $slice<N> {
                let column = self.column(cid);

                $slice::from_slice(rend - rstart, &column.as_ref()[rstart .. rend])
            }
        }
    )
);

macro_rules! row_impl(
  ($t: ident, $tv: ident, $dimension: expr) => (
    impl<N: Copy + Zero> Row<$tv<N>> for $t<N> {
        #[inline]
        fn nrows(&self) -> usize {
            Dimension::dimension(None::<$t<N>>)
        }

        #[inline]
        fn set_row(&mut self, row: usize, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self[(row, i)] = *e;
            }
        }

        #[inline]
        fn row(&self, row: usize) -> $tv<N> {
            let mut res: $tv<N> = ::zero();

            for (i, e) in res.iter_mut().enumerate() {
                *e = self[(row, i)];
            }

            res
        }
    }
  )
);

macro_rules! row_slice_impl(
    ($t: ident, $tv: ident, $slice: ident, $dimension: expr) => (
        impl<N: Clone + Copy + Zero> RowSlice<$slice<N>> for $t<N> {
            fn row_slice(&self, rid: usize, cstart: usize, cend: usize) -> $slice<N> {
                let row = self.row(rid);

                $slice::from_slice(cend - cstart, &row.as_ref()[cstart .. cend])
            }
        }
    )
);

macro_rules! column_impl(
  ($t: ident, $tv: ident, $dimension: expr) => (
    impl<N: Copy + Zero> Column<$tv<N>> for $t<N> {
        #[inline]
        fn ncols(&self) -> usize {
            Dimension::dimension(None::<$t<N>>)
        }

        #[inline]
        fn set_column(&mut self, column: usize, v: $tv<N>) {
            for (i, e) in v.iter().enumerate() {
                self[(i, column)] = *e;
            }
        }

        #[inline]
        fn column(&self, column: usize) -> $tv<N> {
            let mut res: $tv<N> = ::zero();

            for (i, e) in res.iter_mut().enumerate() {
                *e = self[(i, column)];
            }

            res
        }
    }
  )
);

macro_rules! diag_impl(
    ($t: ident, $tv: ident, $dimension: expr) => (
        impl<N: Copy + Zero> Diagonal<$tv<N>> for $t<N> {
            #[inline]
            fn from_diagonal(diagonal: &$tv<N>) -> $t<N> {
                let mut res: $t<N> = ::zero();

                res.set_diagonal(diagonal);

                res
            }

            #[inline]
            fn diagonal(&self) -> $tv<N> {
                let mut diagonal: $tv<N> = ::zero();

                for i in 0 .. $dimension {
                    unsafe { diagonal.unsafe_set(i, self.unsafe_at((i, i))) }
                }

                diagonal
            }
        }

        impl<N: Copy + Zero> DiagMut<$tv<N>> for $t<N> {
            #[inline]
            fn set_diagonal(&mut self, diagonal: &$tv<N>) {
                for i in 0 .. $dimension {
                    unsafe { self.unsafe_set((i, i), diagonal.unsafe_at(i)) }
                }
            }
        }
    )
);

macro_rules! mat_mul_mat_impl(
  ($t: ident, $dimension: expr) => (
    impl<N: Copy + BaseNum> Mul<$t<N>> for $t<N> {
        type Output = $t<N>;
        #[inline]
        fn mul(self, right: $t<N>) -> $t<N> {
            let mut res: $t<N> = ::zero();

            for i in 0 .. $dimension {
                for j in 0 .. $dimension {
                    let mut acc: N = ::zero();

                    unsafe {
                        for k in 0 .. $dimension {
                            acc = acc + self.at_fast((i, k)) * right.at_fast((k, j));
                        }

                        res.set_fast((i, j), acc);
                    }
                }
            }

            res
        }
    }

    impl<N: Copy + BaseNum> MulAssign<$t<N>> for $t<N> {
        #[inline]
        fn mul_assign(&mut self, right: $t<N>) {
            // NOTE: there is probably not any useful optimization to perform here compaired to the
            // version without assignment..
            *self = *self * right
        }
    }
  )
);

macro_rules! vec_mul_mat_impl(
  ($t: ident, $v: ident, $dimension: expr, $zero: expr) => (
    impl<N: Copy + BaseNum> Mul<$t<N>> for $v<N> {
        type Output = $v<N>;

        #[inline]
        fn mul(self, right: $t<N>) -> $v<N> {
            let mut res : $v<N> = $zero();

            for i in 0..$dimension {
                for j in 0..$dimension {
                    unsafe {
                        let val = res.at_fast(i) + self.at_fast(j) * right.at_fast((j, i));
                        res.set_fast(i, val)
                    }
                }
            }

            res
        }
    }

    impl<N: Copy + BaseNum> MulAssign<$t<N>> for $v<N> {
        #[inline]
        fn mul_assign(&mut self, right: $t<N>) {
            // NOTE: there is probably not any useful optimization to perform here compaired to the
            // version without assignment..
            *self = *self * right
        }
    }
  )
);

macro_rules! mat_mul_vec_impl(
  ($t: ident, $v: ident, $dimension: expr, $zero: expr) => (
    impl<N: Copy + BaseNum> Mul<$v<N>> for $t<N> {
        type Output = $v<N>;

        #[inline]
        fn mul(self, right: $v<N>) -> $v<N> {
            let mut res : $v<N> = $zero();

            for i in 0 .. $dimension {
                for j in 0 .. $dimension {
                    unsafe {
                        let val = res.at_fast(i) + self.at_fast((i, j)) * right.at_fast(j);
                        res.set_fast(i, val)
                    }
                }
            }

            res
        }
    }
  )
);

macro_rules! point_mul_mat_impl(
  ($t: ident, $v: ident, $dimension: expr, $zero: expr) => (
      vec_mul_mat_impl!($t, $v, $dimension, $zero);
  )
);

macro_rules! mat_mul_point_impl(
  ($t: ident, $v: ident, $dimension: expr, $zero: expr) => (
      mat_mul_vec_impl!($t, $v, $dimension, $zero);
  )
);

macro_rules! inverse_impl(
  ($t: ident, $dimension: expr) => (
    impl<N: Copy + BaseNum>
    Inverse for $t<N> {
        #[inline]
        fn inverse(&self) -> Option<$t<N>> {
            let mut res : $t<N> = *self;
            if res.inverse_mut() {
                Some(res)
            }
            else {
                None
            }
        }

        fn inverse_mut(&mut self) -> bool {
            let mut res: $t<N> = ::one();

            // inversion using Gauss-Jordan elimination
            for k in 0..$dimension {
                // search a non-zero value on the k-th column
                // FIXME: would it be worth it to spend some more time searching for the
                // max instead?

                let mut n0 = k; // index of a non-zero entry

                while n0 != $dimension {
                    if self[(n0, k)] != ::zero() {
                        break;
                    }

                    n0 = n0 + 1;
                }

                if n0 == $dimension {
                    return false
                }

                // swap pivot line
                if n0 != k {
                    for j in 0..$dimension {
                        self.swap((n0, j), (k, j));
                        res.swap((n0, j), (k, j));
                    }
                }

                let pivot = self[(k, k)];

                for j in k..$dimension {
                    let selfval = self[(k, j)] / pivot;
                    self[(k, j)] = selfval;
                }

                for j in 0..$dimension {
                    let resval = res[(k, j)] / pivot;
                    res[(k, j)] = resval;
                }

                for l in 0..$dimension {
                    if l != k {
                        let normalizer = self[(l, k)];

                        for j in k..$dimension {
                            let selfval = self[(l, j)] - self[(k, j)] * normalizer;
                            self[(l, j)] = selfval;
                        }

                        for j in 0..$dimension {
                            let resval  = res[(l, j)] - res[(k, j)] * normalizer;
                            res[(l, j)] = resval;
                        }
                    }
                }
            }

            *self = res;

            true
        }
    }
  )
);

macro_rules! transpose_impl(
  ($t: ident, $dimension: expr) => (
    impl<N: Copy> Transpose for $t<N> {
        #[inline]
        fn transpose(&self) -> $t<N> {
            let mut res = *self;

            res.transpose_mut();
            res
        }

        #[inline]
        fn transpose_mut(&mut self) {
            for i in 1 .. $dimension {
                for j in 0 .. i {
                    self.swap((i, j), (j, i))
                }
            }
        }
    }
  )
);

macro_rules! approx_eq_impl(
  ($t: ident) => (
    impl<N: ApproxEq<N>> ApproxEq<N> for $t<N> {
        #[inline]
        fn approx_epsilon(_: Option<$t<N>>) -> N {
            ApproxEq::approx_epsilon(None::<N>)
        }

        #[inline]
        fn approx_ulps(_: Option<$t<N>>) -> u32 {
            ApproxEq::approx_ulps(None::<N>)
        }

        #[inline]
        fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
            let mut zip = self.iter().zip(other.iter());
            zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
        }

        #[inline]
        fn approx_eq_ulps(&self, other: &$t<N>, ulps: u32) -> bool {
            let mut zip = self.iter().zip(other.iter());
            zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
        }
    }
  )
);

macro_rules! to_homogeneous_impl(
  ($t: ident, $t2: ident, $dimension: expr, $dim2: expr) => (
    impl<N: BaseNum + Copy> ToHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn to_homogeneous(&self) -> $t2<N> {
            let mut res: $t2<N> = ::one();

            for i in 0..$dimension {
                for j in 0..$dimension {
                    res[(i, j)] = self[(i, j)]
                }
            }

            res
        }
    }
  )
);

macro_rules! from_homogeneous_impl(
  ($t: ident, $t2: ident, $dimension: expr, $dim2: expr) => (
    impl<N: BaseNum + Copy> FromHomogeneous<$t2<N>> for $t<N> {
        #[inline]
        fn from(m: &$t2<N>) -> $t<N> {
            let mut res: $t<N> = ::one();

            for i in 0..$dimension {
                for j in 0..$dimension {
                    res[(i, j)] = m[(i, j)]
                }
            }

            // FIXME: do we have to deal the lost components
            // (like if the 1 is not a 1… do we have to divide?)

            res
        }
    }
  )
);

macro_rules! outer_impl(
    ($t: ident, $m: ident) => (
        impl<N: Copy + Mul<N, Output = N> + Zero> Outer for $t<N> {
            type OuterProductType = $m<N>;

            #[inline]
            fn outer(&self, other: &$t<N>) -> $m<N> {
                let mut res: $m<N> = ::zero();

                for i in 0 .. ::dimension::<$t<N>>() {
                    for j in 0 .. ::dimension::<$t<N>>() {
                        res[(i, j)] = self[i] * other[j]
                    }
                }
                res
            }
        }
    )
);

macro_rules! eigen_qr_impl(
    ($t: ident, $v: ident) => (
        impl<N> EigenQR<N, $v<N>> for $t<N>
            where N: BaseFloat + ApproxEq<N> + Clone {
            fn eigen_qr(&self, eps: &N, niter: usize) -> ($t<N>, $v<N>) {
                linalg::eigen_qr(self, eps, niter)
            }
        }
    )
);


macro_rules! mean_impl(
    ($t: ident, $v: ident, $dimension: expr) => (
        impl<N: BaseNum + Cast<f64> + Clone> Mean<$v<N>> for $t<N> {
            fn mean(&self) -> $v<N> {
                let mut res: $v<N> = ::zero();
                let normalizer: N  = Cast::from(1.0f64 / $dimension as f64);
        
                for i in 0 .. $dimension {
                    for j in 0 .. $dimension {
                        unsafe {
                            let acc = res.unsafe_at(j) + self.unsafe_at((i, j)) * normalizer;
                            res.unsafe_set(j, acc);
                        }
                    }
                }
        
                res
            }
        }
    )
);

macro_rules! mat_display_impl(
    ($t: ident, $dimension: expr) => (
        impl<N: fmt::Display + BaseFloat> fmt::Display for $t<N> {
            // XXX: will will not always work correctly due to rounding errors.
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fn integral_length<N: BaseFloat>(val: &N) -> usize {
                    let mut res = 1;
                    let mut curr: N = ::cast(10.0f64);

                    while curr <= *val {
                        curr = curr * ::cast(10.0f64);
                        res = res + 1;
                    }

                    if val.is_sign_negative() {
                        res + 1
                    }
                    else {
                        res
                    }
                }

                let mut max_decimal_length = 0;
                let mut decimal_lengths: $t<usize> = ::zero();
                for i in 0 .. $dimension {
                    for j in 0 .. $dimension {
                        decimal_lengths[(i, j)] = integral_length(&self[(i, j)].clone());
                        max_decimal_length = ::max(max_decimal_length, decimal_lengths[(i, j)]);
                    }
                }

                let precision = f.precision().unwrap_or(3);
                let max_number_length = max_decimal_length + precision + 1;

                try!(writeln!(f, "  ┌ {:>width$} ┐", "", width = max_number_length * $dimension + $dimension - 1));

                for i in 0 .. $dimension {
                    try!(write!(f, "  │"));
                    for j in 0 .. $dimension {
                        let number_length = decimal_lengths[(i, j)] + precision + 1;
                        let pad = max_number_length - number_length;
                        try!(write!(f, " {:>thepad$}", "", thepad = pad));
                        try!(write!(f, "{:.*}", precision, (*self)[(i, j)]));
                    }
                    try!(writeln!(f, " │"));
                }

                writeln!(f, "  └ {:>width$} ┘", "", width = max_number_length * $dimension + $dimension - 1)
            }
        }
    )
);
