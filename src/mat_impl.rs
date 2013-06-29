#[macro_escape];

macro_rules! mat_impl(
  ($t: ident, $dim: expr) => (
    impl<N> $t<N>
    {
      #[inline]
      pub fn new(mij: [N, ..$dim * $dim]) -> $t<N>
      { $t { mij: mij } }
    }
  )
)

macro_rules! one_impl(
  ($t: ident, [ $($value: ident)|+ ] ) => (
    impl<N: Copy + One + Zero> One for $t<N>
    {
      #[inline]
      fn one() -> $t<N>
      {
        let (_0, _1) = (Zero::zero::<N>(), One::one::<N>());
        return $t::new( [ $( copy $value, )+ ] )
      }
    }
  )
)

macro_rules! zero_impl(
  ($t: ident, [ $($value: ident)|+ ] ) => (
    impl<N: Copy + Zero> Zero for $t<N>
    {
      #[inline]
      fn zero() -> $t<N>
      {
        let _0 = Zero::zero();
        return $t::new( [ $( copy $value, )+ ] )
      }

     #[inline]
     fn is_zero(&self) -> bool
     { self.mij.iter().all(|e| e.is_zero()) }
    }
  )
)

macro_rules! dim_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Dim for $t<N>
    {
      #[inline]
      fn dim() -> uint
      { $dim }
    }
  )
)

macro_rules! mat_indexing_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy> $t<N>
    {
      #[inline]
      pub fn offset(&self, i: uint, j: uint) -> uint
      { i * $dim + j }
    
      #[inline]
      pub fn set(&mut self, i: uint, j: uint, t: &N)
      {
        self.mij[self.offset(i, j)] = copy *t
      }
    
      #[inline]
      pub fn at(&self, i: uint, j: uint) -> N
      {
        copy self.mij[self.offset(i, j)]
      }
    }
  )
)

macro_rules! mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    Mul<$t<N>, $t<N>> for $t<N>
    {
      fn mul(&self, other: &$t<N>) -> $t<N>
      {
        let mut res: $t<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          {
            let mut acc = Zero::zero::<N>();
    
            for iterate(0u, $dim) |k|
            { acc = acc + self.at(i, k) * other.at(k, j); }
    
            res.set(i, j, &acc);
          }
        }
    
        res
      }
    }
  )
)

macro_rules! rmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    RMul<$v<N>> for $t<N>
    {
      fn rmul(&self, other: &$v<N>) -> $v<N>
      {
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          { res.at[i] = res.at[i] + other.at[j] * self.at(i, j); }
        }
    
        res
      }
    }
  )
)

macro_rules! lmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    LMul<$v<N>> for $t<N>
    {
      fn lmul(&self, other: &$v<N>) -> $v<N>
      {
    
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          { res.at[i] = res.at[i] + other.at[j] * self.at(j, i); }
        }
    
        res
      }
    }
  )
)

macro_rules! transform_impl(
  ($t: ident, $v: ident) => (
    impl<N: Copy + DivisionRing + Eq>
    Transform<$v<N>> for $t<N>
    {
      #[inline]
      fn transform_vec(&self, v: &$v<N>) -> $v<N>
      { self.rmul(v) }
    
      #[inline]
      fn inv_transform(&self, v: &$v<N>) -> $v<N>
      { self.inverse().transform_vec(v) }
    }
  )
)

macro_rules! inv_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + Eq + DivisionRing>
    Inv for $t<N>
    {
      #[inline]
      fn inverse(&self) -> $t<N>
      {
        let mut res : $t<N> = copy *self;
    
        res.invert();
    
        res
      }
    
      fn invert(&mut self)
      {
        let mut res: $t<N> = One::one();
        let     _0N: N     = Zero::zero();
    
        // inversion using Gauss-Jordan elimination
        for iterate(0u, $dim) |k|
        {
          // search a non-zero value on the k-th column
          // FIXME: would it be worth it to spend some more time searching for the
          // max instead?
    
          let mut n0 = k; // index of a non-zero entry
    
          while (n0 != $dim)
          {
            if self.at(n0, k) != _0N
            { break; }
    
            n0 = n0 + 1;
          }
    
          // swap pivot line
          if n0 != k
          {
            for iterate(0u, $dim) |j|
            {
              let off_n0_j = self.offset(n0, j);
              let off_k_j  = self.offset(k, j);
    
              swap(self.mij, off_n0_j, off_k_j);
              swap(res.mij,  off_n0_j, off_k_j);
            }
          }
    
          let pivot = self.at(k, k);
    
          for iterate(k, $dim) |j|
          {
            let selfval = &(self.at(k, j) / pivot);
            self.set(k, j, selfval);
          }
    
          for iterate(0u, $dim) |j|
          {
            let resval  = &(res.at(k, j)   / pivot);
            res.set(k, j, resval);
          }
    
          for iterate(0u, $dim) |l|
          {
            if l != k
            {
              let normalizer = self.at(l, k);
    
              for iterate(k, $dim) |j|
              {
                let selfval = &(self.at(l, j) - self.at(k, j) * normalizer);
                self.set(l, j, selfval);
              }
    
              for iterate(0u, $dim) |j|
              {
                let resval  = &(res.at(l, j) - res.at(k, j) * normalizer);
                res.set(l, j, resval);
              }
            }
          }
        }
    
        *self = res;
      }
    }
  )
)

macro_rules! transpose_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy> Transpose for $t<N>
    {
      #[inline]
      fn transposed(&self) -> $t<N>
      {
        let mut res = copy *self;
    
        res.transpose();
    
        res
      }
    
      fn transpose(&mut self)
      {
        for iterate(1u, $dim) |i|
        {
          for iterate(0u, $dim - 1) |j|
          {
            let off_i_j = self.offset(i, j);
            let off_j_i = self.offset(j, i);
    
            swap(self.mij, off_i_j, off_j_i);
          }
        }
      }
    }
  )
)

macro_rules! approx_eq_impl(
  ($t: ident) => (
    impl<N: ApproxEq<N>> ApproxEq<N> for $t<N>
    {
      #[inline]
      fn approx_epsilon() -> N
      { ApproxEq::approx_epsilon::<N, N>() }
    
      #[inline]
      fn approx_eq(&self, other: &$t<N>) -> bool
      {
        let mut zip = self.mij.iter().zip(other.mij.iter());
    
        do zip.all |(a, b)| { a.approx_eq(b) }
      }
    
      #[inline]
      fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool
      {
        let mut zip = self.mij.iter().zip(other.mij.iter());
    
        do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
      }
    }
  )
)

macro_rules! rand_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Rand> Rand for $t<N>
    {
      #[inline]
      fn rand<R: Rng>($param: &mut R) -> $t<N>
      { $t::new([ $( $elem.gen(), )+ ]) }
    }
  )
)
