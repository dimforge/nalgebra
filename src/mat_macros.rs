#[macro_escape];

macro_rules! mat_impl(
  ($t: ident, $comp0: ident $(,$compN: ident)*) => (
    impl<N> $t<N>
    {
      #[inline]
      pub fn new($comp0: N $(, $compN: N )*) -> $t<N>
      {
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
    impl<Nin: NumCast + Clone, Nout: NumCast> MatCast<$t<Nout>> for $t<Nin>
    {
      #[inline]
      pub fn from(m: $t<Nin>) -> $t<Nout>
      { $t::new(NumCast::from(m.$comp0.clone()) $(, NumCast::from(m.$compN.clone()) )*) }
    }
  )
)

macro_rules! iterable_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Iterable<N> for $t<N>
    {
      fn iter<'l>(&'l self) -> VecIterator<'l, N>
      { unsafe { cast::transmute::<&'l $t<N>, &'l [N, ..$dim * $dim]>(self).iter() } }
    }
  )
)

macro_rules! iterable_mut_impl(
  ($t: ident, $dim: expr) => (
    impl<N> IterableMut<N> for $t<N>
    {
      fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N>
      { unsafe { cast::transmute::<&'l mut $t<N>, &'l mut [N, ..$dim * $dim]>(self).mut_iter() } }
    }
  )
)

macro_rules! one_impl(
  ($t: ident, $value0: ident $(, $valueN: ident)* ) => (
    impl<N: Clone + One + Zero> One for $t<N>
    {
      #[inline]
      fn one() -> $t<N>
      {
        let (_0, _1) = (Zero::zero::<N>(), One::one::<N>());
        return $t::new($value0.clone() $(, $valueN.clone() )*)
      }
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

macro_rules! indexable_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone> Indexable<(uint, uint), N> for $t<N>
    {
      #[inline]
      pub fn at(&self, (i, j): (uint, uint)) -> N
      { unsafe { cast::transmute::<&$t<N>, &[N, ..$dim * $dim]>(self)[i * $dim + j].clone() } }

      #[inline]
      pub fn set(&mut self, (i, j): (uint, uint), val: N)
      { unsafe { cast::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)[i * $dim + j] = val } }

      #[inline]
      pub fn swap(&mut self, (i1, j1): (uint, uint), (i2, j2): (uint, uint))
      {
        unsafe {
          cast::transmute::<&mut $t<N>, &mut [N, ..$dim * $dim]>(self)
            .swap(i1 * $dim + j1, i2 * $dim + j2)
        }
      }
    }
  )
)

macro_rules! column_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone, V: Zero + Iterable<N> + IterableMut<N>> Column<V> for $t<N>
    {
      fn set_column(&mut self, col: uint, v: V)
      {
        for v.iter().enumerate().advance |(i, e)|
        {
          if i == Dim::dim::<$t<N>>()
          { break }

          self.set((i, col), e.clone());
        }
      }

      fn column(&self, col: uint) -> V
      {
        let mut res = Zero::zero::<V>();

        for res.mut_iter().enumerate().advance |(i, e)|
        {
          if i >= Dim::dim::<$t<N>>()
          { break }

          *e = self.at((i, col));
        }

        res
      }
    }
  )
)

macro_rules! mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Ring>
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
            { acc = acc + self.at((i, k)) * other.at((k, j)); }
    
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
    impl<N: Clone + Ring>
    RMul<$v<N>> for $t<N>
    {
      fn rmul(&self, other: &$v<N>) -> $v<N>
      {
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          {
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
    impl<N: Clone + Ring>
    LMul<$v<N>> for $t<N>
    {
      fn lmul(&self, other: &$v<N>) -> $v<N>
      {
    
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          {
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
    impl<N: Clone + DivisionRing + Eq>
    Transform<$v<N>> for $t<N>
    {
      #[inline]
      fn transform_vec(&self, v: &$v<N>) -> $v<N>
      { self.rmul(v) }
    
      #[inline]
      fn inv_transform(&self, v: &$v<N>) -> $v<N>
      {
        match self.inverse()
        {
          Some(t) => t.transform_vec(v),
          None    => fail!("Cannot use inv_transform on a non-inversible matrix.")
        }
      }
    }
  )
)

macro_rules! inv_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Eq + DivisionRing>
    Inv for $t<N>
    {
      #[inline]
      fn inverse(&self) -> Option<$t<N>>
      {
        let mut res : $t<N> = self.clone();
    
        if res.inplace_inverse()
        { Some(res) }
        else
        { None }
      }
    
      fn inplace_inverse(&mut self) -> bool
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
            if self.at((n0, k)) != _0N
            { break; }
    
            n0 = n0 + 1;
          }

          if n0 == $dim
          { return false }
    
          // swap pivot line
          if n0 != k
          {
            for iterate(0u, $dim) |j|
            {
              self.swap((n0, j), (k, j));
              res.swap((n0, j), (k, j));
            }
          }
    
          let pivot = self.at((k, k));
    
          for iterate(k, $dim) |j|
          {
            let selfval = self.at((k, j)) / pivot;
            self.set((k, j), selfval);
          }
    
          for iterate(0u, $dim) |j|
          {
            let resval = res.at((k, j)) / pivot;
            res.set((k, j), resval);
          }
    
          for iterate(0u, $dim) |l|
          {
            if l != k
            {
              let normalizer = self.at((l, k));
    
              for iterate(k, $dim) |j|
              {
                let selfval = self.at((l, j)) - self.at((k, j)) * normalizer;
                self.set((l, j), selfval);
              }
    
              for iterate(0u, $dim) |j|
              {
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
    impl<N: Clone> Transpose for $t<N>
    {
      #[inline]
      fn transposed(&self) -> $t<N>
      {
        let mut res = self.clone();
    
        res.transpose();
    
        res
      }
    
      fn transpose(&mut self)
      {
        for iterate(1u, $dim) |i|
        {
          for iterate(0u, $dim - 1) |j|
          { self.swap((i, j), (j, i)) }
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
        let mut zip = self.iter().zip(other.iter());
    
        do zip.all |(a, b)| { a.approx_eq(b) }
      }
    
      #[inline]
      fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool
      {
        let mut zip = self.iter().zip(other.iter());
    
        do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
      }
    }
  )
)

macro_rules! to_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: One + Zero + Clone> ToHomogeneous<$t2<N>> for $t<N>
    {
      fn to_homogeneous(&self) -> $t2<N>
      {
        let mut res: $t2<N> = One::one();

        for iterate(0, $dim) |i|
        {
          for iterate(0, $dim) |j|
          { res.set((i, j), self.at((i, j))) }
        }

        res
      }
    }
  )
)

macro_rules! from_homogeneous_impl(
  ($t: ident, $t2: ident, $dim: expr, $dim2: expr) => (
    impl<N: One + Zero + Clone> FromHomogeneous<$t2<N>> for $t<N>
    {
      fn from_homogeneous(m: &$t2<N>) -> $t<N>
      {
        let mut res: $t<N> = One::one();

        for iterate(0, $dim2) |i|
        {
          for iterate(0, $dim2) |j|
          { res.set((i, j), m.at((i, j))) }
        }

        // FIXME: do we have to deal the lost components
        // (like if the 1 is not a 1… do we have to divide?)

        res
      }
    }
  )
)
