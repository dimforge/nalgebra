#[macro_escape];

macro_rules! clone_impl(
  ($t:ident) => (
    // FIXME: use 'Clone' alone. For the moment, we need 'Copy' because the automatic
    // implementation of Clone for [t, ..n] is badly typed.
    impl<N: Clone + Copy> Clone for $t<N>
    {
      fn clone(&self) -> $t<N>
      {
        $t { at: copy self.at }
      }
    }
  )
)

macro_rules! new_impl(
  ($t: ident, $dim: expr) => (
    impl<N> $t<N>
    {
      #[inline]
      pub fn new(at: [N, ..$dim]) -> $t<N>
      { $t { at: at } }
    }
  )
)

macro_rules! indexable_impl(
  ($t: ident) => (
    impl<N: Clone> Indexable<uint, N> for $t<N>
    {
      #[inline]
      pub fn at(&self, i: uint) -> N
      { self.at[i].clone() }

      #[inline]
      pub fn set(&mut self, i: uint, val: N)
      { self.at[i] = val }
    }
  )
)

macro_rules! new_repeat_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Clone> $t<N>
    {
      #[inline]
      pub fn new_repeat($param: N) -> $t<N>
      { $t{ at: [ $( $elem.clone(), )+ ] } }
    }
  )
)

macro_rules! iterable_impl(
  ($t: ident) => (
    impl<N> Iterable<N> for $t<N>
    {
      fn iter<'l>(&'l self) -> VecIterator<'l, N>
      { self.at.iter() }
    }
  )
)

macro_rules! iterable_mut_impl(
  ($t: ident) => (
    impl<N> IterableMut<N> for $t<N>
    {
      fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N>
      { self.at.mut_iter() }
    }
  )
)

macro_rules! eq_impl(
  ($t: ident) => (
    impl<N: Eq> Eq for $t<N>
    {
      #[inline]
      fn eq(&self, other: &$t<N>) -> bool
      { self.at.iter().zip(other.at.iter()).all(|(a, b)| a == b) }
    
      #[inline]
      fn ne(&self, other: &$t<N>) -> bool
      { self.at.iter().zip(other.at.iter()).all(|(a, b)| a != b) }
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

// FIXME: add the possibility to specialize that
macro_rules! basis_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Copy + DivisionRing + Algebraic + ApproxEq<N>> Basis for $t<N>
    {
      pub fn canonical_basis(f: &fn($t<N>))
      {
        for iterate(0u, $dim) |i|
        {
          let mut basis_element : $t<N> = Zero::zero();
    
          basis_element.at[i] = One::one();
    
          f(basis_element);
        }
      }
    
      pub fn orthonormal_subspace_basis(&self, f: &fn($t<N>))
      {
        // compute the basis of the orthogonal subspace using Gram-Schmidt
        // orthogonalization algorithm
        let mut basis: ~[$t<N>] = ~[];
    
        for iterate(0u, $dim) |i|
        {
          let mut basis_element : $t<N> = Zero::zero();
    
          basis_element.at[i] = One::one();
    
          if basis.len() == $dim - 1
          { break; }
    
          let mut elt = basis_element.clone();
    
          elt = elt - self.scalar_mul(&basis_element.dot(self));
    
          for basis.iter().advance |v|
          { elt = elt - v.scalar_mul(&elt.dot(v)) };
    
          if !elt.sqnorm().approx_eq(&Zero::zero())
          {
            let new_element = elt.normalized();

            f(new_element.clone());

            basis.push(new_element);
          }
        }
      }
    }
  )
)

macro_rules! add_impl(
  ($t: ident) => (
    impl<N: Clone + Add<N,N>> Add<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn add(&self, other: &$t<N>) -> $t<N>
      {
        self.at.iter()
               .zip(other.at.iter())
               .transform(|(a, b)| { *a + *b })
               .collect()
      }
    }
  )
)

macro_rules! sub_impl(
  ($t: ident) => (
    impl<N: Clone + Sub<N,N>> Sub<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn sub(&self, other: &$t<N>) -> $t<N>
      {
        self.at.iter()
               .zip(other.at.iter())
               .transform(| (a, b) | { *a - *b })
               .collect()
      }
    }
  )
)

macro_rules! neg_impl(
  ($t: ident) => (
    impl<N: Neg<N>> Neg<$t<N>> for $t<N>
    {
      #[inline]
      fn neg(&self) -> $t<N>
      { self.at.iter().transform(|a| -a).collect() }
    }
  )
)

macro_rules! dot_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Ring> Dot<N> for $t<N>
    {
      #[inline]
      fn dot(&self, other: &$t<N>) -> N
      {
        let mut res = Zero::zero::<N>();
    
        for iterate(0u, $dim) |i|
        { res = res + self.at[i] * other.at[i]; }
    
        res
      } 
    }
  )
)

macro_rules! sub_dot_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Ring> SubDot<N> for $t<N>
    {
      #[inline]
      fn sub_dot(&self, a: &$t<N>, b: &$t<N>) -> N
      {
        let mut res = Zero::zero::<N>();
    
        for iterate(0u, $dim) |i|
        { res = res + (self.at[i] - a.at[i]) * b.at[i]; }
    
        res
      } 
    }
  )
)

macro_rules! scalar_mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Mul<N, N>> ScalarMul<N> for $t<N>
    {
      #[inline]
      fn scalar_mul(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a * *s).collect() }
    
      #[inline]
      fn scalar_mul_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] * *s; }
      }
    }
  )
)


macro_rules! scalar_div_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Div<N, N>> ScalarDiv<N> for $t<N>
    {
      #[inline]
      fn scalar_div(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a / *s).collect() }
    
      #[inline]
      fn scalar_div_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] / *s; }
      }
    }
  )
)

macro_rules! scalar_add_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Add<N, N>> ScalarAdd<N> for $t<N>
    {
      #[inline]
      fn scalar_add(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a + *s).collect() }
    
      #[inline]
      fn scalar_add_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] + *s; }
      }
    }
  )
)

macro_rules! scalar_sub_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Sub<N, N>> ScalarSub<N> for $t<N>
    {
      #[inline]
      fn scalar_sub(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a - *s).collect() }
    
      #[inline]
      fn scalar_sub_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] - *s; }
      }
    }
  )
)

macro_rules! translation_impl(
  ($t: ident) => (
    impl<N: Clone + Copy + Add<N, N> + Neg<N>> Translation<$t<N>> for $t<N>
    {
      #[inline]
      fn translation(&self) -> $t<N>
      { self.clone() }

      #[inline]
      fn inv_translation(&self) -> $t<N>
      { -self }
    
      #[inline]
      fn translate_by(&mut self, t: &$t<N>)
      { *self = *self + *t; }
    }
  )
)

macro_rules! translatable_impl(
  ($t: ident) => (
    impl<N: Add<N, N> + Neg<N> + Clone> Translatable<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn translated(&self, t: &$t<N>) -> $t<N>
      { self + *t }
    }
  )
)

macro_rules! norm_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Clone + Copy + DivisionRing + Algebraic> Norm<N> for $t<N>
    {
      #[inline]
      fn sqnorm(&self) -> N
      { self.dot(self) }
    
      #[inline]
      fn norm(&self) -> N
      { self.sqnorm().sqrt() }
    
      #[inline]
      fn normalized(&self) -> $t<N>
      {
        let mut res : $t<N> = self.clone();
    
        res.normalize();
    
        res
      }
    
      #[inline]
      fn normalize(&mut self) -> N
      {
        let l = self.norm();
    
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] / l; }
    
        l
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
        let mut zip = self.at.iter().zip(other.at.iter());
    
        do zip.all |(a, b)| { a.approx_eq(b) }
      }
    
      #[inline]
      fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool
      {
        let mut zip = self.at.iter().zip(other.at.iter());
    
        do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
      }
    }
  )
)

macro_rules! zero_impl(
  ($t: ident) => (
    impl<N: Clone + Zero> Zero for $t<N>
    {
      #[inline]
      fn zero() -> $t<N>
      { $t::new_repeat(Zero::zero()) }
    
      #[inline]
      fn is_zero(&self) -> bool
      { self.at.iter().all(|e| e.is_zero()) }
    }
  )
)

macro_rules! one_impl(
  ($t: ident) => (
    impl<N: Clone + One> One for $t<N>
    {
      #[inline]
      fn one() -> $t<N>
      { $t::new_repeat(One::one()) }
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

macro_rules! from_any_iterator_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Clone> FromAnyIterator<N> for $t<N>
    {
      fn from_iterator<'l>($param: &mut VecIterator<'l, N>) -> $t<N>
      { $t { at: [ $( $elem.next().unwrap().clone(), )+ ] } }

      fn from_mut_iterator<'l>($param: &mut VecMutIterator<'l, N>) -> $t<N>
      { $t { at: [ $( $elem.next().unwrap().clone(), )+ ] } }
    }
  )
)

macro_rules! from_iterator_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N, Iter: Iterator<N>> FromIterator<N, Iter> for $t<N>
    {
      fn from_iterator($param: &mut Iter) -> $t<N>
      { $t { at: [ $( $elem.next().unwrap(), )+ ] } }
    }
  )
)

macro_rules! bounded_impl(
  ($t: ident) => (
    impl<N: Bounded + Clone> Bounded for $t<N>
    {
      #[inline]
      fn max_value() -> $t<N>
      { $t::new_repeat(Bounded::max_value()) }
    
      #[inline]
      fn min_value() -> $t<N>
      { $t::new_repeat(Bounded::min_value()) }
    }
  )
)

macro_rules! to_homogeneous_impl(
  ($t: ident, $t2: ident) =>
  {
    impl<N: Clone + One> ToHomogeneous<$t2<N>> for $t<N>
    {
      fn to_homogeneous(&self) -> $t2<N>
      {
        let mut res: $t2<N> = One::one();

        for self.iter().zip(res.mut_iter()).advance |(in, out)|
        { *out = in.clone() }

        res
      }
    }
  }
)

macro_rules! from_homogeneous_impl(
  ($t: ident, $t2: ident, $dim2: expr) =>
  {
    impl<N: Clone + Div<N, N> + One + Zero> FromHomogeneous<$t2<N>> for $t<N>
    {
      fn from_homogeneous(v: &$t2<N>) -> $t<N>
      {
        let mut res: $t<N> = Zero::zero();

        for v.iter().zip(res.mut_iter()).advance |(in, out)|
        { *out = in.clone() }

        res.scalar_div(&v.at[$dim2 - 1]);

        res
      }
    }
  }
)
