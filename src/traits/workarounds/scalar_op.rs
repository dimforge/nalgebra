pub trait ScalarMul<T>
{
  fn scalar_mul(&self, &T) -> Self;
  fn scalar_mul_inplace(&mut self, &T);
}

pub trait ScalarDiv<T>
{
  fn scalar_div(&self, &T) -> Self;
  fn scalar_div_inplace(&mut self, &T);
}

pub trait ScalarAdd<T>
{
  fn scalar_add(&self, &T) -> Self;
  fn scalar_add_inplace(&mut self, &T);
}

pub trait ScalarSub<T>
{
  fn scalar_sub(&self, &T) -> Self;
  fn scalar_sub_inplace(&mut self, &T);
}
