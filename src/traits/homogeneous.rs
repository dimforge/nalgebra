pub trait ToHomogeneous<U>
{
  fn to_homogeneous(&self) -> U;
}

pub trait FromHomogeneous<U>
{
  fn from_homogeneous(&U) -> Self;
}
