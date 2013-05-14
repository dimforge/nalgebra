pub trait Cross<Result>
{
  fn cross(&self, other : &Self) -> Result;
}
