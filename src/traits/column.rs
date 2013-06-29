pub trait Column<C>
{
  fn set_column(&mut self, uint, C);
  fn column(&self, uint) -> C;
}
