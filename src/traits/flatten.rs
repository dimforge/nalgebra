/// Trait of objects which can be written as a list of values, and read from
/// that list.
pub trait Flatten<N>
{
  /// The number of elements needed to flatten this object.
  fn flat_size() -> uint;

  /**
   * Creates a new object from its flattened version. Its flattened version is
   * a continuous list of values. It is assumet that `flat_size` elements will
   * be read.
   *
   *   - `l`: list from which the flat version must be read.
   *   - `off`: index from which (included) the flat version read must begin.
   *            It is assumed that the caller gives a valid input.
   */
  fn from_flattened(l: &[N], off: uint) -> Self; // FIXME: keep (vector + index) or use an iterator?

  /**
   * Creates a flattened version of `self`. The result vector must be of size
   * `flat_size`.
   */
  fn flatten(&self) -> ~[N];

  /**
   * Creates a flattened version of `self` on a vector. It is assumed that
   * `flat_size` elements will be written contiguously.
   *
   *   - `l`: list to which the flat version must be written.
   *   - `off`: index from which (included) the flat version write must begin.
   *            It is assumed that the caller allocated a long enough list.
   */
  fn flatten_to(&self, l: &mut [N], off: uint); // FIXME: keep (vector + index) or use an iterator (to a mutable value…)?
}
