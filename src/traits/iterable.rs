use std::vec;

pub trait Iterable<N>
{
  fn iter<'l>(&'l self) -> vec::VecIterator<'l, N>;
}

pub trait IterableMut<N>
{
  fn mut_iter<'l>(&'l mut self) -> vec::VecMutIterator<'l, N>;
}

/*
 * FIXME: the prevous traits are only workarounds.
 * It should be something like:

pub trait Iterable<'self, N, I: Iterator<N>>
{
  fn iter(&'self self) -> I;
}

pub trait IterableMut<'self, N, I: Iterator<N>>
{
  fn mut_iter(&'self self) -> I;
}

 * but this gives an ICE =(
 * For now, we oblige the iterator to be one specific type which works with
 * everything on this lib.
 */

pub trait FromAnyIterator<N>
{
  fn from_iterator<'l>(&mut vec::VecIterator<'l, N>)        -> Self;
  fn from_mut_iterator<'l>(&mut vec::VecMutIterator<'l, N>) -> Self;
}

/*
 * FIXME: the previous trait is only a workaround.
 * It should be something like:
pub trait FromAnyIterator<N>
{
  fn from_iterator<I: Iterator<N>>(&mut I) -> Self;
}

 * but this gives a wierd error message (the compile seems to mistake N with
 * I…).
 * For now, we oblige the iterator to be one specific type which works with
 * everything on this lib.
 *
 * Note that we dont use the standard std::iterator::FromIterator<N, I: Iterator<N>>
 * because it is too hard to work with on generic code (as a type bound)
 * because we have to name explicitly the type of the iterator.
 *
 */
