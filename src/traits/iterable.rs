use std::vec;

/// Traits of objects which can be iterated through like a vector.
pub trait Iterable<N> {
    /// Gets a vector-like read-only iterator.
    fn iter<'l>(&'l self) -> vec::VecIterator<'l, N>;
}

/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut<N> {
    /// Gets a vector-like read-write iterator.
    fn mut_iter<'l>(&'l mut self) -> vec::VecMutIterator<'l, N>;
}

/*
 * FIXME: the prevous traits are only workarounds.
 * It should be something like:

 pub trait Iterable<'self, N, I: Iterator<N>> {
 fn iter(&'self self) -> I;
 }

 pub trait IterableMut<'self, N, I: Iterator<N>> {
 fn mut_iter(&'self self) -> I;
 }

 * but this gives an ICE =(
 * For now, we oblige the iterator to be one specific type which works with
 * everything on this lib.
 */
