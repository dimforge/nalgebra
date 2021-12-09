//! Module for utility functions used in format conversions.

pub(crate) struct CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    offset: usize,
    count_iter: I,
}

impl<I> CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    pub(crate) fn new<T: IntoIterator<IntoIter = I, Item = usize>>(counts: T) -> Self {
        CountToOffsetIter {
            offset: 0,
            count_iter: counts.into_iter(),
        }
    }
}

impl<I> Iterator for CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.count_iter.next()?;

        let current_offset = self.offset;
        self.offset += next;

        Some(current_offset)
    }
}
