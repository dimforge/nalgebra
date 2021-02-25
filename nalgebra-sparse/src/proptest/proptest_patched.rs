//! Contains a modified implementation of `proptest::strategy::Shuffle`.
//!
//! The current implementation in `proptest` does not generate all permutations, which is
//! problematic for our proptest generators. The issue has been fixed in
//! https://github.com/AltSysrq/proptest/pull/217
//! but it has yet to be merged and released. As soon as this fix makes it into a new release,
//! the modified code here can be removed.
//!
/*!
    This code has been copied and adapted from
    https://github.com/AltSysrq/proptest/blob/master/proptest/src/strategy/shuffle.rs
    The original licensing text is:

    //-
    // Copyright 2017 Jason Lingle
    //
    // Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
    // http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
    // <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
    // option. This file may not be copied, modified, or distributed
    // except according to those terms.

*/

use proptest::num;
use proptest::prelude::Rng;
use proptest::strategy::{NewTree, Shuffleable, Strategy, ValueTree};
use proptest::test_runner::{TestRng, TestRunner};
use std::cell::Cell;

#[derive(Clone, Debug)]
#[must_use = "strategies do nothing unless used"]
pub struct Shuffle<S>(pub(super) S);

impl<S: Strategy> Strategy for Shuffle<S>
where
    S::Value: Shuffleable,
{
    type Tree = ShuffleValueTree<S::Tree>;
    type Value = S::Value;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let rng = runner.new_rng();

        self.0.new_tree(runner).map(|inner| ShuffleValueTree {
            inner,
            rng,
            dist: Cell::new(None),
            simplifying_inner: false,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ShuffleValueTree<V> {
    inner: V,
    rng: TestRng,
    dist: Cell<Option<num::usize::BinarySearch>>,
    simplifying_inner: bool,
}

impl<V: ValueTree> ShuffleValueTree<V>
where
    V::Value: Shuffleable,
{
    fn init_dist(&self, dflt: usize) -> usize {
        if self.dist.get().is_none() {
            self.dist.set(Some(num::usize::BinarySearch::new(dflt)));
        }

        self.dist.get().unwrap().current()
    }

    fn force_init_dist(&self) {
        if self.dist.get().is_none() {
            let _ = self.init_dist(self.current().shuffle_len());
        }
    }
}

impl<V: ValueTree> ValueTree for ShuffleValueTree<V>
where
    V::Value: Shuffleable,
{
    type Value = V::Value;

    fn current(&self) -> V::Value {
        let mut value = self.inner.current();
        let len = value.shuffle_len();
        // The maximum distance to swap elements. This could be larger than
        // `value` if `value` has reduced size during shrinking; that's OK,
        // since we only use this to filter swaps.
        let max_swap = self.init_dist(len);

        // If empty collection or all swaps will be filtered out, there's
        // nothing to shuffle.
        if 0 == len || 0 == max_swap {
            return value;
        }

        let mut rng = self.rng.clone();

        for start_index in 0..len - 1 {
            // Determine the other index to be swapped, then skip the swap if
            // it is too far. This ordering is critical, as it ensures that we
            // generate the same sequence of random numbers every time.

            // NOTE: The below line is the whole reason for the existence of this adapted code
            // We need to be able to swap with the same element, so that some elements remain in
            // place rather being swapped
            // let end_index = rng.gen_range(start_index + 1, len);
            let end_index = rng.gen_range(start_index, len);
            if end_index - start_index <= max_swap {
                value.shuffle_swap(start_index, end_index);
            }
        }

        value
    }

    fn simplify(&mut self) -> bool {
        if self.simplifying_inner {
            self.inner.simplify()
        } else {
            // Ensure that we've initialised `dist` to *something* to give
            // consistent non-panicking behaviour even if called in an
            // unexpected sequence.
            self.force_init_dist();
            if self.dist.get_mut().as_mut().unwrap().simplify() {
                true
            } else {
                self.simplifying_inner = true;
                self.inner.simplify()
            }
        }
    }

    fn complicate(&mut self) -> bool {
        if self.simplifying_inner {
            self.inner.complicate()
        } else {
            self.force_init_dist();
            self.dist.get_mut().as_mut().unwrap().complicate()
        }
    }
}
