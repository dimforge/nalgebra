use alga::general::{ClosedAdd, ClosedMul};
use num::{One, Zero};
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Range};
use std::slice;

use allocator::Allocator;
use constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use sparse::{CsMatrix, CsStorage, CsVector};
use storage::{Storage, StorageMut};
use {DefaultAllocator, Dim, Matrix, MatrixMN, Real, Scalar, Vector, VectorN, U1};

pub struct SymbolicAnalysis {
    pinv: Vec<usize>,
    q: Vec<usize>,
    elimination_tree: Vec<usize>,
    cp: Vec<usize>,
    leftmost: Vec<usize>,
    m2: usize,
    lnz: usize,
    unz: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct EliminationTreeNode {
    parent: usize,
}

impl EliminationTreeNode {
    pub fn root() -> Self {
        EliminationTreeNode {
            parent: usize::max_value(),
        }
    }

    pub fn with_parent(parent: usize) -> Self {
        EliminationTreeNode { parent }
    }

    pub fn is_root(&self) -> bool {
        self.parent == usize::max_value()
    }

    pub fn parent(&self) -> usize {
        self.parent
    }
}

impl<N: Real, D: Dim, S: CsStorage<N, D, D>> CsMatrix<N, D, D, S> {
    fn elimination_tree(&self) -> Vec<EliminationTreeNode> {
        let (nrows, ncols) = self.data.shape();
        assert_eq!(
            nrows.value(),
            ncols.value(),
            "The matrix `self` must be square to compute its elimination tree."
        );

        let mut forest: Vec<_> = iter::repeat(EliminationTreeNode::root())
            .take(nrows.value())
            .collect();
        let mut ancestor: Vec<_> = iter::repeat(usize::max_value())
            .take(nrows.value())
            .collect();

        for k in 0..nrows.value() {
            for irow in self.data.column_row_indices(k) {
                let mut i = irow;

                while i < k {
                    let i_ancestor = ancestor[i];
                    ancestor[i] = k;

                    if i_ancestor == usize::max_value() {
                        forest[i] = EliminationTreeNode::with_parent(k);
                        break;
                    }

                    i = i_ancestor;
                }
            }
        }

        forest
    }

    fn reach(
        &self,
        j: usize,
        max_j: usize,
        tree: &[EliminationTreeNode],
        marks: &mut Vec<bool>,
        out: &mut Vec<usize>,
    ) {
        marks.clear();
        marks.resize(tree.len(), false);

        for irow in self.data.column_row_indices(j) {
            let mut curr = irow;
            while curr != usize::max_value() && curr <= max_j && !marks[curr] {
                marks[curr] = true;
                out.push(curr);
                curr = tree[curr].parent;
            }
        }
    }

    fn column_counts(&self, tree: &[EliminationTreeNode]) -> Vec<usize> {
        let len = self.data.shape().0.value();
        let mut counts: Vec<_> = iter::repeat(0).take(len).collect();
        let mut reach = Vec::new();
        let mut marks = Vec::new();

        for i in 0..len {
            self.reach(i, i, tree, &mut marks, &mut reach);

            for j in reach.drain(..) {
                counts[j] += 1;
            }
        }

        counts
    }

    fn tree_postorder(tree: &[EliminationTreeNode]) -> Vec<usize> {
        // FIXME: avoid all those allocations?
        let mut first_child: Vec<_> = iter::repeat(usize::max_value()).take(tree.len()).collect();
        let mut other_children: Vec<_> =
            iter::repeat(usize::max_value()).take(tree.len()).collect();

        // Build the children list from the parent list.
        // The set of children of the node `i` is given by the linked list
        // starting at `first_child[i]`. The nodes of this list are then:
        // { first_child[i], other_children[first_child[i]], other_children[other_children[first_child[i]], ... }
        for (i, node) in tree.iter().enumerate() {
            if !node.is_root() {
                let brother = first_child[node.parent];
                first_child[node.parent] = i;
                other_children[i] = brother;
            }
        }

        let mut stack = Vec::with_capacity(tree.len());
        let mut postorder = Vec::with_capacity(tree.len());

        for (i, node) in tree.iter().enumerate() {
            if node.is_root() {
                Self::dfs(
                    i,
                    &mut first_child,
                    &other_children,
                    &mut stack,
                    &mut postorder,
                )
            }
        }

        postorder
    }

    fn dfs(
        i: usize,
        first_child: &mut [usize],
        other_children: &[usize],
        stack: &mut Vec<usize>,
        result: &mut Vec<usize>,
    ) {
        stack.clear();
        stack.push(i);

        while let Some(n) = stack.pop() {
            let child = first_child[n];

            if child == usize::max_value() {
                // No children left.
                result.push(n);
            } else {
                stack.push(n);
                stack.push(child);
                first_child[n] = other_children[child];
            }
        }
    }
}
