//! TODO

mod impl_std_ops;
pub mod serial;

/// TODO
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op<T> {
    /// TODO
    NoOp(T),
    /// TODO
    Transpose(T),
}

impl<T> Op<T> {
    /// TODO
    pub fn inner_ref(&self) -> &T {
        self.as_ref().into_inner()
    }

    /// TODO
    pub fn as_ref(&self) -> Op<&T> {
        match self {
            Op::NoOp(obj) => Op::NoOp(&obj),
            Op::Transpose(obj) => Op::Transpose(&obj)
        }
    }

    /// TODO
    pub fn convert<U>(self) -> Op<U>
        where T: Into<U>
    {
        self.map_same_op(T::into)
    }

    /// TODO
    /// TODO: Rewrite the other functions by leveraging this one
    pub fn map_same_op<U, F: FnOnce(T) -> U>(self, f: F) -> Op<U> {
        match self {
            Op::NoOp(obj) => Op::NoOp(f(obj)),
            Op::Transpose(obj) => Op::Transpose(f(obj))
        }
    }

    /// TODO
    pub fn into_inner(self) -> T {
        match self {
            Op::NoOp(obj) | Op::Transpose(obj) => obj,
        }
    }

    /// Applies the transpose operation.
    ///
    /// This operation follows the usual semantics of transposition. In particular, double
    /// transposition is equivalent to no transposition.
    pub fn transposed(self) -> Self {
        match self {
            Op::NoOp(obj) => Op::Transpose(obj),
            Op::Transpose(obj) => Op::NoOp(obj)
        }
    }
}

impl<T> From<T> for Op<T> {
    fn from(obj: T) -> Self {
        Self::NoOp(obj)
    }
}

