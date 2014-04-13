//! Mathematical traits.

pub use self::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Rotate,
                         Rotation, RotationMatrix, RotationWithTranslation, ToHomogeneous,
                         Transform, Transformation, Translate, Translation, UniformSphereSample};

pub use self::structure::{FloatVec, FloatVecExt, Basis, Cast, Col, Dim, Indexable,
                          Iterable, IterableMut, Mat, Row, AnyVec, VecExt};

pub use self::operations::{Absolute, ApproxEq, Cov, Inv, LMul, Mean, Outer, PartialOrd, RMul,
                           ScalarAdd, ScalarSub, Transpose};
pub use self::operations::{PartialOrdering, PartialLess, PartialEqual, PartialGreater, NotComparable};

pub mod geometry;
pub mod structure;
pub mod operations;
