//! Mathematical traits.

pub use self::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Orig,
                         Projector, Rotate, Rotation, RotationMatrix, RotationWithTranslation,
                         ToHomogeneous, Transform, Transformation, Translate, Translation,
                         UniformSphereSample};

pub use self::structure::{FloatVec, FloatVecExt, FloatPnt, FloatPntExt, Basis, Cast, Col, Dim,
                          Indexable, Iterable, IterableMut, Mat, Row, AnyVec, VecExt, AnyPnt,
                          PntExt, PntAsVec, VecAsPnt, ColSlice, RowSlice, Diag, Eye};

pub use self::operations::{Absolute, ApproxEq, Cov, Det, Inv, LMul, Mean, Outer, PartialOrd, RMul,
                           ScalarAdd, ScalarSub, ScalarMul, ScalarDiv, Transpose};
pub use self::operations::{PartialOrdering, PartialLess, PartialEqual, PartialGreater, NotComparable};

pub mod geometry;
pub mod structure;
pub mod operations;
