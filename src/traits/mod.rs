//! Mathematical traits.

pub use self::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Orig,
                         Rotate, Rotation, RotationMatrix, RotationWithTranslation, ToHomogeneous,
                         Transform, Transformation, Translate, Translation, UniformSphereSample};

pub use self::structure::{FloatVec, FloatPnt, Basis, Cast, Col, Dim, Indexable, Iterable,
                          IterableMut, Mat, Row, AnyVec, AnyPnt, PntAsVec, VecAsPnt, ColSlice,
                          RowSlice, Diag, Eye, Shape};

pub use self::operations::{Absolute, ApproxEq, Axpy, Cov, Det, Inv, LMul, Mean, Outer, POrd,
                           RMul, ScalarAdd, ScalarSub, ScalarMul, ScalarDiv, Transpose};
pub use self::operations::{POrdering, PartialLess, PartialEqual, PartialGreater, NotComparable};

pub mod geometry;
pub mod structure;
pub mod operations;
