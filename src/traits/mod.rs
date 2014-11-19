//! Mathematical traits.

pub use traits::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Orig,
                           Rotate, Rotation, RotationMatrix, RotationWithTranslation, ToHomogeneous,
                           Transform, Transformation, Translate, Translation, UniformSphereSample};

pub use traits::structure::{FloatVec, FloatPnt, Basis, Cast, Col, Dim, Indexable, Iterable,
                            IterableMut, Mat, SquareMat, Row, NumVec, NumPnt, PntAsVec, VecAsPnt,
                            ColSlice, RowSlice, Diag, Eye, Shape, BaseFloat, BaseNum, Zero, One,
                            Bounded};

pub use traits::operations::{Absolute, ApproxEq, Axpy, Cov, Det, Inv, LMul, Mean, Outer, POrd,
                             RMul, ScalarAdd, ScalarSub, ScalarMul, ScalarDiv, Transpose, EigenQR};
pub use traits::operations::POrdering;

pub mod geometry;
pub mod structure;
pub mod operations;
