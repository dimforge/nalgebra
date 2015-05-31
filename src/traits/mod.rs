//! Mathematical traits.

pub use traits::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Orig,
                           Rotate, Rotation, RotationMatrix, RotationWithTranslation, RotationTo,
                           ToHomogeneous, Transform, Transformation, Translate, Translation,
                           UniformSphereSample};

pub use traits::structure::{FloatVec, FloatPnt, Basis, Cast, Col, Dim, Indexable, Iterable,
                            IterableMut, Mat, SquareMat, Row, NumVec, NumPnt, PntAsVec, ColSlice,
                            RowSlice, Diag, DiagMut, Eye, Repeat, Shape, BaseFloat, BaseNum,
                            Bounded};

pub use traits::operations::{Absolute, ApproxEq, Axpy, Cov, Det, Inv, Mean, Outer, POrd, Transpose,
                             EigenQR};
pub use traits::operations::POrdering;

pub mod geometry;
pub mod structure;
pub mod operations;
