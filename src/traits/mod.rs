//! Matrixhematical traits.

pub use traits::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Origin,
                           Rotate, Rotation, RotationMatrix, RotationWithTranslation, RotationTo,
                           ToHomogeneous, Transform, Transformation, Translate, Translation,
                           UniformSphereSample};

pub use traits::structure::{FloatVector, FloatPoint, Basis, Cast, Column, Dimension, Indexable, Iterable,
                            IterableMut, Matrix, SquareMatrix, Row, NumVector, NumPoint, PointAsVector, ColumnSlice,
                            RowSlice, Diagonal, DiagonalMut, Eye, Repeat, Shape, BaseFloat, BaseNum,
                            Bounded};

pub use traits::operations::{Absolute, ApproxEq, Axpy, Covariance, Determinant, Inverse, Mean, Outer, PartialOrder, Transpose,
                             EigenQR};
pub use traits::operations::PartialOrdering;

pub mod geometry;
pub mod structure;
pub mod operations;
