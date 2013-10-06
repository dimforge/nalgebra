//! Mathematical traits.

pub use self::geometry::{AbsoluteRotate, Cross, CrossMatrix, Dot, FromHomogeneous, Norm, Rotate,
                         Rotation, RotationMatrix, RotationWithTranslation, ToHomogeneous,
                         Transform, Transformation, Translate, Translation, UniformSphereSample};

pub use self::structure::{AlgebraicVec, AlgebraicVecExt, Basis, Col, Dim, Indexable, Iterable,
                          IterableMut, Mat, MatCast, Row, Vec, VecCast, VecExt};

pub use self::operations::{Absolute, Cov, Inv, LMul, Mean, Outer, RMul, ScalarAdd, ScalarSub,
                           Transpose};

pub mod geometry;
pub mod structure;
pub mod operations;
