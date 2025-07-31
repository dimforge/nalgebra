//! Implementation of matrix market io code.
//!
//! See the [website](https://math.nist.gov/MatrixMarket/formats.html) or the [paper](https://www.researchgate.net/publication/2630533_The_Matrix_Market_Exchange_Formats_Initial_Design) for more details about matrix market.
use crate::SparseFormatError;
use crate::SparseFormatErrorKind;
use crate::{CooMatrix, CscMatrix, CsrMatrix};
use nalgebra::Complex;
use pest::Parser;
use pest::iterators::Pairs;
use std::cmp::PartialEq;
use std::convert::Infallible;
use std::convert::TryFrom;
use std::fmt;
use std::fmt::Formatter;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::num::ParseIntError;
use std::num::TryFromIntError;
use std::path::Path;
use std::str::FromStr;

/// A description of the error that occurred during importing a matrix from a matrix market format data.
#[derive(Debug)]
pub struct MatrixMarketError {
    error_kind: MatrixMarketErrorKind,
    message: String,
}

/// Errors produced by functions that expect well-formed matrix market format data.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MatrixMarketErrorKind {
    /// Parsing failure.
    ///
    /// Indicates that the parser failed, for example due to an unexpected string.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%MatrixMarket invalid invalid invalid invalid
    /// 1 1 1
    /// 1 1 5
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(), MatrixMarketErrorKind::ParsingError);
    /// ```
    ParsingError,

    /// Indicates that the matrix market header is invalid.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%MatrixMarket matrix coordinate real hermitian
    /// % a real matrix can't be hermitian
    /// 1 1 1
    /// 1 1 5
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::InvalidHeader);
    /// ```
    InvalidHeader,

    /// Indicates that the number of data entries in the matrix market file does not match the header.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// % it has one more data entry than specified.
    /// 3 3 1
    /// 2 2 2
    /// 2 3 2
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::EntryMismatch);
    /// ```
    EntryMismatch,

    /// Indicates that the scalar type requested is not compatible with the scalar type stored
    /// in the matrix market file.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// % it should be loaded with load_coo_from_matrix_market_str::<f64>(str) (or f32)
    /// 3 3 2
    /// 2 2 2.22
    /// 2 3 2.22
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::TypeMismatch);
    /// ```
    TypeMismatch,

    /// Indicates that zero has been used as an index in the data.
    ///
    /// **Note**: The matrix market format uses 1-based indexing.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// 1 1 1
    /// 0 0 10
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::ZeroError);
    /// ```
    ZeroError,

    /// Indicates [SparseFormatError] while creating the sparse matrix.
    ///
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// # use nalgebra_sparse::SparseFormatErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// 1 1 1
    /// 4 2 10
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),
    ///     MatrixMarketErrorKind::SparseFormatError(SparseFormatErrorKind::IndexOutOfBounds));
    /// ```
    SparseFormatError(SparseFormatErrorKind),

    /// Indicates that a wrong diagonal element has been provided to the matrix.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// # use nalgebra::Complex;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real skew-symmetric
    /// % skew-symmetric matrix can't have element on diagonal
    /// 5 5 2
    /// 1 1 10
    /// 2 1 5
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::DiagonalError);
    ///
    /// let str = r#"
    /// %%matrixmarket matrix coordinate complex hermitian
    /// % hermitian matrix diagonal element must be a real number
    /// 5 5 2
    /// 1 1 10 2
    /// 2 1 5 2
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<Complex<f64>>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::DiagonalError);
    /// ```
    /// Here the skew matrix shouldn't have an element on the diagonal.
    DiagonalError,

    /// Indicates an [IO error](`std::io::Error`) while reading the data from file.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_file;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let matrix_result = load_coo_from_matrix_market_file::<f64,_>("matrix.mtx");
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::IOError(std::io::ErrorKind::NotFound));
    /// ```
    IOError(std::io::ErrorKind),

    /// Indicates that a (skew-)symmetric (or hermitian) matrix is not a lower triangular matrix.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate integer symmetric
    /// 5 5 2
    /// 1 1 10
    /// 2 3 5
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::NotLowerTriangle);
    /// ```
    NotLowerTriangle,

    /// Indicates that a (skew-)symmetric (or hermitian) matrix is not a square matrix.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate integer symmetric
    /// 5 4 2
    /// 1 1 10
    /// 3 2 5
    /// "#;
    /// let matrix_result = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_result.is_err(), true);
    /// assert_eq!(matrix_result.unwrap_err().kind(),MatrixMarketErrorKind::NonSquare);
    /// ```
    NonSquare,
}

impl MatrixMarketError {
    fn from_kind_and_message(error_type: MatrixMarketErrorKind, message: String) -> Self {
        Self {
            error_kind: error_type,
            message,
        }
    }

    /// The matrix market error kind.
    #[must_use]
    pub fn kind(&self) -> MatrixMarketErrorKind {
        self.error_kind
    }

    /// The underlying error message.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl fmt::Display for MatrixMarketError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix Market error: ")?;
        match self.kind() {
            MatrixMarketErrorKind::ParsingError => {
                write!(f, "ParsingError,")?;
            }
            MatrixMarketErrorKind::InvalidHeader => {
                write!(f, "InvalidHeader,")?;
            }
            MatrixMarketErrorKind::EntryMismatch => {
                write!(f, "EntryMismatch,")?;
            }
            MatrixMarketErrorKind::TypeMismatch => {
                write!(f, "TypeMismatch,")?;
            }
            MatrixMarketErrorKind::SparseFormatError(_) => {
                write!(f, "SparseFormatError,")?;
            }
            MatrixMarketErrorKind::ZeroError => {
                write!(f, "ZeroError,")?;
            }
            MatrixMarketErrorKind::IOError(_) => {
                write!(f, "IOError,")?;
            }
            MatrixMarketErrorKind::DiagonalError => {
                write!(f, "DiagonalError,")?;
            }
            MatrixMarketErrorKind::NotLowerTriangle => {
                write!(f, "NotLowerTriangle,")?;
            }
            MatrixMarketErrorKind::NonSquare => {
                write!(f, "NonSquare,")?;
            }
        }
        write!(f, " message: {}", self.message)
    }
}

impl std::error::Error for MatrixMarketError {}

impl MatrixMarketError {
    fn from_pest_error<T>(error: pest::error::Error<T>) -> Self
    where
        T: fmt::Debug + std::hash::Hash + std::marker::Copy + Ord,
    {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!("Can't parse the data.\n Error: {}", error),
        )
    }
}

impl From<ParseIntError> for MatrixMarketError {
    fn from(err: ParseIntError) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!("Can't parse data as i128.\n Error: {}", err),
        )
    }
}

impl From<SparseFormatError> for MatrixMarketError {
    fn from(err: SparseFormatError) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::SparseFormatError(*err.kind()),
            format!("{}", &err),
        )
    }
}

impl From<std::io::Error> for MatrixMarketError {
    fn from(err: std::io::Error) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::IOError(err.kind()),
            format!("{}", &err),
        )
    }
}

impl From<TryFromIntError> for MatrixMarketError {
    fn from(err: TryFromIntError) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::TypeMismatch,
            format!(
                "Please consider using a larger integer type. Error message: {}",
                &err
            ),
        )
    }
}

// This is needed when calling `i128::try_from(i: i128)`
// but it won't happen
impl From<Infallible> for MatrixMarketError {
    fn from(_err: Infallible) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::TypeMismatch,
            format!("This won't happen"),
        )
    }
}

#[derive(Debug, PartialEq)]
enum Sparsity {
    Sparse,
    Dense,
}
#[derive(Debug, PartialEq)]
enum DataType {
    Real,
    Complex,
    Pattern,
    Integer,
}
#[derive(Debug, PartialEq)]
enum StorageScheme {
    Symmetric,
    General,
    Skew,
    Hermitian,
}
#[derive(Debug, PartialEq)]
struct Typecode {
    sparsity: Sparsity,
    datatype: DataType,
    storagescheme: StorageScheme,
}

impl FromStr for Sparsity {
    type Err = MatrixMarketError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "coordinate" => Ok(Sparsity::Sparse),
            "array" => Ok(Sparsity::Dense),
            _ => Err(MatrixMarketError::from_kind_and_message(
                MatrixMarketErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

impl FromStr for DataType {
    type Err = MatrixMarketError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "real" => Ok(DataType::Real),
            "complex" => Ok(DataType::Complex),
            "integer" => Ok(DataType::Integer),
            "pattern" => Ok(DataType::Pattern),
            _ => Err(MatrixMarketError::from_kind_and_message(
                MatrixMarketErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

impl FromStr for StorageScheme {
    type Err = MatrixMarketError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "skew-symmetric" => Ok(StorageScheme::Skew),
            "general" => Ok(StorageScheme::General),
            "symmetric" => Ok(StorageScheme::Symmetric),
            "hermitian" => Ok(StorageScheme::Hermitian),
            _ => Err(MatrixMarketError::from_kind_and_message(
                MatrixMarketErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

/// Precheck if it's a valid header.
///
/// For more details, please check
///
/// Boisvert, Ronald F., Roldan Pozo, and Karin A. Remington.
/// The matrix market formats: Initial design.
/// Technical report, Applied and Computational Mathematics Division, NIST, 1996.  Section 3.
fn typecode_precheck(tc: &Typecode) -> Result<(), MatrixMarketError> {
    match tc {
        Typecode {
            datatype: DataType::Real,
            storagescheme: StorageScheme::Hermitian,
            ..
        } => Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::InvalidHeader,
            String::from("Real matrix can't be hermitian."),
        )),
        Typecode {
            datatype: DataType::Integer,
            storagescheme: StorageScheme::Hermitian,
            ..
        } => Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::InvalidHeader,
            String::from("Integer matrix can't be hermitian."),
        )),
        Typecode {
            datatype: DataType::Pattern,
            storagescheme: StorageScheme::Hermitian,
            ..
        } => Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::InvalidHeader,
            String::from("Pattern matrix can't be hermitian."),
        )),
        Typecode {
            datatype: DataType::Pattern,
            storagescheme: StorageScheme::Skew,
            ..
        } => Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::InvalidHeader,
            String::from("Pattern matrix can't be skew-symmetric."),
        )),
        Typecode {
            datatype: DataType::Pattern,
            sparsity: Sparsity::Dense,
            ..
        } => Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::InvalidHeader,
            String::from("Dense matrix can't be pattern matrix."),
        )),
        // precheck success
        _ => Ok(()),
    }
}

/// Scalar types supported by the matrix market parser.
mod internal {
    use crate::io::MatrixMarketError;
    use na::{Complex, Scalar};

    pub trait SupportedMatrixMarketScalar: Scalar {
        /// When the matrix is an integer matrix, it will convert a [i128] number to this type.
        fn from_i128(i: i128) -> Result<Self, MatrixMarketError>;
        /// When matrix is a Real matrix, it will convert a [f64] number to this type.
        fn from_f64(f: f64) -> Result<Self, MatrixMarketError>;
        /// When matrix is a Complex matrix, it will convert a [Complex<f64>] number to this type.
        fn from_c64(c: Complex<f64>) -> Result<Self, MatrixMarketError>;
        /// When matrix is a Pattern matrix, it will convert a unit type [unit] to this type.
        fn from_pattern(p: ()) -> Result<Self, MatrixMarketError>;
        /// When matrix is a Skew-symmetric matrix, it will convert itself to its negative.
        fn negative(self) -> Result<Self, MatrixMarketError>;
        /// When matrix is a Hermitian matrix, it will convert itself to its conjugate.
        fn conjugate(self) -> Result<Self, MatrixMarketError>;
        /// Returns the name of SupportedMatrixMarketScalar, used when write the matrix
        fn typename() -> &'static str;
        /// Write the data self to w
        fn write_matrix_market<W: std::fmt::Write>(&self, w: W) -> Result<(), std::fmt::Error>;
    }

    pub trait SupportedMatrixMarketExport<T: SupportedMatrixMarketScalar> {
        /// iterate over triplets
        fn triplet_iter(&self) -> Box<dyn Iterator<Item = (usize, usize, &T)> + '_>;
        /// number of rows
        fn nrows(&self) -> usize;
        /// number of columns
        fn ncols(&self) -> usize;
        /// number of non-zeros
        fn nnz(&self) -> usize;
    }
}

/// A marker trait for supported matrix market scalars.
///
/// This is a sealed trait; it cannot be implemented by external crates. This is done in order to prevent leaking
/// some of the implementation details we currently rely on. We may relax this restriction in the future.
pub trait MatrixMarketScalar: internal::SupportedMatrixMarketScalar {}

/// Implement MatrixMarketScalar for primitive integer types.
macro_rules! mm_int_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {}

        impl internal::SupportedMatrixMarketScalar for $T {
            #[inline]
            fn from_i128(i: i128) -> Result<Self, MatrixMarketError> {
                Ok(Self::try_from(i)?)
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Int type can't be parsed from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Int type can't be parsed from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Int type can't be parsed from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Int type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Ok(-self)
            }
            #[inline]
            fn typename() -> &'static str {
                "integer"
            }
            #[inline]
            fn write_matrix_market<W: std::fmt::Write>(
                &self,
                mut w: W,
            ) -> Result<(), std::fmt::Error> {
                write!(w, "{}", self)
            }
        }
    };
}
/// Implement MatrixMarketScalar for primitive real types.
macro_rules! mm_real_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {}

        impl internal::SupportedMatrixMarketScalar for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("real type can't be parsed from i128"),
                ))
            }
            #[inline]
            fn from_f64(f: f64) -> Result<Self, MatrixMarketError> {
                Ok(f as Self)
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("real type can't be parsed from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("real type can't be parsed from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("real type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Ok(-self)
            }
            #[inline]
            fn typename() -> &'static str {
                "real"
            }
            #[inline]
            fn write_matrix_market<W: std::fmt::Write>(
                &self,
                mut w: W,
            ) -> Result<(), std::fmt::Error> {
                write!(w, "{}", self)
            }
        }
    };
}

/// Implement MatrixMarketScalar for primitive complex types.
macro_rules! mm_complex_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for Complex<$T> {}

        impl internal::SupportedMatrixMarketScalar for Complex<$T> {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Complex type can't be parsed from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Complex type can't be parsed from f64"),
                ))
            }
            #[inline]
            fn from_c64(c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Ok(Self {
                    re: c.re as $T,
                    im: c.im as $T,
                })
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Complex type can't be parsed from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Ok(self.conj())
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Ok(-self)
            }
            #[inline]
            fn typename() -> &'static str {
                "complex"
            }
            #[inline]
            fn write_matrix_market<W: std::fmt::Write>(
                &self,
                mut w: W,
            ) -> Result<(), std::fmt::Error> {
                write!(w, "{} {}", self.re, self.im)
            }
        }
    };
}
/// Implement MatrixMarketScalar for primitive unit types.
macro_rules! mm_pattern_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {}

        impl internal::SupportedMatrixMarketScalar for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Pattern type can't be parsed from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Pattern type can't be parsed from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Pattern type can't be parsed from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(p: ()) -> Result<Self, MatrixMarketError> {
                Ok(p)
            }

            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Pattern type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeMismatch,
                    format!("Pattern type has no negative"),
                ))
            }
            #[inline]
            fn typename() -> &'static str {
                "pattern"
            }
            #[inline]
            fn write_matrix_market<W: std::fmt::Write>(
                &self,
                mut _w: W,
            ) -> Result<(), std::fmt::Error> {
                Ok(())
            }
        }
    };
}

mm_int_impl!(i8);
mm_int_impl!(i16);
mm_int_impl!(i32);
mm_int_impl!(i64);
mm_int_impl!(i128);

mm_real_impl!(f32);
mm_real_impl!(f64);

mm_complex_impl!(f32);
mm_complex_impl!(f64);

mm_pattern_impl!(());

/// A marker trait for sparse matrix types that can be exported to the matrix market format.
///
/// This is a sealed trait; it cannot be implemented by external crates. This is done in order to prevent leaking
/// some of the implementation details we currently rely on. We may relax this restriction in the future.
pub trait MatrixMarketExport<T: MatrixMarketScalar>:
    internal::SupportedMatrixMarketExport<T>
{
}

macro_rules! mm_matrix_impl {
    ($T_MATRIX:ty) => {
        impl<T: MatrixMarketScalar> MatrixMarketExport<T> for $T_MATRIX {}

        impl<T: internal::SupportedMatrixMarketScalar> internal::SupportedMatrixMarketExport<T>
            for $T_MATRIX
        {
            #[inline]
            fn triplet_iter(&self) -> Box<dyn Iterator<Item = (usize, usize, &T)> + '_> {
                Box::new(self.triplet_iter())
            }
            #[inline]
            fn nrows(&self) -> usize {
                self.nrows()
            }
            #[inline]
            fn ncols(&self) -> usize {
                self.ncols()
            }
            #[inline]
            fn nnz(&self) -> usize {
                self.nnz()
            }
        }
    };
}

mm_matrix_impl!(CooMatrix<T>);
mm_matrix_impl!(CsrMatrix<T>);
mm_matrix_impl!(CscMatrix<T>);

#[derive(Parser)]
#[grammar = "io/matrix_market.pest"]
struct MatrixMarketParser;

/// Parses a Matrix Market file at the given path as a `CooMatrix`.
///
/// The matrix market format specification does not clarify whether duplicate entries are allowed. Our importer
/// assumes that this is permitted and produces a `CooMatrix` with possibly duplicate entries.
///
/// **Note**: A current restriction of the importer is that you must use a compatible scalar type when importing.
/// For example, in order to import a matrix stored as `integer` in the matrix market format, you must
/// import it as an integer matrix, otherwise a [TypeMismatch](MatrixMarketErrorKind::TypeMismatch) error
/// will be returned. This restriction may be lifted in the future, and is
/// tracked by issue [#1038](https://github.com/dimforge/nalgebra/issues/1038).
///
/// Errors
/// --------
///
/// See [MatrixMarketErrorKind] for a list of possible error conditions.
///
/// Examples
/// --------
/// ```no_run
/// use nalgebra_sparse::io::load_coo_from_matrix_market_file;
/// // Use e.g. `i32` for integer matrices
/// let matrix = load_coo_from_matrix_market_file::<i32,_>("path/to/matrix.mtx").unwrap();
/// ```
pub fn load_coo_from_matrix_market_file<T, P: AsRef<Path>>(
    path: P,
) -> Result<CooMatrix<T>, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    let file = fs::read_to_string(path)?;
    load_coo_from_matrix_market_str(&file)
}

/// Parses a Matrix Market file described by the given string as a `CooMatrix`.
///
/// See [load_coo_from_matrix_market_file] for more information.
///
/// Errors
/// --------
///
/// See [MatrixMarketErrorKind] for a list of possible error conditions.
///
/// Examples
/// --------
/// ```
/// use nalgebra_sparse::io::load_coo_from_matrix_market_str;
/// let str = r#"
/// %%matrixmarket matrix coordinate integer general
/// 5 4 2
/// 1 1 10
/// 2 3 5
/// "#;
/// // Use e.g. `i32` for integer matrices
/// let matrix = load_coo_from_matrix_market_str::<i32>(str).unwrap();
/// ```
pub fn load_coo_from_matrix_market_str<T>(data: &str) -> Result<CooMatrix<T>, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let file = MatrixMarketParser::parse(Rule::Document, data)
        .map_err(MatrixMarketError::from_pest_error)?
        .next()
        .unwrap();

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut data: Vec<T> = Vec::new();
    let mut lines = file.into_inner();

    let header_line = lines.next().unwrap();
    let header_type = parse_header(&mut header_line.into_inner());
    typecode_precheck(&header_type)?;

    let shape_line = lines.next().unwrap();
    // shape here is number of rows, columns, non-zeros
    let shape: (usize, usize, usize);
    match header_type.sparsity {
        Sparsity::Sparse => {
            shape = parse_sparse_shape(&mut shape_line.into_inner(), &header_type.storagescheme)?;
        }
        Sparsity::Dense => {
            shape = parse_dense_shape(&mut shape_line.into_inner(), &header_type.storagescheme)?;
        }
    }

    // used when constructing dense matrix.
    // If it's sparse matrix, it has no effect.
    let mut current_dense_coordinate: (usize, usize) = (0, 0);
    if header_type.storagescheme == StorageScheme::Skew {
        // for skew dense matrix, the first element starts from (1,0)
        current_dense_coordinate = (1, 0);
    }
    // count how many entries in the matrix data
    let count = lines.clone().count();
    if count != shape.2 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::EntryMismatch,
            format!(
                "{} entries required for the matrix, but {} was provided",
                shape.2, count,
            ),
        ));
    }

    for data_line in lines {
        let entry: (usize, usize, T);
        match header_type {
            Typecode {
                sparsity: Sparsity::Sparse,
                datatype: DataType::Real,
                ..
            } => {
                entry = parse_sparse_real::<T>(&mut data_line.into_inner())?;
            }
            Typecode {
                sparsity: Sparsity::Sparse,
                datatype: DataType::Integer,
                ..
            } => {
                entry = parse_sparse_int::<T>(&mut data_line.into_inner())?;
            }
            Typecode {
                sparsity: Sparsity::Sparse,
                datatype: DataType::Pattern,
                ..
            } => {
                entry = parse_sparse_pattern::<T>(&mut data_line.into_inner())?;
            }
            Typecode {
                sparsity: Sparsity::Sparse,
                datatype: DataType::Complex,
                ..
            } => {
                entry = parse_sparse_complex::<T>(&mut data_line.into_inner())?;
            }
            Typecode {
                sparsity: Sparsity::Dense,
                datatype: DataType::Complex,
                ..
            } => {
                entry = (
                    current_dense_coordinate.0,
                    current_dense_coordinate.1,
                    parse_dense_complex::<T>(&mut data_line.into_inner())?,
                );
                next_dense_coordinate(
                    &mut current_dense_coordinate,
                    shape,
                    &header_type.storagescheme,
                );
            }
            Typecode {
                sparsity: Sparsity::Dense,
                datatype: DataType::Real,
                ..
            } => {
                entry = (
                    current_dense_coordinate.0,
                    current_dense_coordinate.1,
                    parse_dense_real::<T>(&mut data_line.into_inner())?,
                );

                next_dense_coordinate(
                    &mut current_dense_coordinate,
                    shape,
                    &header_type.storagescheme,
                );
            }
            Typecode {
                sparsity: Sparsity::Dense,
                datatype: DataType::Integer,
                ..
            } => {
                entry = (
                    current_dense_coordinate.0,
                    current_dense_coordinate.1,
                    parse_dense_int::<T>(&mut data_line.into_inner())?,
                );
                next_dense_coordinate(
                    &mut current_dense_coordinate,
                    shape,
                    &header_type.storagescheme,
                );
            }
            _ => {
                // it shouldn't happen here, because dense matrix can't be pattern. And it will give InvalidHeader error beforehand.
                entry = (1, 1, T::from_i128(1)?)
            }
        }

        let (r, c, d) = entry;

        match header_type.storagescheme {
            StorageScheme::General => {
                rows.push(r);
                cols.push(c);
                data.push(d);
            }
            StorageScheme::Symmetric => {
                check_lower_triangle(r, c)?;
                rows.push(r);
                cols.push(c);
                data.push(d.clone());
                // don't need to add twice if the element in on diagonal
                if r != c {
                    rows.push(c);
                    cols.push(r);
                    data.push(d);
                }
            }
            StorageScheme::Skew => {
                check_lower_triangle(r, c)?;
                rows.push(r);
                cols.push(c);
                data.push(d.clone());
                // skew-symmetric matrix shouldn't have diagonal element
                if r == c {
                    return Err(MatrixMarketError::from_kind_and_message(
                        MatrixMarketErrorKind::DiagonalError,
                        format!(
                            "There is a diagonal element in skew matrix, in row(and column) {}",
                            r + 1
                        ),
                    ));
                }
                rows.push(c);
                cols.push(r);
                data.push(d.negative()?);
            }
            StorageScheme::Hermitian => {
                check_lower_triangle(r, c)?;
                rows.push(r);
                cols.push(c);
                data.push(d.clone());

                if r == c && d != d.clone().conjugate()? {
                    return Err(MatrixMarketError::from_kind_and_message(
                        MatrixMarketErrorKind::DiagonalError,
                        format!(
                            "There is a diagonal element in hermitian matrix, which is not a real number, in row(and column) {}",
                            r + 1
                        ),
                    ));
                }
                // don't need to add twice if the element in on diagonal
                if r != c {
                    rows.push(c);
                    cols.push(r);
                    data.push(d.conjugate()?);
                }
            }
        }
    }
    Ok(CooMatrix::try_from_triplets(
        shape.0, shape.1, rows, cols, data,
    )?)
}

#[inline]
/// do a quick check it the entry is in the lower triangle part of the matrix
fn check_lower_triangle(r: usize, c: usize) -> Result<(), MatrixMarketError> {
    if c > r {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::NotLowerTriangle,
            format!(
                "Entry: row {} col {} should be put into lower triangle",
                r, c
            ),
        ));
    }
    Ok(())
}

#[inline]
/// Parse a pest structure to a Typecode of the matrix.
fn parse_header(inner: &mut Pairs<'_, Rule>) -> Typecode {
    // unwrap() in this function are guaranteed by parsing the data
    Typecode {
        sparsity: inner
            .next()
            .unwrap()
            .as_str()
            .to_ascii_lowercase()
            .parse::<Sparsity>()
            .unwrap(),
        datatype: inner
            .next()
            .unwrap()
            .as_str()
            .to_ascii_lowercase()
            .parse::<DataType>()
            .unwrap(),
        storagescheme: inner
            .next()
            .unwrap()
            .as_str()
            .to_ascii_lowercase()
            .parse::<StorageScheme>()
            .unwrap(),
    }
}

// Parse shape starts here-------------------------------------------------

/// Parse a pest structure to sparse shape information, including 3 int, which are number of rows, cols and non-zeros.
fn parse_sparse_shape(
    inner: &mut Pairs<'_, Rule>,
    storagescheme: &StorageScheme,
) -> Result<(usize, usize, usize), MatrixMarketError> {
    // unwrap() in this function are guaranteed by parsing the data
    let shape_inner = inner.next().unwrap();
    if shape_inner.as_rule() != Rule::SparseShape {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::ParsingError,format!("
        Shape shape line requires 3 int numbers as number of rows, columns and non-zeros, but line {} was provided here.
        ",shape_inner.as_str())));
    }

    let mut inner = shape_inner.into_inner();

    let r = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    let c = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    let nnz = inner.next().unwrap().as_str().parse::<usize>().unwrap();

    // check for square matrix, when it's not a general matrix
    if *storagescheme != StorageScheme::General && r != c {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::NonSquare,
            format!(
                "(Skew-)Symmetric or hermitian matrix should be square matrix, but it has dimension {} and {}",
                r, c
            ),
        ));
    }

    Ok((r, c, nnz))
}

/// Parse a pest structure to dense shape information, including 2 int, which are number of rows, cols.
fn parse_dense_shape(
    inner: &mut Pairs<'_, Rule>,
    storagescheme: &StorageScheme,
) -> Result<(usize, usize, usize), MatrixMarketError> {
    // unwrap() in this function are guaranteed by parsing the data
    let shape_inner = inner.next().unwrap();
    if shape_inner.as_rule() != Rule::DenseShape {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::ParsingError,format!("
        Shape shape line requires 2 int numbers as number of rows, columns, but line {} was provided here.
        ",shape_inner.as_str())));
    }

    let mut inner = shape_inner.into_inner();
    let r = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    let c = inner.next().unwrap().as_str().parse::<usize>().unwrap();

    // check for square matrix, when it's not a general matrix
    if *storagescheme != StorageScheme::General && r != c {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::NonSquare,
            format!(
                "(Skew-)Symmetric or hermitian matrix should be square matrix, but it has dimension {} and {}",
                r, c
            ),
        ));
    }

    let n: usize;
    // Calculate the number of entries in the dense matrix
    match storagescheme {
        StorageScheme::General => {
            // general matrix should contain r*c entries
            n = r * c;
        }
        StorageScheme::Symmetric | StorageScheme::Hermitian => {
            // it must be square matrix, so r==c is true here
            // Symmetric or Hermitian should contain 1+2...+r  = r*(r+1)/2 entries
            n = r * (r + 1) / 2;
        }
        StorageScheme::Skew => {
            // it must be square matrix, so r==c is true here
            // Skew-Symmetric should contain 1+2...+r-1  = r*(r-1)/2 entries
            n = r * (r - 1) / 2;
        }
    }

    Ok((r, c, n))
}

// Parse shape ends here-------------------------------------------------

// Parse entry starts here-------------------------------------------------

/// Parse a pest structure to sparse real entry, including 2 int, which are number of rows, cols, and a real number as data
fn parse_sparse_real<T>(inner: &mut Pairs<'_, Rule>) -> Result<(usize, usize, T), MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    if entry_inner.as_rule() != Rule::SparseReal {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::ParsingError,format!("
        Spare real matrix requires 2 int number as coordinates and 1 real number as data, but line {} was provided.  
        ",entry_inner.as_str() )));
    }

    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordinate(&mut inner)?;
    let d = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    Ok((r, c, T::from_f64(d)?))
}

/// Parse a pest structure to sparse integer entry, including 2 int, which are number of rows, cols, and a int number as data
fn parse_sparse_int<T>(inner: &mut Pairs<'_, Rule>) -> Result<(usize, usize, T), MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    // Because integer numbers can also be parsed as float numbers, it will be checked again in `parse::<i128>()?`
    if entry_inner.as_rule() != Rule::SparseReal {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!(
                "
        Spare real matrix requires 3 int number as coordinates and data, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordinate(&mut inner)?;
    // Here to guarantee it is an integer number
    let d = inner.next().unwrap().as_str().parse::<i128>()?;
    Ok((r, c, T::from_i128(d)?))
}

/// Parse a pest structure to sparse pattern entry, including 2 int, which are number of rows, cols
fn parse_sparse_pattern<T>(
    inner: &mut Pairs<'_, Rule>,
) -> Result<(usize, usize, T), MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    if entry_inner.as_rule() != Rule::SparsePattern {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!(
                "
        Spare real matrix requires 2 int number as coordinates, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordinate(&mut inner)?;
    Ok((r, c, T::from_pattern(())?))
}

/// Parse a pest structure to sparse complex entry, including 2 int, which are number of rows, cols, and 2 real number as complex data
fn parse_sparse_complex<T>(
    inner: &mut Pairs<'_, Rule>,
) -> Result<(usize, usize, T), MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    if entry_inner.as_rule() != Rule::SparseComplex {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::ParsingError,format!("
        Spare real matrix requires 2 int number as coordinates and 2 real number as complex data, but line {} was provided.  
        ",entry_inner.as_str() )));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordinate(&mut inner)?;
    let real = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    let imag = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    let complex = Complex::<f64>::new(real, imag);
    Ok((r, c, T::from_c64(complex)?))
}

/// Parse a pest structure to dense real entry, including a real number as data
fn parse_dense_real<T>(inner: &mut Pairs<'_, Rule>) -> Result<T, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    if entry_inner.as_rule() != Rule::DenseReal {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!(
                "
        Dense real matrix requires 1 real number as data, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let d = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    Ok(T::from_f64(d)?)
}

/// Parse a pest structure to dense integer entry, including a integer number as data
fn parse_dense_int<T>(inner: &mut Pairs<'_, Rule>) -> Result<T, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    // Because integer numbers can also be parsed as float numbers, it will be checked again in `parse::<i128>()?`
    if entry_inner.as_rule() != Rule::DenseReal {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!(
                "
        Dense real matrix requires 1 int number as data, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    // Here to guarantee it is an integer number
    let d = inner.next().unwrap().as_str().parse::<i128>()?;
    Ok(T::from_i128(d)?)
}

/// Parse a pest structure to dense complex entry, including 2 real number as complex data
fn parse_dense_complex<T>(inner: &mut Pairs<'_, Rule>) -> Result<T, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    // Note: theoretically, 2 positive integers could also become the complex number,
    // but it would be parsed as SparsePattern, because SparsePattern has higher priority.
    // But DenseComplex can't have higher priority,
    // because, it's more often to deal with "normal" SparsePattern, rather than "unnormal" DenseComplex
    if entry_inner.as_rule() != Rule::DenseComplex && entry_inner.as_rule() != Rule::SparsePattern {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!(
                "
        Dense real matrix requires 2 real number as complex data, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let real = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    let imag = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    let complex = Complex::<f64>::new(real, imag);
    Ok(T::from_c64(complex)?)
}

// Parse entry ends here-------------------------------------------------

/// Parse the coordinates information used for sparse matrix
fn parse_sparse_coordinate(
    inner: &mut Pairs<'_, Rule>,
) -> Result<(usize, usize), MatrixMarketError> {
    // unwrap() in this function are guaranteed by parsing the data
    let r = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    let c = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    if r * c == 0 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ZeroError,
            String::from("The data has to be one-indexed"),
        ));
    }
    // The coordinates in matrix market is one-based, but in CooMatrix is zero-based.
    Ok((r - 1, c - 1))
}

/// Calculate the next coordinates used for dense matrix
fn next_dense_coordinate(
    current_dense_coordinate: &mut (usize, usize),
    shape: (usize, usize, usize),
    storagescheme: &StorageScheme,
) {
    // matrix market is column based format.
    // so it follows the order (0,0) -> (1,0) -> ... -> (row, 0) -> (0,1) -> ... ->(row,col)
    // current_dense_coordinate is (row, column)
    match storagescheme {
        StorageScheme::General => {
            if current_dense_coordinate.0 < shape.0 - 1 {
                current_dense_coordinate.0 += 1
            } else {
                // jump to next column, reset row to 1, column add 1
                current_dense_coordinate.0 = 0;
                current_dense_coordinate.1 += 1;
            }
        }
        StorageScheme::Symmetric | StorageScheme::Hermitian => {
            if current_dense_coordinate.0 < shape.0 - 1 {
                current_dense_coordinate.0 += 1
            } else {
                // jump to next column, column add 1, then set row equals to current column
                // for example   (0,0) -> (1,0) -> ... -> (row, 0) -> (1,1) -> ...
                current_dense_coordinate.1 += 1;
                current_dense_coordinate.0 = current_dense_coordinate.1;
            }
        }
        StorageScheme::Skew => {
            if current_dense_coordinate.0 < shape.0 - 1 {
                current_dense_coordinate.0 += 1;
            } else {
                // jump to next column, set row equals to current column, then column add 1
                // skew matrix doesn't have element on diagonal
                // for example  (1,0) -> (2,0) -> ... -> (row, 0) -> (2,1) -> ...
                current_dense_coordinate.1 += 1;
                current_dense_coordinate.0 = current_dense_coordinate.1 + 1;
            }
        }
    }
}

/// Save a sparse matrix as a Matrix Market format string.
///
/// The exporter only writes the matrix into `coordinate` and `general` format.
///
///
/// Examples
/// --------
/// ```
/// # use nalgebra_sparse::CooMatrix;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use nalgebra_sparse::io::{save_to_matrix_market_str};
/// let expected_str = r#"%%matrixmarket matrix coordinate integer general
/// % matrixmarket file generated by nalgebra-sparse.
/// 5 4 2
/// 1 1 10
/// 2 3 5
/// "#;
/// let row_indices = vec![0,1];
/// let col_indices = vec![0,2];
/// let values = vec![10,5];
/// let matrix = CooMatrix::try_from_triplets(5,4,row_indices,col_indices,values)?;
/// let generated_matrixmarket_str = save_to_matrix_market_str(&matrix);
/// assert_eq!(expected_str,generated_matrixmarket_str);
/// # Ok(()) }
/// ```
pub fn save_to_matrix_market_str<T, S>(sparse_matrix: &S) -> String
where
    T: MatrixMarketScalar,
    S: MatrixMarketExport<T>,
{
    let mut bytes = Vec::<u8>::new();
    // This will call impl<A: Allocator> Write for Vec<u8, A>
    // The vector will grow as needed.
    // So, unwrap here won't cause any issue.
    save_to_matrix_market(&mut bytes, sparse_matrix).unwrap();

    String::from_utf8(bytes)
        .expect("Unexpected non UTF-8 data was generated when export to matrix market string")
}

/// Save a sparse matrix to a Matrix Market format file.
///
/// The exporter only saves the matrix with the `coordinate` and `general` matrix market formats.
///
/// Errors
/// --------
///
/// See [MatrixMarketErrorKind] for a list of possible error conditions.
///
/// Examples
/// --------
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use nalgebra_sparse::io::{save_to_matrix_market_file,load_coo_from_matrix_market_str};
/// let str = r#"
/// %%matrixmarket matrix coordinate integer general
/// 5 4 2
/// 1 1 10
/// 2 3 5
/// "#;
/// let matrix = load_coo_from_matrix_market_str::<i32>(&str)?;
/// save_to_matrix_market_file(&matrix,"path/to/matrix.mtx")?;
/// # Ok(()) }
/// ```
pub fn save_to_matrix_market_file<T, S, P>(sparse_matrix: &S, path: P) -> Result<(), std::io::Error>
where
    T: MatrixMarketScalar,
    S: MatrixMarketExport<T>,
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    save_to_matrix_market(&mut file, sparse_matrix)?;
    // Quote from BufWriter doc.
    // > It is critical to call flush before BufWriter<W> is dropped. Though dropping will attempt to flush the contents of the buffer, any errors that happen in the process of dropping will be ignored. Calling flush ensures that the buffer is empty and thus dropping will not even attempt file operations.
    file.flush()
        .expect("Unexpected error when flushing the buffer data to File");
    Ok(())
}

/// Save a sparse matrix to an [std::io::Write] instance.
///
/// This is the most general save functionality. See [save_to_matrix_market_file] and
/// [save_to_matrix_market_str] for higher-level functionality.
pub fn save_to_matrix_market<T, S, W>(mut w: W, sparse_matrix: &S) -> Result<(), std::io::Error>
where
    T: MatrixMarketScalar,
    S: MatrixMarketExport<T>,
    W: Write,
{
    // write header
    writeln!(
        w,
        "%%matrixmarket matrix coordinate {} general",
        T::typename()
    )?;

    //write comment
    writeln!(w, "% matrixmarket file generated by nalgebra-sparse.")?;

    // write shape information
    writeln!(
        w,
        "{} {} {}",
        sparse_matrix.nrows(),
        sparse_matrix.ncols(),
        sparse_matrix.nnz()
    )?;

    //write triplets
    let mut buffer = String::new();
    for (r, c, d) in sparse_matrix.triplet_iter() {
        buffer.clear();
        d.write_matrix_market(&mut buffer)
            .expect("Unexpected format error was generated when write to String");
        writeln!(w, "{} {} {}", r + 1, c + 1, buffer)?;
    }
    Ok(())
}
