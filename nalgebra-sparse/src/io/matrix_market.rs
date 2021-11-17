//! Implementation of matrix market io code.
//!
//! See the [website](https://math.nist.gov/MatrixMarket/formats.html) or the [paper](https://www.researchgate.net/publication/2630533_The_Matrix_Market_Exchange_Formats_Initial_Design) for more details about matrix market.
use crate::coo::CooMatrix;
use crate::SparseFormatError;
use crate::SparseFormatErrorKind;
use nalgebra::base::Scalar;
use nalgebra::Complex;
use pest::iterators::Pairs;
use pest::Parser;
use std::cmp::PartialEq;
use std::convert::Infallible;
use std::convert::TryFrom;
use std::fmt;
use std::fmt::Formatter;
use std::fs;
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
/// > _NOTE:_ Since the matrix market design didn't mention if multiple sparse entries with the same coordiantes are allowed or not, so, it's allowed here.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MatrixMarketErrorKind {
    /// Indicates that some word is not known to MM format
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%MatrixMarket whatever whatever whatever whatever
    /// 1 1 1
    /// 1 1 5
    /// "#;
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::ParsingError);
    /// ```
    ParsingError,

    /// Indicates that header is not valid
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
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::InvalidHeader);
    /// ```
    InvalidHeader,

    /// Indicates that the data entries in .mtx file are more or less than entries specified in .mtx file
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
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::EntryNumUnmatched);
    /// ```
    EntryNumUnmatched,

    /// Indicates that the type T is not matched with the function it called.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_str;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// % it should be called by load_coo_from_matrix_market_str::<f64>(str), or f32;
    /// 3 3 2
    /// 2 2 2.22
    /// 2 3 2.22
    /// "#;
    /// let matrix_error = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::TypeUnmatched);
    /// ```
    TypeUnmatched,

    /// Indicates that zero has been used as an index in the data, or the shape of the matrix, which is not allowed.
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
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::ZeroError);
    /// ```
    ZeroError,

    /// Indicates [SparseFormatError], while creating the sparse matrix.
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
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::SparseFormatError(SparseFormatErrorKind::IndexOutOfBounds));
    /// ```
    SparseFormatError(SparseFormatErrorKind),

    /// Indicates that a wrong diagonal element has been provieded to the matrix
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
    /// let matrix_error = load_coo_from_matrix_market_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::DiagonalError);
    ///
    /// let str = r#"
    /// %%matrixmarket matrix coordinate complex hermitian
    /// % hermitian matrix diagonal element must be a real number
    /// 5 5 2
    /// 1 1 10 2
    /// 2 1 5 2
    /// "#;
    /// let matrix_error = load_coo_from_matrix_market_str::<Complex<f64>>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::DiagonalError);
    /// ```
    /// Here the skew matrix shouldn't have an element on the diagonal
    DiagonalError,

    /// Indicates [io error](`std::io::Error`), while reading the data from file.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_matrix_market_file;
    /// # use nalgebra_sparse::io::MatrixMarketErrorKind;
    /// let file_name = "whatever.mtx";
    /// let matrix_error = load_coo_from_matrix_market_file::<f64,_>(file_name);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::IOError(std::io::ErrorKind::NotFound));
    /// ```
    IOError(std::io::ErrorKind),

    /// Indicates (skew-)symmetric (or hermitian) matrix is not lower triangle matrix.  
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
    /// let matrix_error = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::NotLowerTriangle);
    /// ```
    NotLowerTriangle,

    /// Indicates (skew-)symmetric (or hermitian) matrix is not square matrix.  
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
    /// 2 3 5
    /// "#;
    /// let matrix_error = load_coo_from_matrix_market_str::<i32>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(matrix_error.unwrap_err().kind(),MatrixMarketErrorKind::NotSquareMatrix);
    /// ```
    NotSquareMatrix,
}

impl MatrixMarketError {
    fn from_kind_and_message(error_type: MatrixMarketErrorKind, message: String) -> Self {
        Self {
            error_kind: error_type,
            message,
        }
    }

    /// The operation error kind.
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
        write!(f, "Matrix Market load error: ")?;
        match self.kind() {
            MatrixMarketErrorKind::ParsingError => {
                write!(f, "ParsingError,")?;
            }
            MatrixMarketErrorKind::InvalidHeader => {
                write!(f, "InvalidHeader,")?;
            }
            MatrixMarketErrorKind::EntryNumUnmatched => {
                write!(f, "EntryNumUnmatched,")?;
            }
            MatrixMarketErrorKind::TypeUnmatched => {
                write!(f, "TypeUnmatched,")?;
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
            MatrixMarketErrorKind::NotSquareMatrix => {
                write!(f, "NotSquareMatrix,")?;
            }
        }
        write!(f, " Message: {}", self.message)
    }
}

impl std::error::Error for MatrixMarketError {}

impl<T: fmt::Debug + std::hash::Hash + std::marker::Copy + Ord> From<pest::error::Error<T>>
    for MatrixMarketError
{
    fn from(err: pest::error::Error<T>) -> Self {
        Self::from_kind_and_message(
            MatrixMarketErrorKind::ParsingError,
            format!("Can't parse the data.\n Error: {}", err),
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
            MatrixMarketErrorKind::TypeUnmatched,
            format!(
                "Please consider using a larger integery type. Error message: {}",
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
            MatrixMarketErrorKind::TypeUnmatched,
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
/// For more details, please check Boisvert, Ronald F., Roldan Pozo, and Karin A. Remington. The matrix market formats: Initial design. Technical report, Applied and Computational Mathematics Division, NIST, 1996.  Section 3.
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

/// Base trait for matrix market types.
pub trait MatrixMarketScalar: Scalar {
    /// When matrix is an Integer matrix, it will convert a [i128] number to this type.
    fn from_i128(i: i128) -> Result<Self, MatrixMarketError>;
    /// When matrix is a Real matrix, it will convert a [f64] number to this type.
    fn from_f64(f: f64) -> Result<Self, MatrixMarketError>;
    /// When matrix is a Complx matrix, it will convert a [Complex<f64>] number to this type.
    fn from_c64(c: Complex<f64>) -> Result<Self, MatrixMarketError>;
    /// When matrix is a Pattern matrix, it will convert a unit type [unit] to this type.
    fn from_pattern(p: ()) -> Result<Self, MatrixMarketError>;
    /// When matrix is a Skew-symmetric matrix, it will convert itself to its negative.
    fn negative(self) -> Result<Self, MatrixMarketError>;
    /// When matrix is a Hermitian matrix, it will convert itself to its conjugate.
    fn conjugate(self) -> Result<Self, MatrixMarketError>;
}
/// Implement MatrixMarketScalar for primitive integer types.
macro_rules! mm_int_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {
            #[inline]
            fn from_i128(i: i128) -> Result<Self, MatrixMarketError> {
                Ok(Self::try_from(i)?)
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Int type can't be parsed from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Int type can't be parsed from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Int type can't be parsed from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Int type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Ok(-self)
            }
        }
    };
}
/// Implement MatrixMarketScalar for primitive real types.
macro_rules! mm_real_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
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
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("real type can't be parsed from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("real type can't be parsed from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("real type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Ok(-self)
            }
        }
    };
}
/// Implement MatrixMarketScalar for primitive complex types.
macro_rules! mm_complex_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for Complex<$T> {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Complex type can't be parsed from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
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
                    MatrixMarketErrorKind::TypeUnmatched,
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
        }
    };
}
/// Implement MatrixMarketScalar for primitive unit types.
macro_rules! mm_pattern_impl {
    ($T:ty) => {
        impl MatrixMarketScalar for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Pattern type can't be parsed from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Pattern type can't be parsed from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
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
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Pattern type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MatrixMarketError> {
                Err(MatrixMarketError::from_kind_and_message(
                    MatrixMarketErrorKind::TypeUnmatched,
                    format!("Pattern type has no negative"),
                ))
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

#[derive(Parser)]
#[grammar = "io/matrix_market.pest"]
struct MMParser;

/// Parses a Matrix Market file at the given path, and returns the corresponding sparse matrix as CooMatrix format.
///
/// Errors
/// --------
///
/// See [MatrixMarketErrorKind] for a list of possible error conditions.
///
/// > _NOTE:_ Here uses strong type requirements, which means if the matrix is an integer matrix, e.g. `%%matrixmarket matrix cooridnate integer general`, then you have to load it by `load_coo_from_matrix_market_file<T>`, where T is an integer type. Trying `load_coo_from_matrix_market_file<f64>` will give [TypeUnmatched](MatrixMarketErrorKind::TypeUnmatched) Error. After loading it, you can cast it into a `f64` matrix, by calling [cast](`nalgebra::base::Matrix::cast()`), but be aware of accuracy lose.
pub fn load_coo_from_matrix_market_file<T, P: AsRef<Path>>(
    path: P,
) -> Result<CooMatrix<T>, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    let file = fs::read_to_string(path)?;
    load_coo_from_matrix_market_str(&file)
}

/// Parses a Matrix Market file described by the given string, and returns the corresponding as CooMatrix format.
///
/// Errors
/// --------
///
/// See [MatrixMarketErrorKind] for a list of possible error conditions.
///
/// > _NOTE:_ Here uses strong type requirements, which means if the matrix is an integer matrix, e.g. `%%matrixmarket matrix cooridnate integer general`, then you have to load it by `load_coo_from_matrix_market_str<T>`, where T is an integer type. Trying `load_coo_from_matrix_market_str<f64>` will give [TypeUnmatched](MatrixMarketErrorKind::TypeUnmatched) Error. After loading it, you can cast it into a `f64` matrix, by calling [cast](`nalgebra::base::Matrix::cast()`),but be aware of accuracy lose.

pub fn load_coo_from_matrix_market_str<T>(data: &str) -> Result<CooMatrix<T>, MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let file = MMParser::parse(Rule::Document, data)?.next().unwrap();

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
    let mut current_dense_coordiante: (usize, usize) = (0, 0);
    if header_type.storagescheme == StorageScheme::Skew {
        // for skew dense matrix, the first element starts from (1,0)
        current_dense_coordiante = (1, 0);
    }
    // count how many entries in the matrix data
    let count = lines.clone().count();
    if count != shape.2 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::EntryNumUnmatched,
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
                    current_dense_coordiante.0,
                    current_dense_coordiante.1,
                    parse_dense_complex::<T>(&mut data_line.into_inner())?,
                );
                next_dense_coordiante(
                    &mut current_dense_coordiante,
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
                    current_dense_coordiante.0,
                    current_dense_coordiante.1,
                    parse_dense_real::<T>(&mut data_line.into_inner())?,
                );

                next_dense_coordiante(
                    &mut current_dense_coordiante,
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
                    current_dense_coordiante.0,
                    current_dense_coordiante.1,
                    parse_dense_int::<T>(&mut data_line.into_inner())?,
                );
                next_dense_coordiante(
                    &mut current_dense_coordiante,
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

/// Parse a pest structure to sparse shape information, including 3 int, which are number of rols, cols and non-zeros.
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

    // shape information can't use 0 as dimension
    if r * c * nnz == 0 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ZeroError,
            String::from(
                "
        Matrix can't have 0 as shape dimensions.
        ",
            ),
        ));
    }

    // check for square matirx, when it's not a general matrix
    if *storagescheme != StorageScheme::General && r != c {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::NotSquareMatrix,format!("(Skew-)Symmetric or hermitian matrix should be square matrix, but it has dimension {} and {}",r,c)));
    }

    Ok((r, c, nnz))
}

/// Parse a pest structure to dense shape information, including 2 int, which are number of rols, cols.
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
    // shape information can't use 0 as dimension
    if r * c == 0 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ZeroError,
            String::from(
                "
        Matrix can't have 0 as shape dimensions.
        ",
            ),
        ));
    }

    // check for square matirx, when it's not a general matrix
    if *storagescheme != StorageScheme::General && r != c {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::NotSquareMatrix,format!("(Skew-)Symmetric or hermitian matrix should be square matrix, but it has dimension {} and {}",r,c)));
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

/// Parse a pest structure to sparse real entry, including 2 int, which are number of rols, cols, and a real number as data
fn parse_sparse_real<T>(inner: &mut Pairs<'_, Rule>) -> Result<(usize, usize, T), MatrixMarketError>
where
    T: MatrixMarketScalar,
{
    // unwrap() in this function are guaranteed by parsing the data
    let entry_inner = inner.next().unwrap();
    if entry_inner.as_rule() != Rule::SparseReal {
        return Err(MatrixMarketError::from_kind_and_message(MatrixMarketErrorKind::ParsingError,format!("
        Spare real matrix requires 2 int number as coordiantes and 1 real number as data, but line {} was provided.  
        ",entry_inner.as_str() )));
    }

    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordiante(&mut inner)?;
    let d = inner.next().unwrap().as_str().parse::<f64>().unwrap();
    Ok((r, c, T::from_f64(d)?))
}

/// Parse a pest structure to sparse integer entry, including 2 int, which are number of rols, cols, and a int number as data
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
        Spare real matrix requires 3 int number as coordiantes and data, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordiante(&mut inner)?;
    // Here to guarantee it is an integer number
    let d = inner.next().unwrap().as_str().parse::<i128>()?;
    Ok((r, c, T::from_i128(d)?))
}

/// Parse a pest structure to sparse pattern entry, including 2 int, which are number of rols, cols
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
        Spare real matrix requires 2 int number as coordiantes, but line {} was provided.  
        ",
                entry_inner.as_str()
            ),
        ));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordiante(&mut inner)?;
    Ok((r, c, T::from_pattern(())?))
}

/// Parse a pest structure to sparse complex entry, including 2 int, which are number of rols, cols, and 2 real number as complex data
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
        Spare real matrix requires 2 int number as coordiantes and 2 real number as complex data, but line {} was provided.  
        ",entry_inner.as_str() )));
    }
    let mut inner = entry_inner.into_inner();
    let (r, c) = parse_sparse_coordiante(&mut inner)?;
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

/// Parse the coordiantes information used for sparse matrix
fn parse_sparse_coordiante(
    inner: &mut Pairs<'_, Rule>,
) -> Result<(usize, usize), MatrixMarketError> {
    // unwrap() in this function are guaranteed by parsing the data
    let r = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    let c = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    if r * c == 0 {
        return Err(MatrixMarketError::from_kind_and_message(
            MatrixMarketErrorKind::ZeroError,
            String::from("The data has to be one-indixed"),
        ));
    }
    // The coordiantes in matrix market is one-based, but in CooMatrix is zero-based.
    Ok((r - 1, c - 1))
}

/// Calculate the next coordiantes used for dense matrix
fn next_dense_coordiante(
    current_dense_coordiante: &mut (usize, usize),
    shape: (usize, usize, usize),
    storagescheme: &StorageScheme,
) {
    // matrix market is column based format.
    // so it follows the order (0,0) -> (1,0) -> ... -> (row, 0) -> (0,1) -> ... ->(row,col)
    // current_dense_coordiante is (row, column)
    match storagescheme {
        StorageScheme::General => {
            if current_dense_coordiante.0 < shape.0 - 1 {
                current_dense_coordiante.0 += 1
            } else {
                // jump to next column, reset row to 1, column add 1
                current_dense_coordiante.0 = 0;
                current_dense_coordiante.1 += 1;
            }
        }
        StorageScheme::Symmetric | StorageScheme::Hermitian => {
            if current_dense_coordiante.0 < shape.0 - 1 {
                current_dense_coordiante.0 += 1
            } else {
                // jump to next column, column add 1, then set row equals to current column
                // for example   (0,0) -> (1,0) -> ... -> (row, 0) -> (1,1) -> ...
                current_dense_coordiante.1 += 1;
                current_dense_coordiante.0 = current_dense_coordiante.1;
            }
        }
        StorageScheme::Skew => {
            if current_dense_coordiante.0 < shape.0 - 1 {
                current_dense_coordiante.0 += 1;
            } else {
                // jump to next column, set row equals to current column, then column add 1
                // skew matrix doesn't have element on diagonal
                // for example  (1,0) -> (2,0) -> ... -> (row, 0) -> (2,1) -> ...
                current_dense_coordiante.1 += 1;
                current_dense_coordiante.0 = current_dense_coordiante.1 + 1;
            }
        }
    }
}
