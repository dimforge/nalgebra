//! use mm(or MM) to represent martrix market
use crate::coo::CooMatrix;
use crate::SparseFormatError;
use crate::SparseFormatErrorKind;
use nalgebra::Complex;
use pest::iterators::Pairs;
use pest::Parser;
use std::cmp::PartialEq;
use std::fmt;
use std::fmt::Formatter;
use std::fs;
use std::num::ParseIntError;
use std::ops::Neg;
use std::path::Path;
use std::str::FromStr;

/// A description of the error that occurred during importing a matrix from a matrix market format data.
#[derive(Debug)]
pub struct MMError {
    error_kind: MMErrorKind,
    message: String,
}

/// Errors produced by functions that expect well-formed matrix market format data.
/// > _NOTE1:_ Since the matrix market design didn't mention if duplicate entries are allowed or not, so, it's allowed here.
///
/// > _NOTE2:_ Dense matrices are not supported here. For example, `%%matrixmarket matrix array real general` will give `MMErrorKind::ParsingError`.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MMErrorKind {
    /// Indicates that some word is not known to MM format
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%MatrixMarket whatever whatever whatever whatever
    /// 1 1 1
    /// 1 1 5
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::ParsingError);
    /// ```
    ParsingError,

    /// Indicates that header is not valid
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%MatrixMarket matrix coordinate real hermitian
    /// % a real matrix can't be hermitian
    /// 1 1 1
    /// 1 1 5
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::InvalidHeader);
    /// ```
    InvalidHeader,

    /// Indicates that the data entries in .mtx file are more or less than entries specified in .mtx file
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// % it has one more data entry than specified.
    /// 3 3 1
    /// 2 2 2
    /// 2 3 2
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::EntryNumUnmatched);
    /// ```
    EntryNumUnmatched,

    /// Indicates that the type T is not matched with the function it called.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate integer general
    /// % it should be called by load_coo_from_mm_str::<i32>(str), or any other integer type;
    /// 3 3 1
    /// 2 2 2
    /// 2 3 2
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::TypeUnmatched);
    /// ```
    TypeUnmatched,

    /// Indicates that zero has been used as an index in the data, which is not allowed.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// 1 1 1
    /// 0 0 10
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::ZeroIndexed);
    /// ```
    ZeroIndexed,

    /// Indicates [SparseFormatError], while creating the sparse matrix.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// # use nalgebra_sparse::SparseFormatErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real general
    /// 1 1 1
    /// 4 2 10
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::SparseFormatError(SparseFormatErrorKind::IndexOutOfBounds));
    /// ```
    SparseFormatError(SparseFormatErrorKind),

    /// Indicates that a wrong diagonal element has been provieded to the matrix
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// # use nalgebra::Complex;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate real skew-symmetric
    /// % skew-symmetric matrix can't have element on diagonal
    /// 5 5 2
    /// 1 1 10
    /// 2 1 5
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<f64>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::DiagonalError);
    ///
    /// let str = r#"
    /// %%matrixmarket matrix coordinate complex hermitian
    /// % hermitian matrix diagonal element must be a real number
    /// 5 5 2
    /// 1 1 10 2
    /// 2 1 5 2
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<Complex<f64>>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::DiagonalError);
    /// ```
    /// Here the skew matrix shouldn't have an element on the diagonal
    DiagonalError,

    /// Indicates [io error](`std::io::Error`), while reading the data from file.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_file;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let file_name = "whatever.mtx";
    /// let matrix_error = load_coo_from_mm_file::<f64,_>(file_name);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::IOError(std::io::ErrorKind::NotFound));
    /// ```
    IOError(std::io::ErrorKind),

    /// Indicates (skew-)symmetric (or hermitian) matrix is not lower triangle matrix.  
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use nalgebra_sparse::io::load_coo_from_mm_str;
    /// # use nalgebra_sparse::io::MMErrorKind;
    /// let str = r#"
    /// %%matrixmarket matrix coordinate integer symmetric
    /// 5 5 2
    /// 1 1 10
    /// 2 3 5
    /// "#;
    /// let matrix_error = load_coo_from_mm_str::<i32>(str);
    /// assert_eq!(matrix_error.is_err(), true);
    /// assert_eq!(*matrix_error.unwrap_err().kind(),MMErrorKind::LowerTriangleError);
    /// ```
    LowerTriangleError,
}

impl MMError {
    fn from_kind_and_message(error_type: MMErrorKind, message: String) -> Self {
        Self {
            error_kind: error_type,
            message,
        }
    }

    /// The operation error kind.
    #[must_use]
    pub fn kind(&self) -> &MMErrorKind {
        &self.error_kind
    }

    /// The underlying error message.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl fmt::Display for MMError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix Market load error: ")?;
        match self.kind() {
            MMErrorKind::ParsingError => {
                write!(f, "ParsingError,")?;
            }
            MMErrorKind::InvalidHeader => {
                write!(f, "InvalidHeader,")?;
            }
            MMErrorKind::EntryNumUnmatched => {
                write!(f, "EntryNumUnmatched,")?;
            }
            MMErrorKind::TypeUnmatched => {
                write!(f, "TypeUnmatched,")?;
            }
            MMErrorKind::SparseFormatError(_) => {
                write!(f, "SparseFormatError,")?;
            }
            MMErrorKind::ZeroIndexed => {
                write!(f, "ZeroIndexed,")?;
            }
            MMErrorKind::IOError(_) => {
                write!(f, "IOError,")?;
            }
            MMErrorKind::DiagonalError => {
                write!(f, "DiagonalError,")?;
            }
            MMErrorKind::LowerTriangleError => {
                write!(f, "LowerTriangleError,")?;
            }
        }
        write!(f, " Message: {}", self.message)
    }
}

impl std::error::Error for MMError {}

impl<T: fmt::Debug + std::hash::Hash + std::marker::Copy + Ord> From<pest::error::Error<T>>
    for MMError
{
    fn from(err: pest::error::Error<T>) -> Self {
        Self::from_kind_and_message(
            MMErrorKind::ParsingError,
            format!("Can't parse the data.\n Error: {}", err),
        )
    }
}

impl From<ParseIntError> for MMError {
    fn from(err: ParseIntError) -> Self {
        Self::from_kind_and_message(
            MMErrorKind::ParsingError,
            format!("Can't parse data as i128.\n Error: {}", err),
        )
    }
}

impl From<SparseFormatError> for MMError {
    fn from(err: SparseFormatError) -> Self {
        Self::from_kind_and_message(
            MMErrorKind::SparseFormatError(*err.kind()),
            format!("{}", &err),
        )
    }
}

impl From<std::io::Error> for MMError {
    fn from(err: std::io::Error) -> Self {
        Self::from_kind_and_message(MMErrorKind::IOError(err.kind()), format!("{}", &err))
    }
}

#[derive(Debug, PartialEq)]
enum Sparsity {
    Sparse,
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
    sp: Sparsity,
    dt: DataType,
    ss: StorageScheme,
}

impl FromStr for Sparsity {
    type Err = MMError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "coordinate" => Ok(Sparsity::Sparse),
            _ => Err(MMError::from_kind_and_message(
                MMErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

impl FromStr for DataType {
    type Err = MMError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "real" => Ok(DataType::Real),
            "complex" => Ok(DataType::Complex),
            "integer" => Ok(DataType::Integer),
            "pattern" => Ok(DataType::Pattern),
            _ => Err(MMError::from_kind_and_message(
                MMErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

impl FromStr for StorageScheme {
    type Err = MMError;
    /// Assumes that `word` is already lower case.
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "skew-symmetric" => Ok(StorageScheme::Skew),
            "general" => Ok(StorageScheme::General),
            "symmetric" => Ok(StorageScheme::Symmetric),
            "hermitian" => Ok(StorageScheme::Hermitian),
            _ => Err(MMError::from_kind_and_message(
                MMErrorKind::ParsingError,
                format!("keyword {} is unknown", word),
            )),
        }
    }
}

/// Precheck if it's a valid header.
///
/// For more details, please check Boisvert, Ronald F., Roldan Pozo, and Karin A. Remington. The matrix market formats: Initial design. Technical report, Applied and Computational Mathematics Division, NIST, 1996.  Section 3.
fn typecode_precheck(tc: &Typecode) -> Result<(), MMError> {
    match tc {
        Typecode {
            dt: DataType::Real,
            ss: StorageScheme::Hermitian,
            ..
        } => Err(MMError::from_kind_and_message(
            MMErrorKind::InvalidHeader,
            String::from("Real matrix can't be hermitian."),
        )),
        Typecode {
            dt: DataType::Integer,
            ss: StorageScheme::Hermitian,
            ..
        } => Err(MMError::from_kind_and_message(
            MMErrorKind::InvalidHeader,
            String::from("Integer matrix can't be hermitian."),
        )),
        Typecode {
            dt: DataType::Pattern,
            ss: StorageScheme::Hermitian,
            ..
        } => Err(MMError::from_kind_and_message(
            MMErrorKind::InvalidHeader,
            String::from("Pattern matrix can't be hermitian."),
        )),
        Typecode {
            dt: DataType::Pattern,
            ss: StorageScheme::Skew,
            ..
        } => Err(MMError::from_kind_and_message(
            MMErrorKind::InvalidHeader,
            String::from("Pattern matrix can't be skew-symmetric."),
        )),
        // precheck success
        _ => Ok(()),
    }
}

/// Base trait for matrix market types.
pub trait MMType: Sized + Clone {
    /// When matrix is an Integer matrix, it will convert a [i128] number to this type.
    fn from_i128(i: i128) -> Result<Self, MMError>;
    /// When matrix is a Real matrix, it will convert a [f64] number to this type.
    fn from_f64(f: f64) -> Result<Self, MMError>;
    /// When matrix is a Complx matrix, it will convert a [Complex<f64>] number to this type.
    fn from_c64(c: Complex<f64>) -> Result<Self, MMError>;
    /// When matrix is a Pattern matrix, it will convert a unit type [unit] to this type.
    fn from_pattern(p: ()) -> Result<Self, MMError>;
    /// When matrix is a Skew-symmetric matrix, it will convert itself to its negative.
    fn negative(self) -> Result<Self, MMError>;
    /// When matrix is a Hermitian matrix, it will convert itself to its conjugate.
    fn conjugate(self) -> Result<Self, MMError>;
}
/// Implement MMType for primitive integer types.
macro_rules! mm_int_impl {
    ($T:ty) => {
        impl MMType for $T {
            #[inline]
            fn from_i128(i: i128) -> Result<Self, MMError> {
                Ok(i as Self)
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Int type can't parse from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Int type can't parse from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Int type can't parse from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Int type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MMError>
            where
                Self: Neg<Output = Self>,
            {
                Ok(-self)
            }
        }
    };
}
/// Implement MMType for primitive float types.
macro_rules! mm_float_impl {
    ($T:ty) => {
        impl MMType for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Float type can't parse from i128"),
                ))
            }
            #[inline]
            fn from_f64(f: f64) -> Result<Self, MMError> {
                Ok(f as Self)
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Float type can't parse from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Float type can't parse from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Float type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MMError>
            where
                Self: Neg<Output = Self>,
            {
                Ok(-self)
            }
        }
    };
}
/// Implement MMType for primitive complex types.
macro_rules! mm_complex_impl {
    ($T:ty) => {
        impl MMType for Complex<$T> {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Complex type can't parse from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Complex type can't parse from f64"),
                ))
            }
            #[inline]
            fn from_c64(c: Complex<f64>) -> Result<Self, MMError> {
                Ok(Self {
                    re: c.re as $T,
                    im: c.im as $T,
                })
            }
            #[inline]
            fn from_pattern(_p: ()) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Complex type can't parse from ()"),
                ))
            }
            #[inline]
            fn conjugate(self) -> Result<Self, MMError> {
                Ok(self.conj())
            }
            #[inline]
            fn negative(self) -> Result<Self, MMError>
            where
                Self: Neg<Output = Self>,
            {
                Ok(-self)
            }
        }
    };
}
/// Implement MMType for primitive unit types.
macro_rules! mm_pattern_impl {
    ($T:ty) => {
        impl MMType for $T {
            #[inline]
            fn from_i128(_i: i128) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Pattern type can't parse from i128"),
                ))
            }
            #[inline]
            fn from_f64(_f: f64) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Pattern type can't parse from f64"),
                ))
            }
            #[inline]
            fn from_c64(_c: Complex<f64>) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Pattern type can't parse from Complex<f64>"),
                ))
            }
            #[inline]
            fn from_pattern(p: ()) -> Result<Self, MMError> {
                Ok(p)
            }

            #[inline]
            fn conjugate(self) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
                    format!("Pattern type has no conjugate"),
                ))
            }
            #[inline]
            fn negative(self) -> Result<Self, MMError> {
                Err(MMError::from_kind_and_message(
                    MMErrorKind::TypeUnmatched,
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

mm_float_impl!(f32);
mm_float_impl!(f64);

mm_complex_impl!(f32);
mm_complex_impl!(f64);

mm_pattern_impl!(());

#[derive(Parser)]
#[grammar = "io/matrix_market.pest"]
struct MMParser;

/// Parsing a pest structure to a Typecode of the matrix.
fn parsing_header(inner: &mut Pairs<'_, Rule>, header_type: &mut Typecode) -> Result<(), MMError> {
    // once the data can be parsed, all unwrap() in this function here will not panic.
    header_type.sp = inner
        .next()
        .unwrap()
        .as_str()
        .to_ascii_lowercase()
        .parse::<Sparsity>()
        .unwrap();
    header_type.dt = inner
        .next()
        .unwrap()
        .as_str()
        .to_ascii_lowercase()
        .parse::<DataType>()
        .unwrap();
    header_type.ss = inner
        .next()
        .unwrap()
        .as_str()
        .to_ascii_lowercase()
        .parse::<StorageScheme>()
        .unwrap();
    typecode_precheck(&header_type)
}

/// Parsing a pest structure to 3 int, which are number of rols, cols and non-zeros.
fn parsing_shape(inner: &mut Pairs<'_, Rule>, shape: &mut (usize, usize, usize)) {
    // once the file can be parsed, all unwrap() in this function here will not panic.
    shape.0 = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    shape.1 = inner.next().unwrap().as_str().parse::<usize>().unwrap();
    shape.2 = inner.next().unwrap().as_str().parse::<usize>().unwrap();
}

#[inline]
/// Do a precheck of remaing data. Because in matrix_market.pest file, either 0, or 1 or 2 float number(s) as data is valid, however, for example, for Real matrix, only 1 float is valid. And 2 float for Complex matrix.  
fn check_value_length(inner: &Pairs<'_, Rule>, l: usize) -> Result<(), MMError> {
    let copy = inner.clone();
    let c = copy.count();
    if l != c {
        return Err(MMError::from_kind_and_message(
            MMErrorKind::ParsingError,
            format!("{} data required, but {} provided.", l, c),
        ));
    }
    Ok(())
}

#[inline]
// do a quick check it the entry is in the lower triangle part of the matrix
fn check_lower_triangle(r: usize, c: usize) -> Result<(), MMError> {
    if c > r {
        return Err(MMError::from_kind_and_message(
            MMErrorKind::LowerTriangleError,
            format!(
                "Entry: row {} col {} should be put into lower triangle",
                r, c
            ),
        ));
    }
    Ok(())
}

/// Parses a Matrix Market file at the given path, and returns the corresponding sparse matrix as CooMatrix format.
pub fn load_coo_from_mm_file<T, P: AsRef<Path>>(path: P) -> Result<CooMatrix<T>, MMError>
where
    T: MMType,
{
    let file = fs::read_to_string(path)?;
    load_coo_from_mm_str(&file)
}

/// Parses a Matrix Market file described by the given string, and returns the corresponding as CooMatrix format.
pub fn load_coo_from_mm_str<T>(data: &str) -> Result<CooMatrix<T>, MMError>
where
    T: MMType,
{
    // unwrap() here guaranteed when data can be parsed
    let file = MMParser::parse(Rule::Document, data)?.next().unwrap();
    //  Default typecode
    let mut header_type = Typecode {
        sp: Sparsity::Sparse,
        dt: DataType::Real,
        ss: StorageScheme::General,
    };
    let mut shape = (0, 0, 0);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut data: Vec<T> = Vec::new();

    let mut count = 0;
    for line in file.into_inner() {
        match line.as_rule() {
            Rule::Header => {
                let mut inner = line.into_inner();
                parsing_header(&mut inner, &mut header_type)?;
            }
            Rule::Shape => {
                let mut inner = line.into_inner();
                parsing_shape(&mut inner, &mut shape);
            }
            Rule::Entry => {
                count += 1;
                let mut inner = line.into_inner();
                // NOTE: indices are 1-based.
                // unwrap() here guaranteed when data can be parsed
                let r = inner.next().unwrap().as_str().parse::<usize>().unwrap();
                if r == 0 {
                    return Err(MMError::from_kind_and_message(
                        MMErrorKind::ZeroIndexed,
                        String::from("The data has to be one-indixed"),
                    ));
                }
                let r = r - 1;
                // unwrap() here guaranteed when data can be parsed
                let c = inner.next().unwrap().as_str().parse::<usize>().unwrap();
                if c == 0 {
                    return Err(MMError::from_kind_and_message(
                        MMErrorKind::ZeroIndexed,
                        String::from("The data has to be one-indixed"),
                    ));
                }
                let c = c - 1;
                let d: T;
                match header_type.dt {
                    DataType::Integer => {
                        check_value_length(&inner, 1)?;
                        // unwrap() here guaranteed by check_value_length
                        let i = inner.next().unwrap().as_str().parse::<i128>()?;
                        d = T::from_i128(i)?;
                    }
                    DataType::Real => {
                        check_value_length(&inner, 1)?;
                        // first unwrap() here guaranteed by check_value_length
                        // second unwrap() here guaranteed by parsing the data
                        let i = inner.next().unwrap().as_str().parse::<f64>().unwrap();
                        d = T::from_f64(i)?;
                    }
                    DataType::Complex => {
                        check_value_length(&inner, 2)?;
                        // first unwrap() here guaranteed by check_value_length
                        // second unwrap() here guaranteed by parsing the data
                        let real = inner.next().unwrap().as_str().parse::<f64>().unwrap();
                        // first unwrap() here guaranteed by check_value_length
                        // second unwrap() here guaranteed by parsing the data
                        let imag = inner.next().unwrap().as_str().parse::<f64>().unwrap();
                        // only complex could be hermitian, and check diagonal element is a real number
                        if header_type.ss == StorageScheme::Hermitian && r == c && imag != 0.0 {
                            return Err(MMError::from_kind_and_message(MMErrorKind::DiagonalError,format!("There is a diagonal element in hermitian matrix, in row(and column) {}, but imaginary part is not zero",r)));
                        }
                        d = T::from_c64(Complex::<f64>::new(real, imag))?;
                    }
                    DataType::Pattern => {
                        check_value_length(&inner, 0)?;
                        d = T::from_pattern(())?;
                    }
                }

                match header_type.ss {
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
                            return Err(MMError::from_kind_and_message(MMErrorKind::DiagonalError,format!("There is a diagonal element in skew matrix, in row(and column) {}",r)));
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
                        // don't need to add twice if the element in on diagonal
                        // diagonal element has been checked before to make sure the imaginary part is zero
                        if r != c {
                            rows.push(c);
                            cols.push(r);
                            data.push(d.conjugate()?);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if count != shape.2 {
        return Err(MMError::from_kind_and_message(
            MMErrorKind::EntryNumUnmatched,
            format!(
                "expected {} entries in matrix market file, but found {}",
                shape.2, count
            ),
        ));
    }

    Ok(CooMatrix::try_from_triplets(
        shape.0, shape.1, rows, cols, data,
    )?)
}
