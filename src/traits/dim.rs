/**
 * Trait of objects having a dimension (in term of spacial dimension).
 */
pub trait Dim {
  /// The dimension of the object.
  fn dim() -> uint;
}

// Some dimension token. Useful to restrict the dimension of n-dimensional
// object at the type-level.

/// Dimensional token for 0-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d0;

/// Dimensional token for 1-dimension. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d1;

/// Dimensional token for 2-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d2;

/// Dimensional token for 3-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d3;

/// Dimensional token for 4-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d4;

/// Dimensional token for 5-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d5;

/// Dimensional token for 6-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d6;

/// Dimensional token for 7-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq)]
pub struct d7;

impl Dim for d0
{ fn dim() -> uint { 0 } }

impl Dim for d1
{ fn dim() -> uint { 1 } }

impl Dim for d2
{ fn dim() -> uint { 2 } }

impl Dim for d3
{ fn dim() -> uint { 3 } }

impl Dim for d4
{ fn dim() -> uint { 4 } }

impl Dim for d5
{ fn dim() -> uint { 5 } }

impl Dim for d6
{ fn dim() -> uint { 6 } }

impl Dim for d7
{ fn dim() -> uint { 7 } }
