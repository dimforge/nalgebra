/**
 * Trait of objects having a spacial dimension.
 */
pub trait Dim {
  /// The dimension of the object.
  fn dim() -> uint;
}

// Some dimension token. Useful to restrict the dimension of n-dimensional
// object at the type-level.

/// Dimensional token for 0-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D0;

/// Dimensional token for 1-dimension. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D1;

/// Dimensional token for 2-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D2;

/// Dimensional token for 3-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D3;

/// Dimensional token for 4-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D4;

/// Dimensional token for 5-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D5;

/// Dimensional token for 6-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
#[deriving(Eq, Ord, ToStr)]
pub struct D6;

impl Dim for D0
{ fn dim() -> uint { 0 } }

impl Dim for D1
{ fn dim() -> uint { 1 } }

impl Dim for D2
{ fn dim() -> uint { 2 } }

impl Dim for D3
{ fn dim() -> uint { 3 } }

impl Dim for D4
{ fn dim() -> uint { 4 } }

impl Dim for D5
{ fn dim() -> uint { 5 } }

impl Dim for D6
{ fn dim() -> uint { 6 } }
