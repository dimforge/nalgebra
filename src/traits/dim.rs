pub trait Dim
{
  /// The dimension of the object.
  fn dim() -> uint;
}

// Some dimension token. Useful to restrict the dimension of n-dimensional
// object at the type-level.
/// Dimensional token for 0-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
pub struct d0;
/// Dimensional token for 1-dimension. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
pub struct d1;
/// Dimensional token for 2-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
pub struct d2;
/// Dimensional token for 3-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
pub struct d3;
/// Dimensional token for 4-dimensions. Dimensional tokens are the preferred
/// way to specify at the type level the dimension of n-dimensional objects.
pub struct d4;

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
