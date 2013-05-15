pub trait Dim
{
  fn dim() -> uint;
}

// Some dimension token. Useful to restrict the dimension of n-dimensional
// object at the type-level.
pub struct d0;
pub struct d1;
pub struct d2;
pub struct d3;
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
