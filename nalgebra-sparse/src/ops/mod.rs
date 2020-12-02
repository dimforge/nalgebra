//! TODO

pub mod serial;

/// TODO
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Transpose(pub bool);

impl Transpose {
    /// TODO
    pub fn to_bool(&self) -> bool {
        self.0
    }
}