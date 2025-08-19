use super::Side;
impl Side {
    pub(crate) fn into_lapack_side_character(self) -> u8 {
        match self {
            Side::Left => b'L',
            Side::Right => b'R',
        }
    }
}
