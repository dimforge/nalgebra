/// newtype for a lapack error code
pub struct ErrorCode(i32);

/// utility function to check the info return value of a lapack function
pub fn check_lapack_info(info: i32) -> Result<(), ErrorCode> {
    if info == 0 {
        Ok(())
    } else {
        Err(ErrorCode(info))
    }
}
