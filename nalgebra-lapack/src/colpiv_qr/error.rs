#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Error in lapack backend (code: {0})")]
    Backend(#[from] LapackErrorCode),
    #[error("Wrong matrix dimensions")]
    Dimensions,
    #[error("QR decomposition for underdetermined systems not supported")]
    Underdetermined,
    #[error("Matrix has rank zero")]
    ZeroRank,
}

/// newtype for a lapack error code
#[derive(Debug, Copy, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct LapackErrorCode(pub(crate) i32);

impl PartialEq<i32> for LapackErrorCode {
    #[inline]
    fn eq(&self, other: &i32) -> bool {
        self == &LapackErrorCode(*other)
    }
}

/// utility function to check the info return value of a lapack function
pub fn check_lapack_info(info: i32) -> Result<(), LapackErrorCode> {
    if info == 0 {
        Ok(())
    } else {
        Err(LapackErrorCode(info))
    }
}
