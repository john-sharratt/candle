//! Why a transfer (`file_download`, `file_upload`) did not happen: the fetch
//! or send refused or failed, or the checkout's file could not be read or
//! written.

use thiserror::Error;

use super::FileError;
use crate::public_fetch::FetchError;
use crate::ToolError;

#[derive(Debug, Error)]
pub enum TransferError {
    #[error(transparent)]
    Fetch(#[from] FetchError),
    #[error(transparent)]
    File(#[from] FileError),
}

impl ToolError for TransferError {
    fn code(&self) -> &'static str {
        match self {
            TransferError::Fetch(e) => e.code(),
            TransferError::File(e) => e.code(),
        }
    }

    fn detail(&self) -> String {
        match self {
            TransferError::Fetch(e) => e.detail(),
            TransferError::File(e) => e.detail(),
        }
    }
}
