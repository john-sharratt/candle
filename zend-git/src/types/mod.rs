//! The validated values every operation takes and returns.

pub mod mode;
pub mod oid;
pub mod ref_name;
pub mod remote_url;
pub mod repo_path;
pub mod rev;
pub mod signature;
pub mod tag_name;

pub use mode::FileMode;
pub use oid::{ObjectFormat, Oid};
pub use ref_name::{BranchName, RefName, RemoteName};
pub use remote_url::RemoteUrl;
pub use repo_path::{RepoPath, PROTECTED_SEGMENT};
pub use rev::Rev;
pub use signature::{GitTime, Signature};
pub use tag_name::TagName;
