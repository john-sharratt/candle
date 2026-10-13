//! What a binary says of its build (§3.9): `--build-info`'s answer. A
//! candidate runs each binary beside its own with `--build-info` to learn which
//! were built with it, and a member checks an artifact against its substrate by
//! the last three fields — what the build can open.
//!
//! The answer is this structure's JSON on one line, fields in declaration
//! order, so one build has one answer.

use std::ffi::OsString;

use serde::{Deserialize, Serialize};

use super::this_build;
use crate::frame::BuildId;

/// The one argument that asks a binary for its build.
pub const BUILD_INFO_FLAG: &str = "--build-info";

/// The binaries a build produces, as `kind` names them.
pub const ZEND: &str = "zend";
pub const NPCD: &str = "npcd";
pub const MOTHERD: &str = "motherd";
pub const AGENT: &str = "tokera-agent";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Which binary: [`ZEND`], [`NPCD`], [`MOTHERD`] or [`AGENT`].
    pub kind: String,
    pub count: u32,
    pub hash: String,
    pub local: String,
    /// The OS and CPU architecture it runs on ([`platform`]).
    pub platform: String,
    #[serde(flatten)]
    pub opens: Opens,
}

/// What a build can open (§3.9): the substrate's log format, the record types
/// it decodes, and the model it runs. An agent opens no substrate and runs no
/// model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Opens {
    /// The redo log's `FILE_FORMAT_VERSION`.
    pub log_format: Option<u32>,
    /// The tag of every record type it decodes, ascending.
    pub record_tags: Vec<u8>,
    /// The name of the model it is prescribed (`host::MODEL`'s preset).
    pub model: Option<String>,
}

impl BuildInfo {
    /// This binary's answer, as `kind`.
    pub fn of(kind: &str, opens: Opens) -> BuildInfo {
        let build = this_build();
        BuildInfo {
            kind: kind.to_string(),
            count: build.count,
            hash: build.hash,
            local: build.local,
            platform: platform(),
            opens,
        }
    }

    /// The build it names.
    pub fn build(&self) -> BuildId {
        BuildId {
            count: self.count,
            hash: self.hash.clone(),
            local: self.local.clone(),
        }
    }

    pub fn encode(&self) -> String {
        serde_json::to_string(self).expect("a build's answer always serializes")
    }

    /// One binary's answer, from its output; `None` when it is not one.
    pub fn decode(text: &str) -> Option<BuildInfo> {
        serde_json::from_str(text.trim()).ok()
    }
}

/// The OS and CPU architecture this binary was built for: `windows-x86_64`.
pub fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Whether `args` — the process's, without its own name — ask for the build:
/// exactly the one flag.
pub fn asks_for_build(args: &[OsString]) -> bool {
    matches!(args, [only] if only == BUILD_INFO_FLAG)
}

/// Answer `--build-info` and exit, when it is all this process was given. Every
/// binary calls this first, before it parses anything else: an agent that
/// requires `--config` must still answer.
pub fn answer_if_asked(kind: &str, opens: impl FnOnce() -> Opens) {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    if asks_for_build(&args) {
        println!("{}", BuildInfo::of(kind, opens()).encode());
        std::process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> BuildInfo {
        BuildInfo {
            kind: ZEND.into(),
            count: 4815,
            hash: "9f8e7d6".into(),
            local: String::new(),
            platform: "windows-x86_64".into(),
            opens: Opens {
                log_format: Some(3),
                record_tags: vec![1, 2, 31],
                model: Some("Qwen38_FlashNext_Q4KO".into()),
            },
        }
    }

    /// One build has one answer: its fields in order, on one line, and it
    /// reads back.
    #[test]
    fn the_answer_is_exact_json() {
        let want = concat!(
            r#"{"kind":"zend","count":4815,"hash":"9f8e7d6","local":"","#,
            r#""platform":"windows-x86_64","log_format":3,"record_tags":[1,2,31],"#,
            r#""model":"Qwen38_FlashNext_Q4KO"}"#
        );
        assert_eq!(info().encode(), want);
        assert_eq!(BuildInfo::decode(&format!("{want}\n")), Some(info()));
        assert_eq!(
            info().build(),
            BuildId {
                count: 4815,
                hash: "9f8e7d6".into(),
                local: String::new()
            }
        );
    }

    /// An agent opens no substrate and runs no model.
    #[test]
    fn an_agent_says_it_opens_nothing() {
        let agent = BuildInfo {
            kind: AGENT.into(),
            opens: Opens::default(),
            ..info()
        };
        assert!(agent
            .encode()
            .ends_with(r#""log_format":null,"record_tags":[],"model":null}"#));
    }

    #[test]
    fn output_that_is_not_an_answer_is_none() {
        assert_eq!(BuildInfo::decode("error: unexpected argument"), None);
        assert_eq!(BuildInfo::decode(""), None);
    }

    /// Only the flag alone asks; beside other arguments it is an ordinary,
    /// unknown flag.
    #[test]
    fn only_the_flag_alone_asks() {
        let args = |a: &[&str]| a.iter().map(OsString::from).collect::<Vec<_>>();
        assert!(asks_for_build(&args(&["--build-info"])));
        assert!(!asks_for_build(&args(&[])));
        assert!(!asks_for_build(&args(&["--build-info", "--port", "1"])));
        assert!(!asks_for_build(&args(&["--build-infos"])));
    }

    #[test]
    fn this_binary_names_its_own_build_and_platform() {
        let b = BuildInfo::of(NPCD, Opens::default());
        assert_eq!(b.build(), this_build());
        assert_eq!(b.platform, platform());
        assert!(b.platform.contains('-'));
    }
}
