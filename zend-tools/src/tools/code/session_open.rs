//! code_session_open tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::state::sessions::{CodeEntry, SessionMeta};
use crate::{RegisteredTool, Tool, ToolContext};
use super::{now, CodeError, PYTHON_REPL, NODE_REPL};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionOpenReq {
    #[validate(length(min = 1))]
    pub language: String,
    pub image: Option<String>,
}

#[derive(Serialize)]
pub struct SessionOpenResp {
    pub session_id: String,
    pub language: String,
}

pub struct CodeSessionOpen;

impl Tool for CodeSessionOpen {
    const NAME: &'static str = "code_session_open";
    const DESCRIPTION: &'static str =
        "Open a persistent code execution session. Supported languages: python, python3, \
         bash, sh, javascript, node. State is preserved between exec calls. \
         Close with code_session_close when done.";
    type Request = SessionOpenReq;
    type Response = SessionOpenResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, req: SessionOpenReq) -> Result<SessionOpenResp, CodeError> {
        let lang = req.language.as_str();

        let (interpreter, args, repl_source, ext): (&str, Vec<&str>, Option<&str>, &str) =
            match lang {
                "python" | "python3" => {
                    let py = if cfg!(windows) { "python.exe" } else { "python3" };
                    (py, vec![], Some(PYTHON_REPL), "py")
                }
                "javascript" | "js" | "node" => {
                    let node = if cfg!(windows) { "node.exe" } else { "node" };
                    (node, vec![], Some(NODE_REPL), "js")
                }
                "bash" => ("bash", vec![], None, "sh"),
                "sh" => ("sh", vec![], None, "sh"),
                other => return Err(CodeError::InterpreterNotFound(other.to_string())),
            };

        let temp_path = if let Some(src) = repl_source {
            let p = std::env::temp_dir()
                .join(format!("zend_repl_{}.{}", Uuid::new_v4().simple(), ext));
            std::fs::write(&p, src)
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            Some(p)
        } else {
            None
        };

        let mut cmd = std::process::Command::new(interpreter);
        for a in &args {
            cmd.arg(a);
        }
        if let Some(ref p) = temp_path {
            cmd.arg(p);
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CodeError::InterpreterNotFound(lang.to_string())
            } else {
                CodeError::ExecutionFailed(e.to_string())
            }
        })?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stdout_reader = std::io::BufReader::new(stdout);

        let sid = format!("sess_{}", Uuid::new_v4().simple());
        ctx.sessions.insert_code(CodeEntry {
            meta: SessionMeta {
                session_id: sid.clone(),
                opened_at: now(),
                last_activity: now(),
                alive: true,
            },
            language: lang.to_string(),
            child,
            stdin,
            stdout_reader,
            _temp_script: temp_path,
        });

        Ok(SessionOpenResp {
            session_id: sid,
            language: lang.to_string(),
        })
    }
}

pub const CODE_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<CodeSessionOpen>();
