//! `service_cancel {deploy}`: stop a deploy (`docs/build_deploy_tools.md` §4)
//! — its build while that runs, or its trial, which keep rolls back and
//! records as cancelled. A deploy that has ended is left as it ended.

use axum::http::Method;
use serde_json::{json, Value};

use super::status::render_deploy;
use super::{keep, resolve, Stage, KEEP_CANCEL};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::repo_job::{lost, Handle, NO_AGENTS};
use crate::mcp::tools::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "service_cancel";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Stop a deploy. While it builds, the build job is killed and nothing is tried. While \
            keep tries it, keep stops the trial, puts the previous binary back and starts it, and records \
            the trial as cancelled — the machine runs what it ran before. A deploy that has ended — proven, \
            rolled back, failed — is left as it is, and the answer says how it ended; undoing a proven \
            deploy is another deploy. `deploy` is the handle `service_deploy` answered. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deploy": {"type": "string", "minLength": 18, "maxLength": 64, "description": "The deploy's handle, `<node>:<id>`."}
            },
            "required": ["deploy"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let h = match string(args, "deploy").and_then(Handle::parse) {
        Ok(h) => h,
        Err(e) => return Answer::error(e.replace("not a job", "not a deploy")),
    };
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let d = match resolve(call, &agents, &h, now_ms()).await {
        Ok(d) => d,
        Err(e) => return Answer::error(e),
    };
    match &d.stage {
        Stage::Building => match agents.cancel(h.at(&agents), &h.id).await {
            Ok(_) => Answer::ok(format!(
                "Cancelled deploy {h}: its build of {} on {} is stopped, and nothing will be tried.",
                d.service, d.host
            )),
            Err(e) => Answer::error(lost(&h, &e)),
        },
        Stage::Trying(_) => {
            let body = json!({"service": d.service});
            match keep(call, Method::POST, KEEP_CANCEL, &d.host, Some(&body)).await {
                Ok(_) => Answer::ok(format!(
                    "Cancelled deploy {h}: keep on {} is stopping the trial of {} and putting the \
                     previous binary back. `service_status {{deploy: \"{h}\", wait_s: 60}}` says \
                     when it is done.",
                    d.host, d.service
                )),
                Err(e) => Answer::error(e),
            }
        }
        Stage::Built { stale: false, .. } => Answer::error(format!(
            "Deploy {h} has just built, and keep is being asked for its trial: there is nothing to \
             stop for a moment. Call again in a few seconds."
        )),
        _ => Answer::ok(format!(
            "Nothing to cancel: it has ended.\n{}",
            render_deploy(&d, now_ms(), None)
        )),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}
