//! In-process JavaScript execution on the pure-Rust [`boa_engine`] VM.
//!
//! No external interpreter, no subprocess: code runs in a `boa_engine::Context`
//! created for the call. Because the VM is embedded, it is sandboxed by
//! construction — there is no network or process access, and its only
//! filesystem is the context's file store ([`super::files`]) — and runaway
//! scripts are bounded by the VM's loop / recursion limits (a synchronous eval
//! can't be wall-clock-interrupted, so the op-count limits are the guard).

use std::sync::Arc;

use boa_engine::gc::{Gc, GcRefCell};
use boa_engine::{js_string, Context, JsValue, NativeFunction, Source};

use super::files::Files;
use crate::grants::{Capability, Grants, NotPermitted};
use crate::state::VfsStore;

/// Loop-iteration ceiling before the VM aborts a script. Generous for real
/// computation (~sub-second to a few seconds) while still killing `while(true)`.
const LOOP_ITERATION_LIMIT: u64 = 100_000_000;
/// Call-stack depth ceiling — bounds unbounded recursion.
const RECURSION_LIMIT: usize = 2_000;

/// `console` implemented over two native sinks. Non-string arguments are
/// `JSON.stringify`'d so objects render as their contents rather than
/// `[object Object]`; `log`/`info`/`debug` go to stdout, `warn`/`error` to stderr.
const CONSOLE_PRELUDE: &str = r#"
globalThis.console = (() => {
    const fmt = (args) => args
        .map((x) => typeof x === 'string'
            ? x
            : (() => { try { return JSON.stringify(x); } catch (_) { return String(x); } })())
        .join(' ');
    return {
        log:   (...a) => __zend_out(fmt(a)),
        info:  (...a) => __zend_out(fmt(a)),
        debug: (...a) => __zend_out(fmt(a)),
        warn:  (...a) => __zend_err(fmt(a)),
        error: (...a) => __zend_err(fmt(a)),
    };
})();
"#;

/// The outcome of running a snippet: captured console streams, the final
/// expression value (when not `undefined`), and the thrown error (when it threw
/// or hit a VM limit).
pub struct JsOutcome {
    pub stdout: String,
    pub stderr: String,
    pub result: Option<String>,
    pub error: Option<String>,
}

/// Build a native function that appends its first argument (as a string) plus a
/// newline to `buf`. The closure captures nothing (the buffer arrives via the
/// captures slot), so it satisfies boa's `Copy` bound.
fn sink(buf: Gc<GcRefCell<String>>) -> NativeFunction {
    NativeFunction::from_copy_closure_with_captures(
        |_this, args, buf: &Gc<GcRefCell<String>>, ctx: &mut Context| {
            let line = match args.first() {
                Some(v) => v.to_string(ctx)?.to_std_string_escaped(),
                None => String::new(),
            };
            let mut b = buf.borrow_mut();
            b.push_str(&line);
            b.push('\n');
            Ok(JsValue::undefined())
        },
        buf,
    )
}

/// Run `code` in a fresh VM, after silently replaying `prelude` in order (a
/// session's prior snippets, or a run's injected globals — used to rebuild
/// state). Output produced by the prelude is discarded; only `code`'s console
/// output and final value are captured. The script reads and writes `store`
/// through its `vfs` global; the replayed prelude writes nothing.
///
/// Each prelude snippet, and `code`, runs under its own file-call budget
/// ([`Files::start`]) — so a replayed snippet meets its limit exactly where it
/// first did. The number of snippets a session replays is capped by the caller
/// ([`super::session_exec::MAX_SESSION_SNIPPETS`]).
///
/// Refused without [`Capability::Sandbox`]: no VM is created for a context
/// that may not run model-written code.
pub fn run_js(
    grants: Grants,
    store: &Arc<VfsStore>,
    prelude: &[String],
    code: &str,
) -> Result<JsOutcome, NotPermitted> {
    grants.require(Capability::Sandbox)?;
    let mut context = Context::default();
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(LOOP_ITERATION_LIMIT);
    context
        .runtime_limits_mut()
        .set_recursion_limit(RECURSION_LIMIT);

    let out: Gc<GcRefCell<String>> = Gc::new(GcRefCell::new(String::new()));
    let err: Gc<GcRefCell<String>> = Gc::new(GcRefCell::new(String::new()));
    let files = Files::new(Arc::clone(store));

    // Registering these builtins and evaluating the console prelude are internal
    // setup that cannot fail on a fresh context; surface a failure as an error
    // outcome rather than panicking the tool call.
    if let Err(e) = (|| -> boa_engine::JsResult<()> {
        context.register_global_callable(js_string!("__zend_out"), 1, sink(out.clone()))?;
        context.register_global_callable(js_string!("__zend_err"), 1, sink(err.clone()))?;
        context.eval(Source::from_bytes(CONSOLE_PRELUDE))?;
        files.register(&mut context)?;
        Ok(())
    })() {
        return Ok(JsOutcome {
            stdout: String::new(),
            stderr: String::new(),
            result: None,
            error: Some(format!("engine init failed: {e}")),
        });
    }

    // Replay session history to rebuild state, then drop whatever it printed so
    // only the new snippet's output surfaces.
    for snippet in prelude.iter().filter(|s| !s.is_empty()) {
        files.start(true);
        let _ = context.eval(Source::from_bytes(snippet));
    }
    out.borrow_mut().clear();
    err.borrow_mut().clear();

    files.start(false);
    let (result, error) = match context.eval(Source::from_bytes(code)) {
        Ok(value) => {
            let repr = if value.is_undefined() {
                None
            } else {
                Some(
                    value
                        .to_string(&mut context)
                        .map(|s| s.to_std_string_escaped())
                        .unwrap_or_default(),
                )
            };
            (repr, None)
        }
        Err(e) => (None, Some(e.to_string())),
    };

    let stdout = out.borrow().clone();
    let stderr = err.borrow().clone();
    Ok(JsOutcome {
        stdout,
        stderr,
        result,
        error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::code::files::{CALL_LIMIT, LIST_CALL_LIMIT, LIST_LIMIT};

    fn sandbox() -> Grants {
        Grants::NONE.with(Capability::Sandbox)
    }

    #[test]
    fn no_script_runs_without_the_sandbox_capability() {
        let store = Arc::new(VfsStore::new());
        for without in [Grants::NONE, Grants::NONE.with(Capability::Exec)] {
            assert_eq!(
                run_js(without, &store, &[], "1 + 1").err(),
                Some(NotPermitted(Capability::Sandbox))
            );
        }
        let ran = run_js(sandbox(), &store, &[], "1 + 1").unwrap();
        assert_eq!(ran.result.as_deref(), Some("2"));
    }

    /// **A script tests the file the model just wrote.** `vfs.read` returns
    /// the store's text (null when absent), `vfs.write` lands in the store,
    /// and `vfs.list` sees both.
    #[test]
    fn a_script_reads_and_writes_the_file_store() {
        let store = Arc::new(VfsStore::new());
        store
            .write(
                "scratch/add.js",
                "function add(a, b) { return a + b; }".to_string(),
            )
            .unwrap();
        let ran = run_js(
            sandbox(),
            &store,
            &[],
            "eval(vfs.read('scratch/add.js')); \
             vfs.write('scratch/out.txt', String(add(2, 3))); \
             String(vfs.read('scratch/none.js')) + '|' + JSON.stringify(vfs.list('scratch'))",
        )
        .unwrap();
        assert_eq!(ran.error, None);
        assert_eq!(
            ran.result.as_deref(),
            Some(r#"null|["scratch/add.js","scratch/out.txt"]"#)
        );
        assert_eq!(store.read("scratch/out.txt").unwrap().as_deref(), Some("5"));
    }

    /// **A module the model wrote can be required**, relative to the script
    /// (the workspace root) or to the requiring module, with `.js` optional —
    /// and a Node built-in is refused by name rather than as a bare failure.
    #[test]
    fn require_loads_a_module_from_the_file_store() {
        let store = Arc::new(VfsStore::new());
        store
            .write(
                "scratch/palindrome.js",
                "const clean = require('./util').clean;\n\
                 module.exports = (s) => { const c = clean(s); \
                 return c === [...c].reverse().join(''); };"
                    .to_string(),
            )
            .unwrap();
        store
            .write(
                "scratch/util.js",
                "exports.clean = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '');".to_string(),
            )
            .unwrap();
        let ran = run_js(
            sandbox(),
            &store,
            &[],
            "const isPalindrome = require('./scratch/palindrome.js'); \
             [isPalindrome('A man, a plan, a canal: Panama'), isPalindrome('hello')].join(',')",
        )
        .unwrap();
        assert_eq!(ran.error, None);
        assert_eq!(ran.result.as_deref(), Some("true,false"));

        let builtin = run_js(sandbox(), &store, &[], "require('fs')").unwrap();
        assert!(
            builtin
                .error
                .as_deref()
                .is_some_and(|e| e.contains("vfs.read")),
            "{:?}",
            builtin.error
        );
        let missing = run_js(sandbox(), &store, &[], "require('./nope')").unwrap();
        assert!(missing
            .error
            .as_deref()
            .is_some_and(|e| e.contains("nope.js")));
    }

    /// **Replayed history writes nothing.** A session's old `vfs.write`
    /// replays before every snippet; it must not overwrite what changed since.
    #[test]
    fn a_replayed_write_leaves_the_store_alone() {
        let store = Arc::new(VfsStore::new());
        store.write("notes.txt", "current".to_string()).unwrap();
        let history = ["vfs.write('notes.txt', 'stale');".to_string()];
        let ran = run_js(sandbox(), &store, &history, "1").unwrap();
        assert_eq!(ran.error, None);
        assert_eq!(store.read("notes.txt").unwrap().as_deref(), Some("current"));
    }

    /// **A snippet's file calls are bounded**: a write loop stops at
    /// [`CALL_LIMIT`] with an error naming the limit, and a list loop at
    /// [`LIST_CALL_LIMIT`] — while a replayed history spends none of it.
    #[test]
    fn file_calls_per_snippet_are_bounded() {
        let store = Arc::new(VfsStore::new());
        let ran = run_js(
            sandbox(),
            &store,
            &[],
            "for (let i = 0; i < 5000; i++) vfs.write('x/' + i, '')",
        )
        .unwrap();
        assert!(
            ran.error
                .as_deref()
                .is_some_and(|e| e.contains("call limit")),
            "{:?}",
            ran.error
        );
        assert_eq!(store.paths("x/").len(), CALL_LIMIT as usize);

        let lists = run_js(sandbox(), &store, &[], "for (;;) vfs.list('x/')").unwrap();
        assert!(
            lists.error.as_deref().is_some_and(
                |e| e.contains("vfs.list") && e.contains(&format!("at most {LIST_CALL_LIMIT}"))
            ),
            "{:?}",
            lists.error
        );

        // Each snippet has its own budget: a replayed one does not spend the
        // new snippet's.
        let reads = format!("for (let i = 0; i < {CALL_LIMIT}; i++) vfs.read('x/0');");
        let history = [reads.clone(), reads.clone()];
        let ran = run_js(sandbox(), &store, &history, &format!("{reads} 'ok'")).unwrap();
        assert_eq!(ran.error, None, "the replay spent the snippet's budget");
        assert_eq!(ran.result.as_deref(), Some("ok"));
    }

    /// **A replayed snippet meets its limit where it first did.** One that
    /// caught its limit error succeeded and joined the history; replayed with
    /// no limit, its loop would run to the VM's iteration cap, a workspace
    /// walk per turn.
    #[test]
    fn a_replayed_snippet_is_bounded_like_the_first_run() {
        let store = Arc::new(VfsStore::new());
        let caught = "let lists = 0; try { for (;;) { vfs.list(''); lists++; } } catch (e) {}";
        let first = run_js(sandbox(), &store, &[], caught).unwrap();
        assert_eq!(first.error, None, "the snippet caught its limit");
        let history = [caught.to_string()];
        let ran = run_js(sandbox(), &store, &history, "lists").unwrap();
        assert_eq!(ran.error, None);
        assert_eq!(
            ran.result.as_deref(),
            Some(LIST_CALL_LIMIT.to_string().as_str())
        );
    }

    /// One `vfs.list` returns at most [`LIST_LIMIT`] paths.
    #[test]
    fn a_listing_is_capped() {
        let store = Arc::new(VfsStore::new());
        for i in 0..LIST_LIMIT + 50 {
            store.write(&format!("d/{i:04}"), String::new()).unwrap();
        }
        let ran = run_js(sandbox(), &store, &[], "vfs.list('d').length").unwrap();
        assert_eq!(ran.result.as_deref(), Some(LIST_LIMIT.to_string().as_str()));
    }

    /// `require` resolves `..` against the requiring module and evaluates a
    /// module once, however many times it is required.
    #[test]
    fn require_resolves_parents_and_caches() {
        let store = Arc::new(VfsStore::new());
        store
            .write(
                "lib/count.js",
                "globalThis.loads = (globalThis.loads || 0) + 1; exports.n = 7;".into(),
            )
            .unwrap();
        store
            .write(
                "app/main.js",
                "module.exports = require('../lib/count').n;".into(),
            )
            .unwrap();
        let ran = run_js(
            sandbox(),
            &store,
            &[],
            "[require('./app/main'), require('./lib/count.js').n, globalThis.loads].join(',')",
        )
        .unwrap();
        assert_eq!(ran.error, None);
        assert_eq!(ran.result.as_deref(), Some("7,7,1"));
    }

    /// The store's guards hold inside a script: a protected path throws.
    #[test]
    fn a_script_cannot_reach_a_protected_path() {
        let store = Arc::new(VfsStore::new());
        let ran = run_js(
            sandbox(),
            &store,
            &[],
            "vfs.write('secrets/x.yaml', 'k: v')",
        )
        .unwrap();
        assert!(ran.error.is_some(), "the write was accepted");
        assert_eq!(store.read("secrets/x.yaml").ok().flatten(), None);
    }
}
