//! In-process JavaScript execution on the pure-Rust [`boa_engine`] VM.
//!
//! No external interpreter, no subprocess: code runs in a `boa_engine::Context`
//! created for the call. Because the VM is embedded, it is sandboxed by
//! construction — there is no filesystem, network, or process access unless the
//! host explicitly grants it, and runaway scripts are bounded by the VM's loop /
//! recursion limits (a synchronous eval can't be wall-clock-interrupted, so the
//! op-count limits are the guard).

use boa_engine::gc::{Gc, GcRefCell};
use boa_engine::{js_string, Context, JsValue, NativeFunction, Source};

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

/// Run `code` in a fresh VM, optionally after silently replaying `prelude`
/// (a session's accumulated prior source — used to rebuild variable/function
/// state). Output produced by the prelude is discarded; only `code`'s console
/// output and final value are captured.
pub fn run_js(prelude: &str, code: &str) -> JsOutcome {
    let mut context = Context::default();
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(LOOP_ITERATION_LIMIT);
    context
        .runtime_limits_mut()
        .set_recursion_limit(RECURSION_LIMIT);

    let out: Gc<GcRefCell<String>> = Gc::new(GcRefCell::new(String::new()));
    let err: Gc<GcRefCell<String>> = Gc::new(GcRefCell::new(String::new()));

    // Registering these builtins and evaluating the console prelude are internal
    // setup that cannot fail on a fresh context; surface a failure as an error
    // outcome rather than panicking the tool call.
    if let Err(e) = (|| -> boa_engine::JsResult<()> {
        context.register_global_callable(js_string!("__zend_out"), 1, sink(out.clone()))?;
        context.register_global_callable(js_string!("__zend_err"), 1, sink(err.clone()))?;
        context.eval(Source::from_bytes(CONSOLE_PRELUDE))?;
        Ok(())
    })() {
        return JsOutcome {
            stdout: String::new(),
            stderr: String::new(),
            result: None,
            error: Some(format!("engine init failed: {e}")),
        };
    }

    // Replay session history to rebuild state, then drop whatever it printed so
    // only the new snippet's output surfaces.
    if !prelude.is_empty() {
        let _ = context.eval(Source::from_bytes(prelude));
        out.borrow_mut().clear();
        err.borrow_mut().clear();
    }

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
    JsOutcome {
        stdout,
        stderr,
        result,
        error,
    }
}
