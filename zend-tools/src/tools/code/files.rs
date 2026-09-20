//! The sandbox's only filesystem: a `vfs` global over the context's file store.
//!
//! A script sees exactly what the `file_*` tools see — the session's overlay in
//! the overlay tools modes, the workspace on disk in the one mode granted
//! `disk_write` — through three functions:
//!
//! | Call | Returns |
//! |---|---|
//! | `vfs.read(path)` | the file's text, or `null` when there is no such file |
//! | `vfs.write(path, text)` | the byte count written |
//! | `vfs.list(prefix)` | up to [`LIST_LIMIT`] paths under `prefix` |
//! | `require('./path')` | a CommonJS module from the store — see [`FILES_PRELUDE`] |
//!
//! Reading a file the model has just written is the point: `eval(vfs.read(
//! 'scratch/palindrome.js'))` puts it under test in the same call. The store's
//! own guards apply — a path under `secrets/` throws, as it refuses a tool.
//!
//! A session's history is replayed silently before each new snippet; while it
//! replays, `vfs.write` changes nothing, so an old write cannot overwrite a file
//! the model has changed since. Every snippet — replayed or new — runs under its
//! own call budget ([`CALL_LIMIT`], [`LIST_CALL_LIMIT`]).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use boa_engine::{
    js_string, Context, JsError, JsNativeError, JsResult, JsString, JsValue, NativeFunction, Source,
};

use crate::state::VfsStore;

/// Most paths one `vfs.list` returns.
pub const LIST_LIMIT: usize = 200;

/// Most `vfs.read`/`vfs.write` calls one snippet may make.
///
/// The interpreter's loop limit bounds iterations, not what each costs: a
/// loop of writes grows the store without end, and one of lists walks the
/// whole workspace per call, so a snippet within the loop limit could still
/// hold the file store and the turn for minutes. A test harness needs a few
/// dozen of these calls; a thousand is far past any honest use.
pub const CALL_LIMIT: u32 = 1_000;

/// Most `vfs.list` calls one snippet may make — each walks the workspace.
pub const LIST_CALL_LIMIT: u32 = 20;

/// The `vfs` object, over the native functions [`Files::register`] installs,
/// and a CommonJS `require` that loads modules from the file store.
///
/// `require` is what model-written JavaScript reaches for to test a module it
/// has just written (`module.exports = isPalindrome`), so it resolves the way
/// Node resolves a relative path — against the requiring module's directory,
/// the workspace root for the script itself — with `.js` tried when the path
/// has no extension. A module is evaluated once and cached. Node's built-in
/// modules (`fs`, `path`, …) are not here; asking for one throws, naming `vfs`.
const FILES_PRELUDE: &str = r#"
globalThis.vfs = Object.freeze({
    read:  (path) => __zend_vfs_read(String(path)),
    write: (path, text) => __zend_vfs_write(String(path), String(text)),
    list:  (prefix) => JSON.parse(__zend_vfs_list(String(prefix ?? ''))),
});
globalThis.require = (() => {
    const cache = {};
    const resolve = (base, spec) => {
        const parts = spec.startsWith('/') ? [] : base.split('/').filter((p) => p);
        for (const p of spec.split('/')) {
            if (p === '' || p === '.') continue;
            if (p === '..') parts.pop(); else parts.push(p);
        }
        return parts.join('/');
    };
    const dirOf = (path) => path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';
    const load = (base) => function require(spec) {
        spec = String(spec);
        if (!spec.startsWith('.') && !spec.startsWith('/')) {
            throw new Error(`Cannot find module '${spec}': Node's built-in and npm modules are not available in this sandbox — read files with vfs.read(path), write with vfs.write(path, text)`);
        }
        const path = resolve(base, spec);
        for (const candidate of [path, path + '.js']) {
            if (cache[candidate]) return cache[candidate].exports;
            const src = vfs.read(candidate);
            if (src === null) continue;
            const module = { exports: {} };
            cache[candidate] = module;
            new Function('module', 'exports', 'require', src)(module, module.exports, load(dirOf(candidate)));
            return module.exports;
        }
        throw new Error(`Cannot find module '${spec}' (looked for ${path} and ${path}.js in the file store)`);
    };
    return load('');
})();
"#;

/// A script's view of one file store.
pub struct Files {
    store: Arc<VfsStore>,
    writes: Rc<Cell<bool>>,
    budget: Rc<Budget>,
}

/// The calls a snippet has left — see [`CALL_LIMIT`] and [`LIST_CALL_LIMIT`].
/// Every snippet spends one, a replayed one included: replayed with no limit,
/// a snippet that caught its limit error would loop unbounded the next time.
struct Budget {
    calls: Cell<u32>,
    lists: Cell<u32>,
}

impl Budget {
    fn full() -> Self {
        Self {
            calls: Cell::new(CALL_LIMIT),
            lists: Cell::new(LIST_CALL_LIMIT),
        }
    }

    fn refill(&self) {
        self.calls.set(CALL_LIMIT);
        self.lists.set(LIST_CALL_LIMIT);
    }

    /// Spend one call from `left`, or throw naming `limit` when none is left.
    fn spend(&self, left: &Cell<u32>, what: &str, limit: u32) -> JsResult<()> {
        match left.get().checked_sub(1) {
            Some(rest) => {
                left.set(rest);
                Ok(())
            }
            None => Err(thrown(format!(
                "{what} call limit reached: one snippet may make at most {limit}"
            ))),
        }
    }
}

impl Files {
    /// Access to `store`, writable, with a full budget.
    pub fn new(store: Arc<VfsStore>) -> Self {
        Self {
            store,
            writes: Rc::new(Cell::new(true)),
            budget: Rc::new(Budget::full()),
        }
    }

    /// Begin a snippet: a fresh budget, and — when `replaying` a snippet that
    /// already ran — `vfs.write` reports the bytes it would write and writes
    /// nothing.
    pub fn start(&self, replaying: bool) {
        self.writes.set(!replaying);
        self.budget.refill();
    }

    /// Install `vfs` in `context`.
    pub fn register(&self, context: &mut Context) -> JsResult<()> {
        let store = Arc::clone(&self.store);
        let budget = Rc::clone(&self.budget);
        // SAFETY: the closure captures an `Arc<VfsStore>` and an `Rc<Budget>`
        // (plain cells) — no value the garbage collector traces — which is the
        // invariant `from_closure` requires.
        let read = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                budget.spend(&budget.calls, "vfs.read/vfs.write", CALL_LIMIT)?;
                let path = arg(args, 0, ctx)?;
                match store.read(&path) {
                    Ok(Some(text)) => Ok(JsValue::from(JsString::from(text.as_str()))),
                    Ok(None) => Ok(JsValue::null()),
                    Err(e) => Err(thrown(e.to_string())),
                }
            })
        };
        let store = Arc::clone(&self.store);
        let writes = Rc::clone(&self.writes);
        let budget = Rc::clone(&self.budget);
        // SAFETY: captures an `Arc<VfsStore>`, an `Rc<Cell<bool>>` and an
        // `Rc<Budget>`, none of which holds a traced value.
        let write = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                budget.spend(&budget.calls, "vfs.read/vfs.write", CALL_LIMIT)?;
                let path = arg(args, 0, ctx)?;
                let text = arg(args, 1, ctx)?;
                let bytes = text.len();
                if writes.get() {
                    store
                        .write(&path, text)
                        .map_err(|e| thrown(e.to_string()))?;
                }
                Ok(JsValue::from(bytes as f64))
            })
        };
        let store = Arc::clone(&self.store);
        let budget = Rc::clone(&self.budget);
        // SAFETY: captures an `Arc<VfsStore>` and an `Rc<Budget>` only.
        let list = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                budget.spend(&budget.lists, "vfs.list", LIST_CALL_LIMIT)?;
                let prefix = arg(args, 0, ctx)?;
                let paths: Vec<String> =
                    store.paths(&prefix).into_iter().take(LIST_LIMIT).collect();
                let json = serde_json::to_string(&paths).map_err(|e| thrown(e.to_string()))?;
                Ok(JsValue::from(JsString::from(json.as_str())))
            })
        };
        context.register_global_callable(js_string!("__zend_vfs_read"), 1, read)?;
        context.register_global_callable(js_string!("__zend_vfs_write"), 2, write)?;
        context.register_global_callable(js_string!("__zend_vfs_list"), 1, list)?;
        context.eval(Source::from_bytes(FILES_PRELUDE))?;
        Ok(())
    }
}

/// Argument `i` as a Rust string; a missing argument is the empty string.
fn arg(args: &[JsValue], i: usize, ctx: &mut Context) -> JsResult<String> {
    match args.get(i) {
        Some(v) => Ok(v.to_string(ctx)?.to_std_string_escaped()),
        None => Ok(String::new()),
    }
}

/// A JS `Error` carrying `message`, thrown into the script.
fn thrown(message: String) -> JsError {
    JsNativeError::error().with_message(message).into()
}
