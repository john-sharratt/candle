//! Prints the per-session timing and the span's tenant census, so a table row
//! can be read against the tenants that produced it.
//!
//! The census is taken from the same publish as the span figure it breaks down,
//! because two moments read apart look like a tenant holding ground nobody
//! counted. A difference between the two is not always a defect: a guest
//! model's ground is held by a span holder that is not a slot arena, so the line
//! names that residual rather than asserting an identity that holds only without
//! a guest. Blocked submit times are printed in session order, since the turns
//! behind a blocked submit cannot reach the scheduler until it returns.

use super::*;

/// How long each submit held its caller, in session order — the turns behind a
/// blocked submit cannot even reach the scheduler until it returns.
pub(super) fn print_submit_blocked(blocked: &[Duration]) {
    let ms: Vec<String> = blocked
        .iter()
        .map(|d| format!("{:.1}", d.as_secs_f64() * 1000.0))
        .collect();
    println!("  submit blocked (ms, per session): [{}]", ms.join(", "));
}

/// A batch's two windows, as the gate's table would read them.
pub(super) fn print_timing(timing: &BatchTiming, sequences: usize) {
    println!(
        "  prefill: {} tok in {:.2}s  = {:.1} t/s",
        timing.prefill_tokens,
        timing.prefill_s,
        timing.prefill_tps(),
    );
    println!(
        "  decode:  {} tok in {:.2}s  = {:.1} t/s   ({sequences} sequences, {} tokens streamed)",
        timing.decode_tokens,
        timing.decode_s,
        timing.decode_tps(),
        timing.streamed_tokens,
    );
    println!(
        "  complete: last reply {:.2}s after the last first token ({:.2}s past the last token)",
        timing.complete_s,
        timing.complete_s - timing.decode_s,
    );
}

/// One raw reading, whatever the wave is doing.
/// Break the `span` column down by tenant, **for the sample that column came
/// from**.
///
/// **`span` on its own cannot be acted on.** It is every span tenant's regions as
/// one number, counted as legitimately in use — so a tenant standing on thirty
/// mostly-empty arenas reads exactly like one standing on thirty full ones, and a
/// compaction that reclaimed nothing reads exactly like one that reclaimed
/// everything. `use%` is what says whether a tenant has room to give back, and
/// `pools` is what says whether a packing walk could ever recover it: a tenant
/// spread over several strides pays at least one region per stride however small
/// its slots are, and no walk recovers that — only giving it fewer strides does.
///
/// **Taken from the report's own publish, not sampled here.** The first version
/// of this called `arena_census` at print time, minutes after the phase-B row it
/// was meant to explain, and the two disagreed by an order of magnitude — 18
/// tenant regions against a `span` of 219. That reads exactly like an uncounted
/// tenant holding 201 regions, and it is nothing but two moments compared as one.
/// The rows now arrive in `KvSection` beside `span_regions`, so the two are one
/// moment and the difference below is real rather than skew.
///
/// **A difference is not necessarily a defect.** `span_regions` counts every
/// `SpanRegion` handle, and the slot arenas are not its only holder: a guest
/// model's ground claims them too, and anything else that takes one in future
/// will. So the line reports the residual and names what it could be, rather than
/// asserting an identity that only holds while no guest is resident.
pub(super) fn print_tenant_census(label: &str) {
    let Some((report, _)) = memory_report::latest() else {
        return;
    };
    let rows = &report.kv.span_tenants;
    let span = report.kv.span_regions;
    println!("\n=== Span tenants ({label}) ===\n");
    println!(
        "  {:<28} {:>7} {:>9} {:>6} {:>6}",
        "tenant", "regions", "held MiB", "use%", "pools"
    );
    for r in rows {
        let reserved = r.regions * REGION_BYTES;
        let use_pct = (r.held_bytes * 100).checked_div(reserved).unwrap_or(0);
        println!(
            "  {:<28} {:>7} {:>9} {:>6} {:>6}",
            r.tenant,
            r.regions,
            r.held_bytes >> 20,
            use_pct,
            r.pools,
        );
    }
    let total: usize = rows.iter().map(|r| r.regions).sum();
    // The reconciliation is the point of printing this at all: the residual is
    // ground held by a `SpanRegion` holder that is not a slot arena — a guest
    // model's ground today — and naming it is what keeps it from being read as an
    // uncounted tenant or as sampling skew.
    println!(
        "  {:<28} {:>7}   ({})",
        "total",
        total,
        match span.checked_sub(total) {
            Some(0) => format!("= span {span}"),
            Some(rest) => format!("span {span}; {rest} in non-arena holders (guest ground)"),
            // The census is taken under the pools' lock and `span_regions` under
            // the region pool's, so a release between them can leave the rows
            // ahead. Worth saying, not worth calling a fault.
            None => format!("span {span}; census ahead by {}", total - span),
        },
    );
}
