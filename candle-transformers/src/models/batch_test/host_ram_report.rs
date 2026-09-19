//! This process's host RAM at the end of each config's decode: resident and
//! committed bytes by what backs them, and the allocations holding the most.
//!
//! The pinned-RAM report says what the process page-locked; this says what the
//! rest of its resident memory is. Page-locked host memory is held by the
//! driver outside the working set, so it shows here as committed private
//! memory that is not resident — the pinned gauge is printed beside it so the
//! two can be told apart.

use candle::vram::host_pinned_bytes;
use candle::vram::process_ram::{ProcessRam, RegionKind};

const MIB: f64 = 1024.0 * 1024.0;

/// How many allocations the detail list names.
const LARGEST: usize = 16;

fn mib(b: u64) -> String {
    format!("{:.0}", b as f64 / MIB)
}

/// A power-of-two size in the largest unit that keeps it whole.
fn size_label(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{} GiB", b >> 30),
        b if b >= 1 << 20 => format!("{} MiB", b >> 20),
        b if b >= 1 << 10 => format!("{} KiB", b >> 10),
        b => format!("{b} B"),
    }
}

/// The per-kind rows for one capture.
fn rows(p: &ProcessRam) -> [(&'static str, String); 7] {
    let image = p.total(RegionKind::Image);
    let mapped = p.total(RegionKind::Mapped);
    let private = p.total(RegionKind::Private);
    [
        (
            "resident, all",
            mib(image.resident + mapped.resident + private.resident),
        ),
        ("  image (exe + DLLs)", mib(image.resident)),
        ("  mapped files", mib(mapped.resident)),
        ("  private", mib(private.resident)),
        ("committed, private", mib(private.committed)),
        ("committed, mapped", mib(mapped.committed)),
        ("committed, image", mib(image.committed)),
    ]
}

/// Size buckets holding at least this much resident memory are listed.
const BUCKET_FLOOR: u64 = 64 << 20;

/// One line: the process's private resident memory at `label`, and every size
/// bucket holding at least [`BUCKET_FLOOR`] of it.
pub fn print_host_ram_line(label: &str) {
    let Some(p) = ProcessRam::capture() else {
        return;
    };
    let private = p.total(RegionKind::Private);
    let buckets: Vec<String> = p
        .private_by_size()
        .into_iter()
        .filter(|b| b.resident >= BUCKET_FLOOR)
        .map(|b| format!("≤{} ×{}: {}", size_label(b.upper), b.count, mib(b.resident)))
        .collect();
    println!(
        "  [host ram] {label}: private resident {} MiB of {} MiB committed | {}",
        mib(private.resident),
        mib(private.committed),
        buckets.join(" | ")
    );
}

/// Print one column per `(label, capture)`, then the largest allocations of
/// the last capture.
pub fn print_host_ram(columns: &[(String, Option<ProcessRam>)]) {
    let Some(last) = columns.iter().rev().find_map(|(_, p)| p.as_ref()) else {
        return;
    };
    const LABEL_W: usize = 22;
    let width = columns
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(0)
        .max(12);
    println!("\n=== Host RAM, this process, at the end of each decode (MiB) ===");
    println!(
        "  pinned now (driver-locked, outside the working set): {} MiB",
        mib(host_pinned_bytes())
    );
    let mut header = format!("{:<LABEL_W$}", "");
    for (label, _) in columns {
        header.push_str(&format!(" │ {label:>width$}"));
    }
    println!("{header}");
    let table: Vec<Option<[(&str, String); 7]>> =
        columns.iter().map(|(_, p)| p.as_ref().map(rows)).collect();
    for i in 0..7 {
        let name = rows(last)[i].0;
        let mut line = format!("{name:<LABEL_W$}");
        for r in &table {
            let v = r.as_ref().map_or("-".to_string(), |r| r[i].1.clone());
            line.push_str(&format!(" │ {v:>width$}"));
        }
        println!("{line}");
    }
    println!("  private allocations by committed size (last column):");
    println!(
        "    {:>12} {:>7} {:>10} {:>10}",
        "size ≤", "count", "resident", "committed"
    );
    for b in last.private_by_size() {
        println!(
            "    {:>12} {:>7} {:>10} {:>10}",
            size_label(b.upper),
            b.count,
            mib(b.resident),
            mib(b.committed)
        );
    }
    println!("  largest resident allocations (last column):");
    println!(
        "    {:>18}  {:<8} {:>10} {:>10}  name",
        "base", "kind", "resident", "committed"
    );
    for a in last.largest_resident(LARGEST) {
        println!(
            "    {:>#18x}  {:<8} {:>10} {:>10}  {}",
            a.alloc_base,
            format!("{:?}", a.kind),
            mib(a.resident),
            mib(a.committed),
            a.name.as_deref().unwrap_or("-")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::vram::process_ram::RegionRecord;

    #[test]
    fn size_labels_use_the_largest_whole_unit() {
        assert_eq!(size_label(16 << 20), "16 MiB");
        assert_eq!(size_label(2 << 30), "2 GiB");
        assert_eq!(size_label(4096), "4 KiB");
        assert_eq!(size_label(512), "512 B");
    }

    #[test]
    fn rows_sum_resident_across_kinds() {
        let p = ProcessRam::from_records([
            RegionRecord {
                alloc_base: 0x1000,
                kind: RegionKind::Image,
                committed: 3 << 20,
                resident: 2 << 20,
            },
            RegionRecord {
                alloc_base: 0x9000,
                kind: RegionKind::Private,
                committed: 10 << 20,
                resident: 4 << 20,
            },
        ]);
        let r = rows(&p);
        assert_eq!(r[0], ("resident, all", "6".to_string()));
        assert_eq!(r[1].1, "2");
        assert_eq!(r[3].1, "4");
        assert_eq!(r[4].1, "10");
    }
}
