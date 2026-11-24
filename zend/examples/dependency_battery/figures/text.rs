//! Numbers as the figures print them: coordinates to one decimal, shares as
//! percentages, effects as signed points, counts with thousands separators,
//! and the ticks of an axis with their labels.

/// `x` to one decimal, without a trailing `.0` and never as `-0`.
pub fn n(x: f64) -> String {
    let r = (x * 10.0).round() / 10.0;
    let r = if r == 0.0 { 0.0 } else { r };
    let s = format!("{r:.1}");
    match s.strip_suffix(".0") {
        Some(whole) => whole.to_string(),
        None => s,
    }
}

/// `s` safe inside an SVG attribute or text node.
pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A share `0..1` as a percentage to one decimal.
pub fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// A share `0..1` as a whole percentage.
pub fn pct0(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

/// A difference of shares as signed points, the minus a true minus sign.
pub fn signed(x: f64) -> String {
    let p = (x * 1000.0).round() / 10.0;
    if p < 0.0 {
        format!("\u{2212}{:.1}", -p)
    } else {
        format!("+{p:.1}")
    }
}

/// `x` with a comma every three digits.
pub fn thousands(x: u64) -> String {
    let digits = x.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A p-value: three decimals, or scientific below a thousandth.
pub fn p_value(p: f64) -> String {
    if p >= 0.001 {
        format!("{p:.3}")
    } else {
        format!("{p:.1e}")
    }
}

/// `lo`, `lo + step`, … up to `hi`, each with its label.
pub fn ticks(lo: f64, hi: f64, step: f64, label: impl Fn(f64) -> String) -> Vec<(f64, String)> {
    let count = ((hi - lo) / step).round() as i64;
    (0..=count)
        .map(|i| lo + i as f64 * step)
        .map(|v| (v, label(v)))
        .collect()
}

/// `x` rounded up to a multiple of `step`.
pub fn ceil_to(x: f64, step: f64) -> f64 {
    (x / step - 1e-9).ceil() * step
}

/// `x` rounded down to a multiple of `step`.
pub fn floor_to(x: f64, step: f64) -> f64 {
    (x / step + 1e-9).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_print_to_one_decimal_without_a_trailing_zero() {
        assert_eq!(n(270.0), "270");
        assert_eq!(n(335.24), "335.2");
        assert_eq!(n(-0.04), "0");
        assert_eq!(n(-3.25), "-3.3");
    }

    #[test]
    fn numbers_print_as_the_figures_label_them() {
        assert_eq!(pct(0.371_527), "37.2%");
        assert_eq!(pct0(0.486), "49%");
        assert_eq!(signed(0.366_796), "+36.7");
        assert_eq!(signed(-0.031_372), "\u{2212}3.1");
        assert_eq!(thousands(7943), "7,943");
        assert_eq!(thousands(1_025_000), "1,025,000");
        assert_eq!(thousands(17), "17");
        assert_eq!(p_value(0.047_022), "0.047");
        assert_eq!(p_value(1.830_5e-15), "1.8e-15");
        assert_eq!(esc("a<b & \"c\">"), "a&lt;b &amp; &quot;c&quot;&gt;");
    }

    #[test]
    fn ticks_step_from_low_to_high() {
        assert_eq!(
            ticks(-10.0, 10.0, 10.0, n),
            vec![
                (-10.0, "-10".into()),
                (0.0, "0".into()),
                (10.0, "10".into())
            ]
        );
        assert_eq!((ceil_to(0.432, 0.1), floor_to(-0.027, 0.1)), (0.5, -0.1));
        assert_eq!((ceil_to(0.5, 0.1), floor_to(0.0, 0.1)), (0.5, 0.0));
    }
}
