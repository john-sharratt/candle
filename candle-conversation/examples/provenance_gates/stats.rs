//! The summary statistics the gates report.

/// Spearman's rank correlation of two equal-length series.
pub fn spearman(a: &[usize], b: &[usize]) -> f32 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f32;
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let cov: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

/// Each value's rank, ties sharing their mean rank.
fn ranks(v: &[usize]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by_key(|&i| v[i]);
    let mut r = vec![0.0f32; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f32 / 2.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Recall at `k` and the mean reciprocal rank of 1-based `ranks`.
pub fn recall(ranks: &[usize], k: usize) -> f32 {
    ranks.iter().filter(|&&x| x <= k).count() as f32 / ranks.len().max(1) as f32
}

pub fn mrr(ranks: &[usize]) -> f32 {
    ranks.iter().map(|&x| 1.0 / x as f32).sum::<f32>() / ranks.len().max(1) as f32
}

/// The Gini coefficient of `counts`: 0 when every one is equal, `1 - 1/n`
/// when one holds everything.
pub fn gini(counts: &[usize]) -> f32 {
    let mut v: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    v.sort_by(f64::total_cmp);
    let n = v.len() as f64;
    let total: f64 = v.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let weighted: f64 = v
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64 + 1.0) * x)
        .sum();
    ((2.0 * weighted) / (n * total) - (n + 1.0) / n) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_is_zero_when_equal_and_three_quarters_when_one_of_four_takes_all() {
        assert_eq!(gini(&[3, 3, 3, 3]), 0.0);
        assert!((gini(&[0, 0, 0, 12]) - 0.75).abs() < 1e-6);
    }
}
