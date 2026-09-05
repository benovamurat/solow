//! Tukey's Honestly Significant Difference (HSD) post-hoc test.
//!
//! Mirrors the reference `pairwise_tukeyhsd`. For all pairs of groups it
//! reports the mean differences, the studentized-range based confidence
//! intervals, the adjusted p-values, and the reject decisions at family-wise
//! error rate `alpha`. The critical value and p-values come from the
//! studentized-range distribution (see [`crate::srange`]).

use crate::srange::{srange_ppf, srange_sf};

/// Result of a pairwise Tukey HSD comparison.
#[derive(Debug, Clone)]
pub struct TukeyHsdResult {
    /// `(i, j)` group-index pairs (upper triangle, `i < j`).
    pub pairs: Vec<(usize, usize)>,
    /// Mean difference `mean_j − mean_i` for each pair.
    pub meandiffs: Vec<f64>,
    /// Standard error of each pairwise difference (studentized-range scaling).
    pub std_pairs: Vec<f64>,
    /// Lower and upper confidence bounds for each mean difference.
    pub confint: Vec<(f64, f64)>,
    /// Studentized-range adjusted p-value for each pair.
    pub pvalues: Vec<f64>,
    /// Reject the null of equal means at level `alpha`.
    pub reject: Vec<bool>,
    /// Critical value of the studentized range at level `alpha`.
    pub q_crit: f64,
    /// Total residual degrees of freedom `N − k`.
    pub df_total: f64,
    /// Pooled within-group variance (mean squared error).
    pub variance: f64,
}

/// All pairwise Tukey HSD comparisons.
///
/// `data` holds the response values and `groups` the integer group label of
/// each observation (labels need not be contiguous; distinct labels define the
/// groups, sorted ascending). `alpha` is the family-wise error rate.
pub fn pairwise_tukeyhsd(data: &[f64], groups: &[usize], alpha: f64) -> TukeyHsdResult {
    assert_eq!(data.len(), groups.len(), "data and groups length mismatch");

    // Collect the sorted unique group labels.
    let mut labels: Vec<usize> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let k = labels.len();

    // Per-group sums, counts.
    let mut nobs = vec![0.0_f64; k];
    let mut sums = vec![0.0_f64; k];
    let label_idx = |g: usize| labels.iter().position(|&l| l == g).unwrap();
    for (&v, &g) in data.iter().zip(groups) {
        let gi = label_idx(g);
        nobs[gi] += 1.0;
        sums[gi] += v;
    }
    let means: Vec<f64> = sums.iter().zip(&nobs).map(|(&s, &n)| s / n).collect();

    // Pooled within-group variance: SS_within / (N − k).
    let total_n: f64 = nobs.iter().sum();
    let mut ss_within = 0.0;
    for (&v, &g) in data.iter().zip(groups) {
        let gi = label_idx(g);
        let d = v - means[gi];
        ss_within += d * d;
    }
    let df_total = total_n - k as f64;
    let variance = ss_within / df_total;

    // All upper-triangle pairs.
    let mut pairs = Vec::new();
    let mut meandiffs = Vec::new();
    let mut std_pairs = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            pairs.push((i, j));
            // meandiffs use mean[j] − mean[i] (reference sign convention).
            meandiffs.push(means[j] - means[i]);
            // var of the studentized-range difference: var * (1/n_i + 1/n_j) / 2.
            let var_pair = variance * (1.0 / nobs[i] + 1.0 / nobs[j]) / 2.0;
            std_pairs.push(var_pair.sqrt());
        }
    }

    let kf = k as f64;
    let q_crit = srange_ppf(1.0 - alpha, kf, df_total);

    let mut pvalues = Vec::with_capacity(pairs.len());
    let mut reject = Vec::with_capacity(pairs.len());
    let mut confint = Vec::with_capacity(pairs.len());
    for idx in 0..pairs.len() {
        let st_range = meandiffs[idx].abs() / std_pairs[idx];
        let pval = srange_sf(st_range, kf, df_total);
        pvalues.push(pval);
        reject.push(st_range > q_crit);
        let crit_int = std_pairs[idx] * q_crit;
        confint.push((meandiffs[idx] - crit_int, meandiffs[idx] + crit_int));
    }

    TukeyHsdResult {
        pairs,
        meandiffs,
        std_pairs,
        confint,
        pvalues,
        reject,
        q_crit,
        df_total,
        variance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_groups_pairs() {
        let data = [1.0, 2.0, 3.0, 5.0, 6.0, 4.0, 8.0, 9.0, 7.0, 10.0];
        let groups = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2];
        let res = pairwise_tukeyhsd(&data, &groups, 0.05);
        assert_eq!(res.pairs.len(), 3);
        assert_eq!(res.df_total, 7.0);
        // Group means clearly separated -> all rejected.
        assert!(res.reject.iter().all(|&r| r));
    }
}
