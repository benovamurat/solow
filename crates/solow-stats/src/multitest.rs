//! Multiple-hypothesis-testing p-value corrections.

/// Correction method for [`multipletests`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiTestMethod {
    /// Bonferroni one-step correction.
    Bonferroni,
    /// Benjamini–Hochberg FDR step-up procedure (independent / positive corr.).
    FdrBh,
    /// Holm step-down (Bonferroni) procedure.
    Holm,
}

/// Result of [`multipletests`]: rejection flags and adjusted p-values, both in
/// the original input order.
#[derive(Debug, Clone)]
pub struct MultiTestResult {
    /// `true` where the corresponding null hypothesis is rejected at `alpha`.
    pub reject: Vec<bool>,
    /// Corrected (adjusted) p-values, clamped to `[0, 1]`.
    pub pvals_corrected: Vec<f64>,
}

/// Adjust a set of p-values for multiple testing.
///
/// Returns rejection decisions at family-wise / false-discovery level `alpha`
/// and the corrected p-values, in the original order of `pvals`. Mirrors the
/// reference `multipletests` for the `bonferroni`, `fdr_bh`, and `holm`
/// methods.
pub fn multipletests(pvals: &[f64], alpha: f64, method: MultiTestMethod) -> MultiTestResult {
    let ntests = pvals.len();
    if ntests == 0 {
        return MultiTestResult {
            reject: vec![],
            pvals_corrected: vec![],
        };
    }

    match method {
        MultiTestMethod::Bonferroni => {
            // No sorting required; operates in input order directly.
            let nt = ntests as f64;
            let alphac_bonf = alpha / nt;
            let reject = pvals.iter().map(|&p| p <= alphac_bonf).collect();
            let pvals_corrected = pvals.iter().map(|&p| (p * nt).min(1.0)).collect();
            MultiTestResult {
                reject,
                pvals_corrected,
            }
        }
        MultiTestMethod::Holm => holm(pvals, alpha),
        MultiTestMethod::FdrBh => fdr_bh(pvals, alpha),
    }
}

/// Argsort indices of `pvals` in ascending order (stable, matching numpy's
/// default quicksort-stable behaviour for distinct keys).
fn argsort(pvals: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..pvals.len()).collect();
    idx.sort_by(|&a, &b| pvals[a].total_cmp(&pvals[b]));
    idx
}

/// Scatter `sorted` back to the original order given the sort indices.
fn unsort(sorted: &[f64], sortind: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; sorted.len()];
    for (k, &orig) in sortind.iter().enumerate() {
        out[orig] = sorted[k];
    }
    out
}

fn unsort_bool(sorted: &[bool], sortind: &[usize]) -> Vec<bool> {
    let mut out = vec![false; sorted.len()];
    for (k, &orig) in sortind.iter().enumerate() {
        out[orig] = sorted[k];
    }
    out
}

fn holm(pvals: &[f64], alpha: f64) -> MultiTestResult {
    let ntests = pvals.len();
    let sortind = argsort(pvals);
    let sorted: Vec<f64> = sortind.iter().map(|&i| pvals[i]).collect();

    // notreject_i = p_i > alpha / (ntests - i)
    let mut notreject: Vec<bool> = (0..ntests)
        .map(|i| sorted[i] > alpha / (ntests - i) as f64)
        .collect();
    // From the first non-rejection onward, force non-rejection.
    let notrejectmin = notreject.iter().position(|&b| b).unwrap_or(ntests);
    for nr in notreject.iter_mut().skip(notrejectmin) {
        *nr = true;
    }
    let reject_sorted: Vec<bool> = notreject.iter().map(|&b| !b).collect();

    // pvals_corrected = cummax(p_i * (ntests - i)), then clamp to 1.
    let mut running = f64::NEG_INFINITY;
    let mut corrected: Vec<f64> = (0..ntests)
        .map(|i| {
            let v = sorted[i] * (ntests - i) as f64;
            running = running.max(v);
            running
        })
        .collect();
    for c in corrected.iter_mut() {
        if *c > 1.0 {
            *c = 1.0;
        }
    }

    MultiTestResult {
        reject: unsort_bool(&reject_sorted, &sortind),
        pvals_corrected: unsort(&corrected, &sortind),
    }
}

fn fdr_bh(pvals: &[f64], alpha: f64) -> MultiTestResult {
    let ntests = pvals.len();
    let sortind = argsort(pvals);
    let sorted: Vec<f64> = sortind.iter().map(|&i| pvals[i]).collect();
    let nf = ntests as f64;

    // ecdffactor_i = (i+1)/ntests
    let ecdf: Vec<f64> = (0..ntests).map(|i| (i + 1) as f64 / nf).collect();

    // reject where p_i <= ecdf_i * alpha; then fill below the last rejection.
    let mut reject_sorted: Vec<bool> = (0..ntests).map(|i| sorted[i] <= ecdf[i] * alpha).collect();
    if let Some(rejectmax) = reject_sorted.iter().rposition(|&b| b) {
        for r in reject_sorted.iter_mut().take(rejectmax) {
            *r = true;
        }
    }

    // corrected = reverse-cumulative-min(p_i / ecdf_i), clamped to 1.
    let raw: Vec<f64> = (0..ntests).map(|i| sorted[i] / ecdf[i]).collect();
    let mut corrected = vec![0.0; ntests];
    let mut running = f64::INFINITY;
    for i in (0..ntests).rev() {
        running = running.min(raw[i]);
        corrected[i] = running.min(1.0);
    }

    MultiTestResult {
        reject: unsort_bool(&reject_sorted, &sortind),
        pvals_corrected: unsort(&corrected, &sortind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonferroni_scales_and_clamps() {
        let r = multipletests(&[0.01, 0.2, 0.6], 0.05, MultiTestMethod::Bonferroni);
        assert!((r.pvals_corrected[0] - 0.03).abs() < 1e-12);
        assert!((r.pvals_corrected[2] - 1.0).abs() < 1e-12); // 1.8 clamped
        assert_eq!(r.reject, vec![true, false, false]);
    }

    #[test]
    fn fdr_bh_monotone() {
        let r = multipletests(&[0.001, 0.01, 0.03, 0.5], 0.05, MultiTestMethod::FdrBh);
        // Corrected p-values are non-decreasing in the original (already sorted) order.
        for w in r.pvals_corrected.windows(2) {
            assert!(w[0] <= w[1] + 1e-12);
        }
    }
}
