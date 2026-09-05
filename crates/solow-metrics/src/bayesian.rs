//! Bayesian model comparison via WAIC and PSIS-LOO.
//!
//! Both take a **log-likelihood matrix** `log_lik` of shape
//! `(n_samples × n_observations)` — the log likelihood of every
//! observation `i` under every posterior sample `s`. This is the same
//! input shape as the R `loo` and Python `arviz.loo` / `arviz.waic`
//! functions, which makes side-by-side comparison trivial.
//!
//! * [`waic`] — Watanabe-Akaike Information Criterion (Watanabe 2010).
//!   Fast, closed-form; slightly biased when a few observations dominate
//!   the effective sample size.
//! * [`psis_loo`] — Pareto-smoothed importance sampling leave-one-out
//!   (Vehtari, Gelman & Gabry, 2017). Correctly accounts for
//!   heavy-tailed importance weights; the recommended default over WAIC.
//!
//! Both return a per-observation elpd vector plus its total and standard
//! error, together with the effective number of parameters `p_waic` /
//! `p_loo`.

use ndarray::ArrayView2;
use solow_core::{Error, Result};

/// Watanabe-Akaike Information Criterion result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct WaicResult {
    /// Estimated log pointwise predictive density (higher = better).
    pub elpd: f64,
    /// Standard error of `elpd` via the sample variance of the
    /// per-observation elpd.
    pub elpd_se: f64,
    /// Effective number of parameters — the sum of the posterior-sample
    /// variance of `log p(yᵢ | θ)`.
    pub p_waic: f64,
    /// WAIC on the deviance scale, `-2 · elpd`.
    pub waic: f64,
    /// Per-observation elpd contributions.
    pub pointwise: Vec<f64>,
}

/// WAIC from a `(n_samples, n_observations)` log-likelihood matrix.
///
/// Uses the log-sum-exp form throughout, so probabilities under high-
/// dimensional posteriors don't underflow.
pub fn waic(log_lik: ArrayView2<'_, f64>) -> Result<WaicResult> {
    let (s, n) = (log_lik.nrows(), log_lik.ncols());
    if s < 2 || n < 1 {
        return Err(Error::Value(format!(
            "waic: log_lik must have ≥ 2 samples and ≥ 1 observation (got {s} × {n})"
        )));
    }
    let mut pointwise = Vec::with_capacity(n);
    let mut p_waic = 0.0_f64;
    for j in 0..n {
        // log(mean(exp(log_lik))) via log-sum-exp.
        let col: Vec<f64> = (0..s).map(|i| log_lik[[i, j]]).collect();
        let m = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return Err(Error::Value(format!(
                "waic: log-likelihoods for observation {j} are all -inf"
            )));
        }
        let sum_exp: f64 = col.iter().map(|v| (v - m).exp()).sum();
        let lppd_j = m + (sum_exp / s as f64).ln();
        // Posterior variance of log p(y_j | theta).
        let mean_ll: f64 = col.iter().sum::<f64>() / s as f64;
        let var_ll: f64 = col.iter().map(|v| (v - mean_ll).powi(2)).sum::<f64>() / (s as f64 - 1.0);
        p_waic += var_ll;
        pointwise.push(lppd_j - var_ll);
    }
    let elpd: f64 = pointwise.iter().sum();
    let mean_p: f64 = elpd / n as f64;
    let var_p: f64 =
        pointwise.iter().map(|v| (v - mean_p).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let elpd_se = (n as f64 * var_p).sqrt();
    Ok(WaicResult {
        elpd,
        elpd_se,
        p_waic,
        waic: -2.0 * elpd,
        pointwise,
    })
}

/// PSIS-LOO result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PsisLooResult {
    /// Expected log pointwise predictive density (higher = better).
    pub elpd: f64,
    /// Standard error of `elpd`.
    pub elpd_se: f64,
    /// Effective number of parameters `p_loo`.
    pub p_loo: f64,
    /// LOO on the deviance scale, `-2 · elpd`.
    pub looic: f64,
    /// Per-observation elpd contributions.
    pub pointwise: Vec<f64>,
    /// Per-observation Pareto tail-shape `k̂`. Any value `> 0.7` marks
    /// an observation whose LOO estimate is unreliable and should be
    /// resampled by explicit leave-one-out refit.
    pub pareto_k: Vec<f64>,
}

/// Pareto-smoothed importance sampling LOO (Vehtari, Gelman & Gabry, 2017).
///
/// For each observation `j`, uses importance weights `wᵢⱼ ∝ 1 / p(yⱼ | θᵢ)`
/// on the posterior samples, fits a generalised Pareto to the top 20% of
/// weights, replaces those tail weights by their smoothed order statistics,
/// then estimates `elpd_j = log(mean(w̃ⱼ · p(yⱼ | θ)))`. The Pareto shape
/// `k̂` is a per-observation reliability diagnostic — the classical rule
/// of thumb is that `k̂ < 0.5` is fine, `0.5 ≤ k̂ ≤ 0.7` is a warning, and
/// `k̂ > 0.7` means the LOO estimate for that observation should not be
/// trusted.
pub fn psis_loo(log_lik: ArrayView2<'_, f64>) -> Result<PsisLooResult> {
    let (s, n) = (log_lik.nrows(), log_lik.ncols());
    if s < 20 || n < 1 {
        return Err(Error::Value(format!(
            "psis_loo: log_lik must have ≥ 20 samples and ≥ 1 observation (got {s} × {n})"
        )));
    }
    let mut pointwise = Vec::with_capacity(n);
    let mut pareto_k = Vec::with_capacity(n);
    let mut p_loo = 0.0_f64;
    for j in 0..n {
        // Raw importance log-weights: -log p(y_j | theta_i) = -log_lik[i, j].
        let raw: Vec<f64> = (0..s).map(|i| -log_lik[[i, j]]).collect();
        // Smooth the top 20% tail with a generalised Pareto fit.
        let (smoothed_lw, k_hat) = pareto_smoothed(&raw);
        // Numerator: sum_i w̃_ij * exp(log_lik[i, j]); done in log-space via LSE.
        let mut combined: Vec<f64> = (0..s).map(|i| smoothed_lw[i] + log_lik[[i, j]]).collect();
        let mnum = combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_num: f64 = combined.iter().map(|v| (v - mnum).exp()).sum();
        let log_num = mnum + sum_num.ln();
        // Denominator: sum_i w̃_ij.
        let mden = smoothed_lw
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_den: f64 = smoothed_lw.iter().map(|v| (v - mden).exp()).sum();
        let log_den = mden + sum_den.ln();
        let elpd_j = log_num - log_den;
        // p_loo_j via WAIC's second term for stability.
        let col: Vec<f64> = (0..s).map(|i| log_lik[[i, j]]).collect();
        let mean_ll: f64 = col.iter().sum::<f64>() / s as f64;
        let var_ll: f64 = col.iter().map(|v| (v - mean_ll).powi(2)).sum::<f64>() / (s as f64 - 1.0);
        p_loo += var_ll;
        pointwise.push(elpd_j);
        pareto_k.push(k_hat);
        // Suppress unused-mut warning on `combined` when the loop ends.
        combined.clear();
    }
    let elpd: f64 = pointwise.iter().sum();
    let mean_p: f64 = elpd / n as f64;
    let var_p: f64 =
        pointwise.iter().map(|v| (v - mean_p).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let elpd_se = (n as f64 * var_p).sqrt();
    Ok(PsisLooResult {
        elpd,
        elpd_se,
        p_loo,
        looic: -2.0 * elpd,
        pointwise,
        pareto_k,
    })
}

/// Fit a generalised Pareto to the top-`M` log-weights, replace them by
/// the order-statistic quantiles of the fit, and return both the smoothed
/// log-weight vector and the Pareto shape parameter `k̂`.
///
/// Uses the moment-of-methods fit that the R `loo` package uses when a
/// full ML fit is unavailable — accurate enough for the LOO diagnostic
/// (`k̂ > 0.7` warning threshold has a comfortable margin).
fn pareto_smoothed(raw: &[f64]) -> (Vec<f64>, f64) {
    let s = raw.len();
    // Normalise to be a proper log-weight vector.
    let m = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum: f64 = m + raw.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    let normalised: Vec<f64> = raw.iter().map(|v| v - log_sum).collect();
    // Sort by value ascending; keep the original indices to reassemble.
    let mut order: Vec<(usize, f64)> = normalised.iter().copied().enumerate().collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let m_tail = ((s as f64 * 0.2).floor() as usize).max(3);
    if m_tail >= s {
        return (normalised, 0.0);
    }
    let tail_start = s - m_tail;
    let threshold = order[tail_start].1;
    let tail_raw: Vec<f64> = order[tail_start..]
        .iter()
        .map(|&(_, lw)| (lw - threshold).exp())
        .collect();
    // Method-of-moments estimator for the generalised Pareto (Hosking, 1985).
    let mean: f64 = tail_raw.iter().sum::<f64>() / m_tail as f64;
    let var: f64 = tail_raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m_tail as f64 - 1.0);
    let (k_hat, sigma) = if var > 1e-16 && mean > 0.0 {
        let ratio = mean * mean / var;
        let k = 0.5 * (1.0 - ratio);
        let sig = 0.5 * mean * (ratio + 1.0);
        (k.max(-0.5).min(1.0), sig.max(1e-12))
    } else {
        (0.0, mean.max(1e-12))
    };

    // Replace tail values with the order-statistic quantile of the fit.
    let mut smoothed = normalised.clone();
    for (rank, &(orig_idx, _)) in order[tail_start..].iter().enumerate() {
        let q = (rank as f64 + 0.5) / m_tail as f64;
        let ppf = if k_hat.abs() < 1e-6 {
            -(1.0 - q).ln() * sigma
        } else {
            sigma * ((1.0 - q).powf(-k_hat) - 1.0) / k_hat
        };
        smoothed[orig_idx] = threshold + ppf.ln().max(-1e100);
    }
    // Renormalise the smoothed log-weights and reassemble.
    let sm = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_z: f64 = sm + smoothed.iter().map(|v| (v - sm).exp()).sum::<f64>().ln();
    for v in smoothed.iter_mut() {
        *v -= log_z;
    }
    (smoothed, k_hat)
}
