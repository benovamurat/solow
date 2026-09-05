//! Bootstrap resampling for confidence intervals on any scalar statistic.
//!
//! The [`bootstrap_ci`] entry point takes a sample size `n`, a `statistic`
//! callback that computes a scalar from a resampled index list, the number
//! of bootstrap replicates, a confidence level, and a bootstrap method, and
//! returns the point estimate together with a two-sided confidence
//! interval. Passing indices instead of raw data keeps the callback typed
//! independently of the underlying container — the caller closes over
//! whatever data structure they use.
//!
//! Three interval kinds are supported:
//!
//! * [`BootstrapMethod::Percentile`] — the classical `[Q_{α/2}, Q_{1−α/2}]`
//!   percentile of the bootstrap distribution.
//! * [`BootstrapMethod::Basic`] — the "reverse-percentile" `[2θ̂ - Q_{1−α/2},
//!   2θ̂ - Q_{α/2}]` interval, which corrects for asymmetric bias.
//! * [`BootstrapMethod::Bca`] — Efron's bias-corrected and accelerated
//!   interval, using a jackknife estimate of the acceleration constant.
//!   BCa requires `n ≥ 2` and adjusts the percentile levels for bias `z₀`
//!   and skewness `â`; it is the recommended default for asymmetric or
//!   biased statistics.
//!
//! Resampling uses the same deterministic MMIX-LCG as the other splitters,
//! so a fixed seed produces byte-identical bootstrap replicates on every
//! run and platform.

use solow_core::{Error, Result};

/// Bootstrap confidence-interval method.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Classical percentile interval.
    Percentile,
    /// Reverse-percentile ("basic") interval.
    Basic,
    /// Efron's bias-corrected and accelerated interval.
    Bca,
}

/// A point estimate together with its bootstrap confidence interval.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct BootstrapCi {
    /// The statistic evaluated on the original sample.
    pub point: f64,
    /// Two-sided confidence-interval lower bound.
    pub low: f64,
    /// Two-sided confidence-interval upper bound.
    pub high: f64,
    /// Confidence level `1 - α` this interval was computed at.
    pub confidence: f64,
    /// The method that produced this interval.
    pub method: BootstrapMethod,
    /// Sorted bootstrap replicates of the statistic.
    pub replicates: Vec<f64>,
}

impl BootstrapCi {
    /// Standard error of the statistic: standard deviation of the bootstrap
    /// replicates.
    pub fn standard_error(&self) -> f64 {
        let n = self.replicates.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let mean: f64 = self.replicates.iter().sum::<f64>() / n;
        let s2: f64 = self
            .replicates
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        s2.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (MMIX 64-bit LCG) — same constants as splitters.rs
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// bootstrap_ci
// ---------------------------------------------------------------------------

/// Compute a bootstrap confidence interval for `statistic(indices)` on a
/// sample of size `n`.
///
/// `statistic` is called once on the identity indices to produce the point
/// estimate, then `n_boot` times on `n`-of-`n` resamples-with-replacement.
/// Any error from the callback is propagated. `n_boot` should be `≥ 999`
/// for tight quantile estimates (a typical default is 1000-2000).
pub fn bootstrap_ci<F>(
    n: usize,
    statistic: F,
    n_boot: usize,
    confidence: f64,
    method: BootstrapMethod,
    seed: u64,
) -> Result<BootstrapCi>
where
    F: Fn(&[usize]) -> Result<f64>,
{
    if n == 0 {
        return Err(Error::Value("bootstrap_ci: n must be positive".into()));
    }
    if n_boot == 0 {
        return Err(Error::Value("bootstrap_ci: n_boot must be ≥ 1".into()));
    }
    if !(0.0 < confidence && confidence < 1.0) {
        return Err(Error::Value(format!(
            "bootstrap_ci: confidence must be in (0, 1), got {confidence}"
        )));
    }
    let identity: Vec<usize> = (0..n).collect();
    let point = statistic(&identity)?;
    if !point.is_finite() {
        return Err(Error::Value(
            "bootstrap_ci: statistic returned a non-finite value on the full sample".into(),
        ));
    }

    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut replicates = Vec::with_capacity(n_boot);
    let mut sample = vec![0usize; n];
    for _ in 0..n_boot {
        for slot in sample.iter_mut() {
            *slot = uniform_index(&mut state, n as u64);
        }
        let s = statistic(&sample)?;
        if s.is_finite() {
            replicates.push(s);
        }
    }
    if replicates.is_empty() {
        return Err(Error::Value(
            "bootstrap_ci: every bootstrap replicate produced a non-finite value".into(),
        ));
    }
    replicates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let (lo_q, hi_q) = match method {
        BootstrapMethod::Percentile => (alpha / 2.0, 1.0 - alpha / 2.0),
        BootstrapMethod::Basic => (alpha / 2.0, 1.0 - alpha / 2.0),
        BootstrapMethod::Bca => bca_quantiles(&replicates, point, &statistic, n, alpha)?,
    };
    let q_low = quantile_sorted(&replicates, lo_q);
    let q_high = quantile_sorted(&replicates, hi_q);
    let (low, high) = match method {
        BootstrapMethod::Percentile | BootstrapMethod::Bca => (q_low, q_high),
        BootstrapMethod::Basic => (2.0 * point - q_high, 2.0 * point - q_low),
    };
    Ok(BootstrapCi {
        point,
        low,
        high,
        confidence,
        method,
        replicates,
    })
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    // Type-7 linear interpolation (R's default; matches numpy.quantile).
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    (1.0 - frac) * sorted[lo] + frac * sorted[hi]
}

fn bca_quantiles<F>(
    sorted: &[f64],
    point: f64,
    statistic: &F,
    n: usize,
    alpha: f64,
) -> Result<(f64, f64)>
where
    F: Fn(&[usize]) -> Result<f64>,
{
    if n < 2 {
        return Err(Error::Value(
            "bootstrap_ci: BCa requires n ≥ 2 for a jackknife acceleration estimate".into(),
        ));
    }
    // Bias-correction z0 from the fraction of replicates below the point estimate.
    let below = sorted.iter().filter(|&&v| v < point).count() as f64;
    let mut prop = below / sorted.len() as f64;
    // Clip to avoid ±∞ from the inverse-normal at the ends.
    let eps = 1.0 / (2.0 * sorted.len() as f64);
    prop = prop.clamp(eps, 1.0 - eps);
    let z0 = normal_inverse_cdf(prop);

    // Jackknife acceleration: replace each index by every other row once.
    let mut jack = Vec::with_capacity(n);
    let mut leave_one: Vec<usize> = Vec::with_capacity(n - 1);
    for i in 0..n {
        leave_one.clear();
        for j in 0..n {
            if j != i {
                leave_one.push(j);
            }
        }
        let s = statistic(&leave_one)?;
        if s.is_finite() {
            jack.push(s);
        }
    }
    if jack.is_empty() {
        return Err(Error::Value(
            "bootstrap_ci: jackknife produced no finite values".into(),
        ));
    }
    let jm: f64 = jack.iter().sum::<f64>() / jack.len() as f64;
    let num: f64 = jack.iter().map(|v| (jm - v).powi(3)).sum();
    let den: f64 = jack.iter().map(|v| (jm - v).powi(2)).sum();
    let a = if den > 0.0 {
        num / (6.0 * den.powf(1.5))
    } else {
        0.0
    };

    let z_alpha_lo = normal_inverse_cdf(alpha / 2.0);
    let z_alpha_hi = normal_inverse_cdf(1.0 - alpha / 2.0);
    let adj = |z: f64| {
        let denom = 1.0 - a * (z0 + z);
        // Guard against a degenerate acceleration that would flip the sign.
        if denom == 0.0 {
            return normal_cdf(z0 + z);
        }
        normal_cdf(z0 + (z0 + z) / denom)
    };
    let lo_q = adj(z_alpha_lo).clamp(1e-6, 1.0 - 1e-6);
    let hi_q = adj(z_alpha_hi).clamp(1e-6, 1.0 - 1e-6);
    Ok((lo_q, hi_q))
}

// ---------------------------------------------------------------------------
// Normal CDF / inverse CDF (Acklam's rational approximation)
// ---------------------------------------------------------------------------

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 — accurate to ~1.5e-7, which is plenty for
    // the BCa quantile adjustment (whose input error dominates the erf
    // approximation error at these confidence levels).
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-(x * x)).exp();
    sign * y
}

fn normal_inverse_cdf(p: f64) -> f64 {
    // Peter Acklam's rational approximation for the standard-normal inverse
    // CDF (max relative error ~1.15e-9 in central region; ~1e-6 in tails).
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}
