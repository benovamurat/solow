//! Change-point detection for univariate time series.
//!
//! * [`cusum`] — the classic Cumulative-SUM chart with a symmetric
//!   two-sided decision rule.
//! * [`pelt`] — Killick-Fearnhead-Eckley (2012) exact PELT algorithm
//!   for a Gaussian change-in-mean cost with a caller-supplied
//!   linear penalty.
//! * [`binary_segmentation`] — the classical top-down greedy
//!   Auger-Lawrence (1989) change-point scan.
//!
//! These functions return change-point *indices* — the first sample of
//! each new segment — with 0 always implied and the length of the
//! series always excluded.

use ndarray::ArrayView1;
use solow_core::{Error, Result};

/// A single CUSUM alarm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CusumAlarm {
    /// Sample index at which the alarm fires.
    pub index: usize,
    /// Sign of the departure (`+1` = upward mean shift, `-1` = downward).
    pub direction: i32,
    /// The CUSUM statistic at the alarm.
    pub statistic: f64,
}

/// Two-sided CUSUM chart.
///
/// * `target` is the in-control mean.
/// * `sigma` is the in-control standard deviation.
/// * `k` is the reference value in standard-deviation units (default 0.5).
/// * `h` is the alarm threshold in standard-deviation units (default 5.0).
pub fn cusum(
    x: ArrayView1<'_, f64>,
    target: f64,
    sigma: f64,
    k: f64,
    h: f64,
) -> Result<Vec<CusumAlarm>> {
    if x.is_empty() {
        return Err(Error::Value("cusum: empty series".into()));
    }
    if sigma <= 0.0 {
        return Err(Error::Value("cusum: sigma must be > 0".into()));
    }
    let mut alarms = Vec::new();
    let mut sh = 0.0_f64;
    let mut sl = 0.0_f64;
    let ks = k * sigma;
    let hs = h * sigma;
    for (i, &v) in x.iter().enumerate() {
        let dev = v - target;
        sh = (sh + dev - ks).max(0.0);
        sl = (sl - dev - ks).max(0.0);
        if sh > hs {
            alarms.push(CusumAlarm { index: i, direction: 1, statistic: sh });
            sh = 0.0;
        }
        if sl > hs {
            alarms.push(CusumAlarm { index: i, direction: -1, statistic: sl });
            sl = 0.0;
        }
    }
    Ok(alarms)
}

/// PELT (Killick-Fearnhead-Eckley 2012) exact segmentation under a
/// Gaussian change-in-mean cost with linear penalty `β`.
///
/// The output is a sorted list of change-point indices; each index is
/// the FIRST sample of a new segment. The trivial 0 and `n` boundaries
/// are omitted.
pub fn pelt(x: ArrayView1<'_, f64>, penalty: f64) -> Result<Vec<usize>> {
    let n = x.len();
    if n == 0 {
        return Err(Error::Value("pelt: empty series".into()));
    }
    if penalty < 0.0 {
        return Err(Error::Value("pelt: penalty must be ≥ 0".into()));
    }
    // Prefix sums for O(1) segment cost calculation.
    let mut ps = vec![0.0_f64; n + 1];
    let mut ps2 = vec![0.0_f64; n + 1];
    for i in 0..n {
        ps[i + 1] = ps[i] + x[i];
        ps2[i + 1] = ps2[i] + x[i] * x[i];
    }
    let segment_cost = |s: usize, e: usize| -> f64 {
        // Sum of squares minus segment length × squared mean, plus a
        // small ridge to keep numeric stability at length 1.
        let len = (e - s) as f64;
        if len <= 0.0 {
            return 0.0;
        }
        let mean = (ps[e] - ps[s]) / len;
        let sum_sq = ps2[e] - ps2[s];
        sum_sq - len * mean * mean
    };
    let mut f = vec![f64::INFINITY; n + 1];
    let mut cp = vec![0_usize; n + 1];
    f[0] = -penalty;
    let mut rr = vec![0_usize];
    for t in 1..=n {
        let mut best = f64::INFINITY;
        let mut best_r = 0;
        let mut r_next = Vec::with_capacity(rr.len());
        for &r in &rr {
            let cost = f[r] + segment_cost(r, t) + penalty;
            if cost < best {
                best = cost;
                best_r = r;
            }
            // Pruning: keep r if r's own optimal value could still beat f[t].
            if f[r] + segment_cost(r, t) <= best {
                r_next.push(r);
            }
        }
        f[t] = best;
        cp[t] = best_r;
        r_next.push(t);
        rr = r_next;
    }
    // Trace back.
    let mut cps: Vec<usize> = Vec::new();
    let mut t = n;
    while t > 0 {
        let prev = cp[t];
        if prev > 0 {
            cps.push(prev);
        }
        t = prev;
    }
    cps.sort();
    Ok(cps)
}

/// Binary segmentation — top-down recursive maximum-t-statistic split
/// (Auger-Lawrence 1989). Halts when no split's t-statistic exceeds
/// `threshold`.
pub fn binary_segmentation(x: ArrayView1<'_, f64>, threshold: f64) -> Result<Vec<usize>> {
    let n = x.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    segment(&x.to_vec(), 0, n, threshold, &mut out);
    out.sort();
    Ok(out)
}

fn segment(x: &[f64], s: usize, e: usize, threshold: f64, out: &mut Vec<usize>) {
    if e - s < 4 {
        return;
    }
    let mut best_t = -1.0_f64;
    let mut best_k = s + 1;
    for k in (s + 2)..(e - 1) {
        let (m1, v1) = mean_var(&x[s..k]);
        let (m2, v2) = mean_var(&x[k..e]);
        let n1 = (k - s) as f64;
        let n2 = (e - k) as f64;
        let sp2 = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / (n1 + n2 - 2.0).max(1.0);
        let t = (m1 - m2).abs() / (sp2 * (1.0 / n1 + 1.0 / n2)).sqrt().max(1e-30);
        if t > best_t {
            best_t = t;
            best_k = k;
        }
    }
    if best_t > threshold {
        out.push(best_k);
        segment(x, s, best_k, threshold, out);
        segment(x, best_k, e, threshold, out);
    }
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn cusum_fires_on_a_clear_upward_shift() {
        // 20 samples at mean 0, then 20 at mean 2.
        let mut v = vec![0.0_f64; 20];
        v.extend(vec![2.0_f64; 20]);
        let x = Array1::from(v);
        let alarms = cusum(x.view(), 0.0, 1.0, 0.5, 4.0).unwrap();
        assert!(!alarms.is_empty());
        assert_eq!(alarms[0].direction, 1);
    }

    #[test]
    fn pelt_finds_a_single_change_point_in_a_step_function() {
        let mut v = vec![0.0_f64; 30];
        v.extend(vec![5.0_f64; 30]);
        let x = Array1::from(v);
        let cps = pelt(x.view(), 1.0).unwrap();
        assert!(!cps.is_empty());
        // Change point should be near index 30.
        assert!(cps.iter().any(|&c| c >= 25 && c <= 35));
    }

    #[test]
    fn binary_segmentation_finds_the_step_boundary() {
        let mut v = vec![0.0_f64; 20];
        v.extend(vec![5.0_f64; 20]);
        let x = Array1::from(v);
        let cps = binary_segmentation(x.view(), 3.0).unwrap();
        assert!(cps.iter().any(|&c| (c as i64 - 20).abs() <= 2));
    }
}
