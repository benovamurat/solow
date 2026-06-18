//! Log-rank and weighted log-rank tests for equality of survival distributions.
//!
//! [`survdiff`] compares the survival distributions of two or more groups using
//! the (weighted) log-rank test. It mirrors the reference
//! `survfunc.survdiff` for the single-stratum, no-entry case and supports the
//! standard weight families:
//!
//! * [`WeightType::LogRank`] — the unweighted log-rank test (Mantel–Cox).
//! * [`WeightType::GehanBreslow`] — weights by the number at risk.
//! * [`WeightType::TaroneWare`] — weights by the square root of the number at
//!   risk.
//! * [`WeightType::FlemingHarrington`] — weights by `S(t-)^p`, where `S` is the
//!   pooled Kaplan–Meier estimate at the previous event time.
//!
//! The statistic is `(O − E)' V⁻¹ (O − E)`, distributed as chi-square with
//! `g − 1` degrees of freedom under the null, and the returned p-value is the
//! upper tail of that chi-square distribution.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::chi2_sf;
use solow_linalg::solve;

/// Weight family for the (weighted) log-rank test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightType {
    /// Unweighted log-rank test (all weights equal to 1).
    LogRank,
    /// Gehan–Breslow: weight by the total number at risk at each event time.
    GehanBreslow,
    /// Tarone–Ware: weight by the square root of the number at risk.
    TaroneWare,
    /// Fleming–Harrington: weight by `S(t-)^p` (pooled KM at previous time).
    FlemingHarrington(f64),
}

/// Outcome of a (weighted) log-rank test.
#[derive(Clone, Debug)]
pub struct SurvDiffResult {
    /// The chi-square test statistic.
    pub chisq: f64,
    /// The upper-tail p-value under the chi-square(`g - 1`) null.
    pub pvalue: f64,
    /// Degrees of freedom (`number of groups - 1`).
    pub df: usize,
}

/// Test the equality of two or more survival distributions.
///
/// `time[i]` is the event or censoring time, `status[i]` is `1.0` for an
/// observed event and `0.0` for right-censoring, and `group[i]` is a group
/// label (any `f64`; distinct values define the groups, ordered ascending).
///
/// Returns the chi-square statistic, its p-value, and the degrees of freedom.
pub fn survdiff(
    time: &[f64],
    status: &[f64],
    group: &[f64],
    weight_type: WeightType,
) -> Result<SurvDiffResult> {
    let n = time.len();
    if status.len() != n || group.len() != n {
        return Err(Error::Shape("time/status/group length mismatch".into()));
    }
    if n == 0 {
        return Err(Error::Shape("empty sample".into()));
    }

    // Distinct group labels (ascending) -> `gr`.
    let mut gr: Vec<f64> = group.to_vec();
    gr.sort_by(|a, b| a.total_cmp(b));
    gr.dedup();
    let ng = gr.len();
    if ng < 2 {
        return Err(Error::Shape("survdiff requires at least two groups".into()));
    }

    // Unique event/censoring times (ascending) and the inverse map.
    let mut utimes: Vec<f64> = time.to_vec();
    utimes.sort_by(|a, b| a.total_cmp(b));
    utimes.dedup();
    let ml = utimes.len();
    let time_rank = |t: f64| -> usize { utimes.partition_point(|&u| u < t) };

    // Per-group event counts (obsv) and risk-set sizes (nrisk) at each time.
    let mut obsv: Vec<Array1<f64>> = vec![Array1::zeros(ml); ng];
    let mut nrisk: Vec<Array1<f64>> = vec![Array1::zeros(ml); ng];
    let group_idx = |g: f64| -> usize { gr.iter().position(|&x| x == g).unwrap() };

    // n[g][k] = number of subjects in group g whose time equals utimes[k].
    let mut nbin: Vec<Array1<f64>> = vec![Array1::zeros(ml); ng];
    for i in 0..n {
        let gi = group_idx(group[i]);
        let k = time_rank(time[i]);
        nbin[gi][k] += 1.0;
        if status[i].round() as i64 == 1 {
            obsv[gi][k] += 1.0;
        }
    }
    // Risk set: reverse cumulative sum of nbin (no entry/left truncation).
    for g in 0..ng {
        let mut acc = 0.0;
        for k in (0..ml).rev() {
            acc += nbin[g][k];
            nrisk[g][k] = acc;
        }
    }

    // Pooled observed events and total at risk at each time.
    let mut obs = Array1::<f64>::zeros(ml);
    let mut nrisk_tot = Array1::<f64>::zeros(ml);
    for g in 0..ng {
        for k in 0..ml {
            obs[k] += obsv[g][k];
            nrisk_tot[k] += nrisk[g][k];
        }
    }

    // Indices where the total risk set exceeds 1 (others contribute nothing).
    let ix: Vec<usize> = (0..ml).filter(|&k| nrisk_tot[k] > 1.0).collect();

    // Weight series w[k].
    let weights: Option<Array1<f64>> = match weight_type {
        WeightType::LogRank => None,
        WeightType::GehanBreslow => Some(nrisk_tot.clone()),
        WeightType::TaroneWare => Some(nrisk_tot.mapv(f64::sqrt)),
        WeightType::FlemingHarrington(p) => {
            // sp = cumprod(1 - obs/nrisk_tot); weights = roll(sp^p, 1); w[0]=1.
            let mut sp = Array1::<f64>::zeros(ml);
            let mut logcum = 0.0;
            for k in 0..ml {
                let frac = 1.0 - obs[k] / nrisk_tot[k];
                logcum += frac.ln();
                sp[k] = logcum.exp();
            }
            let mut w = sp.mapv(|v| v.powf(p));
            // np.roll(w, 1): shift right by one, wrap last -> first, then w[0]=1.
            let mut rolled = Array1::<f64>::zeros(ml);
            for k in 0..ml {
                rolled[k] = w[(k + ml - 1) % ml];
            }
            rolled[0] = 1.0;
            w = rolled;
            Some(w)
        }
    };

    let dfs = ng - 1;

    // r[g][k] = nrisk[g][k] / clip(nrisk_tot[k], 1e-10).
    let mut r: Vec<Array1<f64>> = vec![Array1::zeros(ml); ng];
    for g in 0..ng {
        for k in 0..ml {
            let denom = nrisk_tot[k].max(1e-10);
            r[g][k] = nrisk[g][k] / denom;
        }
    }

    // var_denom = clip(nrisk_tot - 1, 1e-10).
    let var_denom: Array1<f64> = nrisk_tot.mapv(|v| (v - 1.0).max(1e-10));
    // var_scalar_part[k] = obs * (nrisk_tot - obs) / var_denom.
    let var_scalar: Array1<f64> =
        Array1::from_iter((0..ml).map(|k| obs[k] * (nrisk_tot[k] - obs[k]) / var_denom[k]));

    // Build O-E vector and variance matrix using groups 1..dfs (reference uses
    // the first group as reference).
    let mut obs_vec = Array1::<f64>::zeros(dfs);
    let mut var_mat = Array2::<f64>::zeros((dfs, dfs));

    for g in 1..=dfs {
        // oe[k] = obsv[g][k] - r[g][k] * obs[k]
        let mut oe = Array1::<f64>::zeros(ml);
        for k in 0..ml {
            oe[k] = obsv[g][k] - r[g][k] * obs[k];
        }

        // var row over the dfs other groups: for column c (1..=dfs),
        //   r[c][k] * (indicator(c == g) - r[g][k]) * var_scalar[k]
        // accumulated over kept time indices.
        let mut var_row = Array1::<f64>::zeros(dfs);

        // Apply weights if present.
        let (oe_w, w2): (Array1<f64>, Option<Array1<f64>>) = match &weights {
            None => (oe, None),
            Some(w) => {
                let mut oew = Array1::<f64>::zeros(ml);
                for k in 0..ml {
                    oew[k] = w[k] * oe[k];
                }
                (oew, Some(w.mapv(|v| v * v)))
            }
        };

        for &k in &ix {
            obs_vec[g - 1] += oe_w[k];
            for (ci, c) in (1..=dfs).enumerate() {
                let ind = if c == g { 1.0 } else { 0.0 };
                let mut v = r[c][k] * (ind - r[g][k]) * var_scalar[k];
                if let Some(w2v) = &w2 {
                    v *= w2v[k];
                }
                var_row[ci] += v;
            }
        }
        for ci in 0..dfs {
            var_mat[[g - 1, ci]] = var_row[ci];
        }
    }

    // chisq = (O-E)' V^{-1} (O-E); pvalue = chi2 upper tail with dfs df.
    let sol = solve(&var_mat, &obs_vec)?;
    let chisq = obs_vec.dot(&sol);
    let pvalue = chi2_sf(chisq, dfs as f64);

    Ok(SurvDiffResult {
        chisq,
        pvalue,
        df: dfs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_groups_zero_statistic() {
        // Two groups with identical data -> O = E exactly -> chisq = 0.
        let time = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let status = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let group = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let res = survdiff(&time, &status, &group, WeightType::LogRank).unwrap();
        assert!(res.chisq.abs() < 1e-12, "chisq={}", res.chisq);
        assert!((res.pvalue - 1.0).abs() < 1e-12);
        assert_eq!(res.df, 1);
    }

    #[test]
    fn requires_two_groups() {
        let time = [1.0, 2.0, 3.0];
        let status = [1.0, 1.0, 1.0];
        let group = [0.0, 0.0, 0.0];
        assert!(survdiff(&time, &status, &group, WeightType::LogRank).is_err());
    }

    #[test]
    fn statistic_nonnegative() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let group = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        for wt in [
            WeightType::LogRank,
            WeightType::GehanBreslow,
            WeightType::TaroneWare,
            WeightType::FlemingHarrington(1.0),
        ] {
            let res = survdiff(&time, &status, &group, wt).unwrap();
            assert!(res.chisq >= 0.0);
            assert!((0.0..=1.0).contains(&res.pvalue));
        }
    }
}
