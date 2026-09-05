//! Cumulative incidence functions for competing-risks data.
//!
//! [`CumIncidenceRight`] estimates the cause-specific cumulative incidence
//! function `I(t, j) = P(T <= t and J = j)` for competing events under right
//! censoring, together with its (Aalen-type) standard error. It mirrors the
//! reference `survfunc.CumIncidenceRight` for the unweighted, no-covariate
//! case.
//!
//! The estimator first forms the pooled (all-cause) Kaplan–Meier survival
//! function, then accumulates each cause's incidence as
//! `I_j(t) = Σ_{s ≤ t} S(s⁻) · d_j(s) / n(s)`, where `S(s⁻)` is the pooled
//! survival just before time `s`, `d_j(s)` the number of cause-`j` events at
//! `s`, and `n(s)` the size of the risk set.

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// Estimated cumulative incidence functions for competing risks.
///
/// Construct with [`CumIncidenceRight::new`]. `status[i]` encodes the event
/// type: `0` denotes right-censoring and `1, 2, …, J` denote the competing
/// causes. The estimates are reported at the distinct observation times
/// [`CumIncidenceRight::times`].
#[derive(Clone, Debug)]
pub struct CumIncidenceRight {
    /// Distinct observation times (ascending) at which incidence is reported.
    pub times: Array1<f64>,
    /// `cinc[j]` = estimated cumulative incidence for cause `j+1`.
    pub cinc: Vec<Array1<f64>>,
    /// `cinc_se[j]` = standard error of `cinc[j]` (may contain `NaN` once the
    /// risk set is exhausted).
    pub cinc_se: Vec<Array1<f64>>,
}

impl CumIncidenceRight {
    /// Estimate the cumulative incidence functions from competing-risks data.
    ///
    /// `time` and `status` must have equal length. `status` values must be
    /// non-negative; `0` is censoring and `1..=J` are the competing causes.
    pub fn new(time: &[f64], status: &[f64]) -> Result<Self> {
        let nobs = time.len();
        if status.len() != nobs {
            return Err(Error::Shape("time and status length differ".into()));
        }
        if nobs == 0 {
            return Err(Error::Shape("empty sample".into()));
        }

        // Distinct times (ascending) and the inverse map rtime[i] -> rank.
        let mut order: Vec<usize> = (0..nobs).collect();
        order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
        let mut utime: Vec<f64> = Vec::new();
        let mut rtime: Vec<usize> = vec![0; nobs];
        for &i in &order {
            if utime.is_empty() || time[i] != *utime.last().unwrap() {
                utime.push(time[i]);
            }
            rtime[i] = utime.len() - 1;
        }
        let ml = utime.len();

        // All-cause death counts and risk set sizes at each unique time.
        // status0 = (status >= 1); d_all[k] = sum of status0 at time-rank k.
        let mut d_all = vec![0.0_f64; ml];
        let mut nbin = vec![0.0_f64; ml];
        for i in 0..nobs {
            let k = rtime[i];
            nbin[k] += 1.0;
            if status[i] >= 1.0 {
                d_all[k] += 1.0;
            }
        }
        // n[k] = risk set just before time k = reverse cumulative sum of nbin.
        let mut n = vec![0.0_f64; ml];
        let mut acc = 0.0;
        for k in (0..ml).rev() {
            acc += nbin[k];
            n[k] = acc;
        }

        // Pooled (all-cause) product-limit survival sp[k] over ALL times.
        let mut sp = vec![0.0_f64; ml];
        let mut logcum = 0.0;
        for k in 0..ml {
            let mut frac = 1.0 - d_all[k] / n[k];
            if frac < 1e-16 {
                frac = 1e-16;
            }
            logcum += frac.ln();
            sp[k] = logcum.exp();
        }

        // Number of competing causes J = max(status).
        let ngrp = status.iter().cloned().fold(0.0_f64, f64::max).round() as usize;
        if ngrp == 0 {
            return Err(Error::Shape(
                "no events: status has no positive cause labels".into(),
            ));
        }

        // Cause-specific event counts d[j][k].
        let mut d: Vec<Vec<f64>> = vec![vec![0.0_f64; ml]; ngrp];
        for i in 0..nobs {
            let lab = status[i].round() as i64;
            if lab >= 1 {
                let j = (lab - 1) as usize;
                if j < ngrp {
                    d[j][rtime[i]] += 1.0;
                }
            }
        }

        // sp0[k] = S(t_{k-1}) / n[k]   with S(t_{-1}) := 1.
        let mut sp0 = vec![0.0_f64; ml];
        for k in 0..ml {
            let s_prev = if k == 0 { 1.0 } else { sp[k - 1] };
            sp0[k] = s_prev / n[k];
        }

        // Cumulative incidence ip[j][k] = cumsum_k(sp0 * d[j]).
        let mut ip: Vec<Vec<f64>> = vec![vec![0.0_f64; ml]; ngrp];
        for j in 0..ngrp {
            let mut c = 0.0;
            for k in 0..ml {
                c += sp0[k] * d[j][k];
                ip[j][k] = c;
            }
        }

        // Standard errors (Aalen-type variance, matching the reference).
        // da = sum_j d[j]  (all-cause deaths).
        let mut da = vec![0.0_f64; ml];
        for k in 0..ml {
            for dj in d.iter() {
                da[k] += dj[k];
            }
        }

        let mut se: Vec<Array1<f64>> = Vec::with_capacity(ngrp);
        for j in 0..ngrp {
            // ra1[k] = da / (n*(n-da)); v = ip^2 * cumsum(ra1)
            //          - 2*ip*cumsum(ip*ra1) + cumsum(ip^2 * ra1)
            // ra2[k] = (n - d[j]) * d[j] / n; v += cumsum(sp0^2 * ra2)
            // ra3[k] = sp0 * d[j] / n; v += -2*ip*cumsum(ra3) + 2*cumsum(ip*ra3)
            let mut v = vec![0.0_f64; ml];

            // Block 1.
            let mut c_ra1 = 0.0;
            let mut c_ip_ra1 = 0.0;
            let mut c_ip2_ra1 = 0.0;
            for k in 0..ml {
                let denom = n[k] * (n[k] - da[k]);
                let ra1 = da[k] / denom; // may be +/-inf or NaN when denom == 0
                c_ra1 += ra1;
                c_ip_ra1 += ip[j][k] * ra1;
                c_ip2_ra1 += ip[j][k] * ip[j][k] * ra1;
                v[k] = ip[j][k] * ip[j][k] * c_ra1 - 2.0 * ip[j][k] * c_ip_ra1 + c_ip2_ra1;
            }
            // Block 2.
            let mut c_sp0_ra2 = 0.0;
            for k in 0..ml {
                let ra2 = (n[k] - d[j][k]) * d[j][k] / n[k];
                c_sp0_ra2 += sp0[k] * sp0[k] * ra2;
                v[k] += c_sp0_ra2;
            }
            // Block 3.
            let mut c_ra3 = 0.0;
            let mut c_ip_ra3 = 0.0;
            for k in 0..ml {
                let ra3 = sp0[k] * d[j][k] / n[k];
                c_ra3 += ra3;
                c_ip_ra3 += ip[j][k] * ra3;
                v[k] += -2.0 * ip[j][k] * c_ra3 + 2.0 * c_ip_ra3;
            }

            let se_j: Vec<f64> = v.iter().map(|&x| x.sqrt()).collect();
            se.push(Array1::from_vec(se_j));
        }

        Ok(CumIncidenceRight {
            times: Array1::from_vec(utime),
            cinc: ip.into_iter().map(Array1::from_vec).collect(),
            cinc_se: se,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cause_matches_one_minus_km() {
        // With a single cause, the cumulative incidence is 1 - S(t) (the
        // complement of the Kaplan-Meier survival), since there are no
        // competing events.
        let time = [1.0, 2.0, 3.0, 4.0];
        let status = [1.0, 1.0, 1.0, 1.0];
        let ci = CumIncidenceRight::new(&time, &status).unwrap();
        assert_eq!(ci.cinc.len(), 1);
        // KM with all events: S = 0.75, 0.5, 0.25, 0 -> incidence = 0.25,...,1.
        let exp = [0.25, 0.5, 0.75, 1.0];
        for (k, &e) in exp.iter().enumerate() {
            assert!((ci.cinc[0][k] - e).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn two_causes_sum_to_one_minus_survival() {
        // The two cause-specific incidences should sum to the all-cause
        // cumulative incidence (1 - pooled KM survival) at every time.
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let ci = CumIncidenceRight::new(&time, &status).unwrap();
        assert_eq!(ci.cinc.len(), 2);
        // No censoring -> all-cause incidence reaches 1 at the last time.
        let last = ci.times.len() - 1;
        let total = ci.cinc[0][last] + ci.cinc[1][last];
        assert!((total - 1.0).abs() < 1e-12, "total={total}");
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(CumIncidenceRight::new(&[1.0, 2.0], &[1.0]).is_err());
    }
}
