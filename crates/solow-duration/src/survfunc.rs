//! Right-censored survival function estimation (Kaplan–Meier).
//!
//! [`SurvfuncRight`] computes the product-limit estimator of the survival
//! function `S(t) = P(T > t)` from right-censored data, together with the
//! Greenwood estimate of its standard error. The construction matches the
//! reference implementation exactly: the reported times are the *distinct
//! event times only* (times at which at least one failure occurred), censored
//! times never appearing on their own.

use ndarray::Array1;
use solow_core::error::{Error, Result};

/// The Kaplan–Meier estimate of a right-censored survival function.
///
/// Construct with [`SurvfuncRight::new`], passing observation times and a
/// status indicator (`1.0` = event/failure observed, `0.0` = right-censored).
#[derive(Clone, Debug)]
pub struct SurvfuncRight {
    /// Distinct event times, ascending. Only times with at least one failure.
    pub surv_times: Array1<f64>,
    /// The product-limit survival probability `S(t)` at each event time.
    pub surv_prob: Array1<f64>,
    /// Greenwood standard error of `S(t)` at each event time (`NaN` where the
    /// risk set is exhausted, i.e. `n == d`).
    pub surv_prob_se: Array1<f64>,
    /// Size of the risk set just prior to each event time.
    pub n_risk: Array1<f64>,
    /// Number of events (failures) at each event time.
    pub n_events: Array1<f64>,
}

impl SurvfuncRight {
    /// Estimate the survival function from right-censored data.
    ///
    /// `time` and `status` must have equal length. `status[i]` is treated as an
    /// event when it rounds to `1` and as censoring otherwise.
    pub fn new(time: &[f64], status: &[f64]) -> Result<Self> {
        if time.len() != status.len() {
            return Err(Error::Shape("time and status length differ".into()));
        }
        if time.is_empty() {
            return Err(Error::Shape("empty time vector".into()));
        }

        // Distinct times, ascending, plus the inverse mapping (each
        // observation's index into the unique-time array).
        let mut order: Vec<usize> = (0..time.len()).collect();
        order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
        let mut utime: Vec<f64> = Vec::new();
        let mut rtime: Vec<usize> = vec![0; time.len()];
        for &i in &order {
            if utime.is_empty() || time[i] != *utime.last().unwrap() {
                utime.push(time[i]);
            }
            rtime[i] = utime.len() - 1;
        }
        let ml = utime.len();

        // d[k] = number of failures at the k-th distinct time.
        // raw_n[k] = number of observations whose time equals the k-th time.
        let mut d = vec![0.0_f64; ml];
        let mut raw_n = vec![0.0_f64; ml];
        for i in 0..time.len() {
            let k = rtime[i];
            raw_n[k] += 1.0;
            if status[i].round() as i64 == 1 {
                d[k] += 1.0;
            }
        }

        // n[k] = size of risk set just before the k-th time = sum of raw_n at
        // times >= the k-th time (reverse cumulative sum).
        let mut n = vec![0.0_f64; ml];
        let mut acc = 0.0;
        for k in (0..ml).rev() {
            acc += raw_n[k];
            n[k] = acc;
        }

        // Retain only times where an event occurred.
        let keep: Vec<usize> = (0..ml).filter(|&k| d[k] > 0.0).collect();
        let nk = keep.len();
        let dk: Vec<f64> = keep.iter().map(|&k| d[k]).collect();
        let nrisk: Vec<f64> = keep.iter().map(|&k| n[k]).collect();
        let times: Vec<f64> = keep.iter().map(|&k| utime[k]).collect();

        // Product-limit survival probability via cumulative sum of logs.
        let mut sp = vec![0.0_f64; nk];
        let mut zero_flag = vec![false; nk];
        let mut log_cumsum = 0.0;
        for j in 0..nk {
            let mut frac = 1.0 - dk[j] / nrisk[j];
            if frac < 1e-16 {
                frac = 1e-16;
                zero_flag[j] = true;
            }
            log_cumsum += frac.ln();
            sp[j] = log_cumsum.exp();
            if zero_flag[j] {
                sp[j] = 0.0;
            }
        }

        // Greenwood standard error.
        // term = d / (n * (n - d)); NaN where n == d or n == 0; cumulative sum;
        // sqrt; then multiply by S(t) where the value is finite or S != 0.
        let mut se = vec![0.0_f64; nk];
        let mut cum = 0.0;
        for j in 0..nk {
            let denom = (nrisk[j] * (nrisk[j] - dk[j])).max(1e-12);
            let mut term = dk[j] / denom;
            if nrisk[j] == dk[j] || nrisk[j] == 0.0 {
                term = f64::NAN;
            }
            cum += term;
            let s = cum.sqrt();
            // locs = isfinite(se) | (sp != 0)
            if s.is_finite() || sp[j] != 0.0 {
                se[j] = s * sp[j];
            } else {
                se[j] = f64::NAN;
            }
        }

        Ok(SurvfuncRight {
            surv_times: Array1::from_vec(times),
            surv_prob: Array1::from_vec(sp),
            surv_prob_se: Array1::from_vec(se),
            n_risk: Array1::from_vec(nrisk),
            n_events: Array1::from_vec(dk),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_events_no_censoring() {
        // With all distinct event times and no censoring, S drops by 1/n each
        // step: 1 - i/n.
        let time = [1.0, 2.0, 3.0, 4.0];
        let status = [1.0, 1.0, 1.0, 1.0];
        let s = SurvfuncRight::new(&time, &status).unwrap();
        assert_eq!(s.surv_times.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        let exp = [0.75, 0.5, 0.25, 0.0];
        for (i, &e) in exp.iter().enumerate() {
            assert!((s.surv_prob[i] - e).abs() < 1e-12);
        }
        // Last point: n == d, so Greenwood SE is NaN.
        assert!(s.surv_prob_se[3].is_nan());
    }

    #[test]
    fn censored_times_excluded_from_surv_times() {
        // A censored observation at a time with no event must not appear.
        let time = [1.0, 2.0, 3.0];
        let status = [1.0, 0.0, 1.0];
        let s = SurvfuncRight::new(&time, &status).unwrap();
        assert_eq!(s.surv_times.to_vec(), vec![1.0, 3.0]);
        // At t=1: 1 - 1/3; at t=3: risk set is just {t=3}, so S -> 0.
        assert!((s.surv_prob[0] - (2.0 / 3.0)).abs() < 1e-12);
        assert!((s.surv_prob[1]).abs() < 1e-12);
    }

    #[test]
    fn mismatched_lengths_error() {
        assert!(SurvfuncRight::new(&[1.0, 2.0], &[1.0]).is_err());
    }
}
