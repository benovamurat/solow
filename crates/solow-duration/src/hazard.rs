//! Cox proportional-hazards regression (`PHReg`) with Breslow ties.
//!
//! [`PHReg`] estimates the regression coefficients of a Cox proportional
//! hazards model by maximizing the Breslow partial log-likelihood with a
//! Newton step (driving the gradient of the partial likelihood to zero). This
//! mirrors the reference's `hazard_regression.PHReg` with `ties='breslow'` for
//! the single-stratum, no-entry, no-offset case.
//!
//! The estimator exposes the coefficient vector [`PHRegResults::params`], its
//! standard errors [`PHRegResults::bse`] (from the inverse observed
//! information), z-statistics [`PHRegResults::tvalues`], two-sided normal
//! p-values [`PHRegResults::pvalues`], and the maximized partial
//! log-likelihood [`PHRegResults::llf`].

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// Pre-computed risk-set bookkeeping for a single (unstratified) sample.
///
/// All indices reference rows of the *filtered, time-sorted* exog matrix
/// [`PHReg::exog_s`].
struct Surv {
    /// `ufailt_ix[k]` = indices of subjects failing at the k-th distinct
    /// failure time (sorted ascending).
    ufailt_ix: Vec<Vec<usize>>,
    /// `risk_enter[k]` = indices of subjects entering the risk set at the
    /// k-th distinct failure time.
    risk_enter: Vec<Vec<usize>>,
}

/// A Cox proportional-hazards model awaiting estimation.
#[derive(Clone, Debug)]
pub struct PHReg {
    /// Filtered, time-sorted covariate matrix (rows = informative subjects).
    exog_s: Array2<f64>,
    /// Number of covariates.
    k: usize,
    surv_ufailt_ix: Vec<Vec<usize>>,
    surv_risk_enter: Vec<Vec<usize>>,
    maxiter: usize,
    gtol: f64,
}

impl PHReg {
    /// Build a Cox PH model from event times, covariates and status flags.
    ///
    /// `time[i]` is the event or censoring time, `exog` has one row per subject
    /// (covariates in columns, no implicit intercept — the Cox baseline hazard
    /// absorbs it), and `status[i]` is `1.0` for an observed event and `0.0`
    /// for right-censoring.
    pub fn new(time: &[f64], exog: &Array2<f64>, status: &[f64]) -> Result<Self> {
        let n = time.len();
        if exog.nrows() != n || status.len() != n {
            return Err(Error::Shape("time/exog/status length mismatch".into()));
        }
        if n == 0 {
            return Err(Error::Shape("empty sample".into()));
        }
        let k = exog.ncols();

        let has_event = (0..n).any(|i| status[i].round() as i64 == 1);
        if !has_event {
            return Err(Error::Convergence("no events in sample".into()));
        }

        // Reproduce the reference's subject filtering for a single stratum with
        // entry time 0 (no left truncation):
        //   * keep subjects whose entry (0) <= last failure time (always true);
        //   * drop subjects censored strictly before the first failure time.
        let mut first_failure = f64::INFINITY;
        for i in 0..n {
            if status[i].round() as i64 == 1 && time[i] < first_failure {
                first_failure = time[i];
            }
        }
        let mut rows: Vec<usize> = (0..n).filter(|&i| time[i] >= first_failure).collect();

        // Order by time within the stratum (stable sort, matching argsort).
        rows.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

        // Build the filtered/sorted exog and the corresponding time/status.
        let m = rows.len();
        let mut exog_s = Array2::<f64>::zeros((m, k));
        let mut time_s = vec![0.0_f64; m];
        let mut status_s = vec![0.0_f64; m];
        for (new_i, &old_i) in rows.iter().enumerate() {
            for j in 0..k {
                exog_s[[new_i, j]] = exog[[old_i, j]];
            }
            time_s[new_i] = time[old_i];
            status_s[new_i] = status[old_i];
        }

        let surv = build_surv(&time_s, &status_s);

        Ok(PHReg {
            exog_s,
            k,
            surv_ufailt_ix: surv.ufailt_ix,
            surv_risk_enter: surv.risk_enter,
            maxiter: 100,
            gtol: 1e-10,
        })
    }

    fn surv(&self) -> Surv {
        Surv {
            ufailt_ix: self.surv_ufailt_ix.clone(),
            risk_enter: self.surv_risk_enter.clone(),
        }
    }

    /// Breslow partial log-likelihood evaluated at `params`.
    pub fn breslow_loglike(&self, params: &Array1<f64>) -> f64 {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();

        // Linear predictor, shifted by its max for numerical stability.
        let mut linpred = self.exog_s.dot(params);
        let lpmax = linpred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        linpred.mapv_inplace(|v| v - lpmax);
        let e_linpred: Vec<f64> = linpred.iter().map(|&v| v.exp()).collect();

        let mut like = 0.0;
        let mut xp0 = 0.0;
        for i in (0..nuft).rev() {
            for &ix in &surv.risk_enter[i] {
                xp0 += e_linpred[ix];
            }
            for &ix in &surv.ufailt_ix[i] {
                like += linpred[ix] - xp0.ln();
            }
            // No risk_exit in the no-entry case (all exit at time-bin 0,
            // handled implicitly by never removing within the backward loop).
        }
        like
    }

    /// Gradient of the Breslow partial log-likelihood at `params`.
    pub fn breslow_gradient(&self, params: &Array1<f64>) -> Array1<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();

        let mut linpred = self.exog_s.dot(params);
        let lpmax = linpred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        linpred.mapv_inplace(|v| v - lpmax);
        let e_linpred: Vec<f64> = linpred.iter().map(|&v| v.exp()).collect();

        let mut grad = Array1::<f64>::zeros(self.k);
        let mut xp0 = 0.0;
        let mut xp1 = Array1::<f64>::zeros(self.k);

        for i in (0..nuft).rev() {
            for &ix in &surv.risk_enter[i] {
                xp0 += e_linpred[ix];
                let row = self.exog_s.row(ix);
                for j in 0..self.k {
                    xp1[j] += e_linpred[ix] * row[j];
                }
            }
            for &ix in &surv.ufailt_ix[i] {
                let row = self.exog_s.row(ix);
                for j in 0..self.k {
                    grad[j] += row[j] - xp1[j] / xp0;
                }
            }
        }
        grad
    }

    /// Hessian of the Breslow partial log-likelihood at `params`.
    ///
    /// Negative-definite at the maximum; its negative is the observed
    /// information used for standard errors.
    pub fn breslow_hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();

        let mut linpred = self.exog_s.dot(params);
        let lpmax = linpred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        linpred.mapv_inplace(|v| v - lpmax);
        let e_linpred: Vec<f64> = linpred.iter().map(|&v| v.exp()).collect();

        let mut hess = Array2::<f64>::zeros((self.k, self.k));
        let mut xp0 = 0.0;
        let mut xp1 = Array1::<f64>::zeros(self.k);
        let mut xp2 = Array2::<f64>::zeros((self.k, self.k));

        for i in (0..nuft).rev() {
            for &ix in &surv.risk_enter[i] {
                let el = e_linpred[ix];
                xp0 += el;
                let row = self.exog_s.row(ix);
                for a in 0..self.k {
                    xp1[a] += el * row[a];
                    for b in 0..self.k {
                        xp2[[a, b]] += el * row[a] * row[b];
                    }
                }
            }
            let mfail = surv.ufailt_ix[i].len() as f64;
            for a in 0..self.k {
                for b in 0..self.k {
                    let val = xp2[[a, b]] / xp0 - (xp1[a] * xp1[b]) / (xp0 * xp0);
                    hess[[a, b]] += mfail * val;
                }
            }
        }
        // The reference returns -hess.
        hess.mapv_inplace(|v| -v);
        hess
    }

    /// Estimate the model by Newton iteration on the partial likelihood.
    pub fn fit(&self) -> Result<PHRegResults> {
        let start = Array1::<f64>::zeros(self.k);
        let opt = newton_stationary(
            &start,
            |b| {
                let f = self.breslow_loglike(b);
                let g = self.breslow_gradient(b);
                let h = self.breslow_hessian(b);
                (f, g, h)
            },
            self.maxiter,
            self.gtol,
        )?;

        let params = opt.x;
        // `breslow_hessian` returns the second derivative of the log partial
        // likelihood (negative definite at the maximum). The observed
        // information is its negative, and the coefficient covariance is the
        // inverse of that information matrix.
        let d2l = self.breslow_hessian(&params);
        let info = d2l.mapv(|v| -v);
        let cov = inv(&info)?;

        let bse = Array1::from_iter((0..self.k).map(|i| cov[[i, i]].sqrt()));
        let tvalues = Array1::from_iter((0..self.k).map(|i| params[i] / bse[i]));
        let pvalues = Array1::from_iter((0..self.k).map(|i| 2.0 * norm_sf(tvalues[i].abs())));
        let llf = self.breslow_loglike(&params);

        Ok(PHRegResults {
            params,
            bse,
            tvalues,
            pvalues,
            cov_params: cov,
            llf,
            converged: opt.converged,
        })
    }
}

/// Build the per-failure-time risk-set indices for a single stratum with no
/// left truncation (entry time 0). Mirrors the reference `PHSurvivalTime`.
fn build_surv(time_s: &[f64], status_s: &[f64]) -> Surv {
    let m = time_s.len();

    // Unique failure times (ascending).
    let mut ft: Vec<f64> = (0..m)
        .filter(|&i| status_s[i].round() as i64 == 1)
        .map(|i| time_s[i])
        .collect();
    ft.sort_by(|a, b| a.total_cmp(b));
    let mut uft: Vec<f64> = Vec::new();
    for &t in &ft {
        if uft.is_empty() || t != *uft.last().unwrap() {
            uft.push(t);
        }
    }
    let nuft = uft.len();

    // ufailt_ix[k] = indices of subjects who fail at uft[k].
    let mut ufailt_ix: Vec<Vec<usize>> = vec![Vec::new(); nuft];
    for (i, &t) in time_s.iter().enumerate().take(m) {
        if status_s[i].round() as i64 == 1 {
            let k = uft.iter().position(|&u| u == t).unwrap();
            ufailt_ix[k].push(i);
        }
    }

    // risk_enter[k] = indices entering the risk set at uft[k]:
    // searchsorted(uft, t, "right") - 1, the last failure time <= t.
    let mut risk_enter: Vec<Vec<usize>> = vec![Vec::new(); nuft];
    for (i, &t) in time_s.iter().enumerate().take(m) {
        // number of uft strictly <= t, minus 1
        let cnt = uft.iter().filter(|&&u| u <= t).count();
        if cnt >= 1 {
            risk_enter[cnt - 1].push(i);
        }
    }

    Surv {
        ufailt_ix,
        risk_enter,
    }
}

/// Results of a fitted Cox proportional-hazards model.
#[derive(Clone, Debug)]
pub struct PHRegResults {
    /// Estimated regression coefficients (log hazard ratios).
    pub params: Array1<f64>,
    /// Standard errors of the coefficients.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values from the standard normal distribution.
    pub pvalues: Array1<f64>,
    /// Coefficient covariance matrix (inverse observed information).
    pub cov_params: Array2<f64>,
    /// Maximized Breslow partial log-likelihood.
    pub llf: f64,
    /// Whether the Newton iteration converged to the gradient tolerance.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradient_zero_at_optimum() {
        // A small data set; after fitting, the gradient should vanish.
        let time = [4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let status = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let exog = array![[0.5_f64], [1.2], [-0.3], [0.8], [0.1], [-1.0], [0.4]];
        let model = PHReg::new(&time, &exog, &status).unwrap();
        let res = model.fit().unwrap();
        assert!(res.converged);
        let g = model.breslow_gradient(&res.params);
        assert!(g.iter().all(|&v| v.abs() < 1e-8));
    }

    #[test]
    fn hessian_matches_numeric_gradient_diff() {
        // The analytic Hessian (negated, as returned) should equal the
        // negative numeric derivative of the gradient.
        let time = [1.0, 2.0, 3.0, 4.0, 5.0];
        let status = [1.0, 1.0, 0.0, 1.0, 1.0];
        let exog = array![[0.2_f64], [-0.5], [1.0], [0.3], [-0.8]];
        let model = PHReg::new(&time, &exog, &status).unwrap();
        let b = array![0.1_f64];
        // breslow_hessian returns the second derivative of the log-likelihood
        // (negative definite at the max), matching the reference convention.
        let h = model.breslow_hessian(&b)[[0, 0]];
        let eps = 1e-6;
        let gp = model.breslow_gradient(&array![0.1 + eps])[0];
        let gm = model.breslow_gradient(&array![0.1 - eps])[0];
        let num_d2l = (gp - gm) / (2.0 * eps);
        assert!((h - num_d2l).abs() < 1e-4, "h={h}, num={num_d2l}");
    }
}
