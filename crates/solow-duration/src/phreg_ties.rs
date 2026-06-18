//! Cox proportional-hazards regression with selectable tie handling.
//!
//! [`PHRegTies`] extends the Breslow-only [`crate::PHReg`] with an explicit
//! choice between the **Breslow** and **Efron** approximations for tied event
//! times via the [`Ties`] enum. The Efron correction is the more accurate of
//! the two when several subjects fail at exactly the same time and is the
//! default in many statistical packages.
//!
//! The model maximizes the partial log-likelihood with a Newton step (driving
//! the gradient to zero), mirroring the reference
//! `hazard_regression.PHReg` with `ties='breslow'` or `ties='efron'` for the
//! single-stratum, no-entry, no-offset case. It exposes the coefficient vector
//! [`PHRegTiesResults::params`], its standard errors
//! [`PHRegTiesResults::bse`] (from the inverse observed information),
//! z-statistics [`PHRegTiesResults::tvalues`], two-sided normal p-values
//! [`PHRegTiesResults::pvalues`], and the maximized partial log-likelihood
//! [`PHRegTiesResults::llf`].

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// Method for handling tied event times in the Cox partial likelihood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ties {
    /// Breslow approximation (simplest; assumes the risk set is unchanged
    /// while the tied failures occur).
    Breslow,
    /// Efron approximation (more accurate; progressively deflates the tied
    /// failures' contribution to the risk-set denominator).
    Efron,
}

/// Pre-computed risk-set bookkeeping for a single (unstratified) sample.
///
/// All indices reference rows of the *filtered, time-sorted* covariate matrix.
struct Surv {
    /// `ufailt_ix[k]` = indices of subjects failing at the k-th distinct
    /// failure time (sorted ascending).
    ufailt_ix: Vec<Vec<usize>>,
    /// `risk_enter[k]` = indices of subjects entering the risk set at the
    /// k-th distinct failure time.
    risk_enter: Vec<Vec<usize>>,
}

/// A Cox proportional-hazards model with a selectable tie-handling method.
#[derive(Clone, Debug)]
pub struct PHRegTies {
    /// Filtered, time-sorted covariate matrix (rows = informative subjects).
    exog_s: Array2<f64>,
    /// Number of covariates.
    k: usize,
    surv_ufailt_ix: Vec<Vec<usize>>,
    surv_risk_enter: Vec<Vec<usize>>,
    ties: Ties,
    maxiter: usize,
    gtol: f64,
}

impl PHRegTies {
    /// Build a Cox PH model with the chosen tie-handling method.
    ///
    /// `time[i]` is the event or censoring time, `exog` has one row per subject
    /// (covariates in columns, no implicit intercept — the Cox baseline hazard
    /// absorbs it), and `status[i]` is `1.0` for an observed event and `0.0`
    /// for right-censoring.
    pub fn new(time: &[f64], exog: &Array2<f64>, status: &[f64], ties: Ties) -> Result<Self> {
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
        // entry time 0 (no left truncation): drop subjects censored strictly
        // before the first failure time.
        let mut first_failure = f64::INFINITY;
        for i in 0..n {
            if status[i].round() as i64 == 1 && time[i] < first_failure {
                first_failure = time[i];
            }
        }
        let mut rows: Vec<usize> = (0..n).filter(|&i| time[i] >= first_failure).collect();

        // Order by time within the stratum (stable sort, matching argsort).
        rows.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

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

        Ok(PHRegTies {
            exog_s,
            k,
            surv_ufailt_ix: surv.ufailt_ix,
            surv_risk_enter: surv.risk_enter,
            ties,
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

    /// Max-shifted exponentiated linear predictor (numerical stability).
    fn e_linpred(&self, params: &Array1<f64>) -> (Array1<f64>, Vec<f64>) {
        let mut linpred = self.exog_s.dot(params);
        let lpmax = linpred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        linpred.mapv_inplace(|v| v - lpmax);
        let e_linpred: Vec<f64> = linpred.iter().map(|&v| v.exp()).collect();
        (linpred, e_linpred)
    }

    /// Breslow partial log-likelihood evaluated at `params`.
    pub fn breslow_loglike(&self, params: &Array1<f64>) -> f64 {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (linpred, e_linpred) = self.e_linpred(params);

        let mut like = 0.0;
        let mut xp0 = 0.0;
        for i in (0..nuft).rev() {
            for &ix in &surv.risk_enter[i] {
                xp0 += e_linpred[ix];
            }
            for &ix in &surv.ufailt_ix[i] {
                like += linpred[ix] - xp0.ln();
            }
        }
        like
    }

    /// Efron partial log-likelihood evaluated at `params`.
    pub fn efron_loglike(&self, params: &Array1<f64>) -> f64 {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (linpred, e_linpred) = self.e_linpred(params);

        let mut like = 0.0;
        let mut xp0 = 0.0;
        for i in (0..nuft).rev() {
            for &ix in &surv.risk_enter[i] {
                xp0 += e_linpred[ix];
            }
            let fail = &surv.ufailt_ix[i];
            let xp0f: f64 = fail.iter().map(|&ix| e_linpred[ix]).sum();
            for &ix in fail {
                like += linpred[ix];
            }
            let m = fail.len();
            for j in 0..m {
                let jf = j as f64 / m as f64;
                like -= (xp0 - jf * xp0f).ln();
            }
        }
        like
    }

    /// Partial log-likelihood under the configured tie-handling method.
    pub fn loglike(&self, params: &Array1<f64>) -> f64 {
        match self.ties {
            Ties::Breslow => self.breslow_loglike(params),
            Ties::Efron => self.efron_loglike(params),
        }
    }

    /// Breslow gradient of the partial log-likelihood at `params`.
    pub fn breslow_gradient(&self, params: &Array1<f64>) -> Array1<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (_linpred, e_linpred) = self.e_linpred(params);

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

    /// Efron gradient of the partial log-likelihood at `params`.
    pub fn efron_gradient(&self, params: &Array1<f64>) -> Array1<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (_linpred, e_linpred) = self.e_linpred(params);

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
            let fail = &surv.ufailt_ix[i];
            if fail.is_empty() {
                continue;
            }
            // Tied-failure accumulators.
            let xp0f: f64 = fail.iter().map(|&ix| e_linpred[ix]).sum();
            let mut xp1f = Array1::<f64>::zeros(self.k);
            for &ix in fail {
                let row = self.exog_s.row(ix);
                for j in 0..self.k {
                    xp1f[j] += e_linpred[ix] * row[j];
                    grad[j] += row[j];
                }
            }
            let m = fail.len();
            for jj in 0..m {
                let jf = jj as f64 / m as f64;
                let denom = xp0 - jf * xp0f;
                for j in 0..self.k {
                    grad[j] -= (xp1[j] - jf * xp1f[j]) / denom;
                }
            }
        }
        grad
    }

    /// Gradient under the configured tie-handling method.
    pub fn gradient(&self, params: &Array1<f64>) -> Array1<f64> {
        match self.ties {
            Ties::Breslow => self.breslow_gradient(params),
            Ties::Efron => self.efron_gradient(params),
        }
    }

    /// Breslow Hessian of the partial log-likelihood at `params`.
    ///
    /// Negative-definite at the maximum; its negative is the observed
    /// information used for standard errors.
    pub fn breslow_hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (_linpred, e_linpred) = self.e_linpred(params);

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
        hess.mapv_inplace(|v| -v);
        hess
    }

    /// Efron Hessian of the partial log-likelihood at `params`.
    ///
    /// Negative-definite at the maximum; its negative is the observed
    /// information used for standard errors.
    pub fn efron_hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let surv = self.surv();
        let nuft = surv.ufailt_ix.len();
        let (_linpred, e_linpred) = self.e_linpred(params);

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
            let fail = &surv.ufailt_ix[i];
            if fail.is_empty() {
                continue;
            }
            // Tied-failure accumulators.
            let xp0f: f64 = fail.iter().map(|&ix| e_linpred[ix]).sum();
            let mut xp1f = Array1::<f64>::zeros(self.k);
            let mut xp2f = Array2::<f64>::zeros((self.k, self.k));
            for &ix in fail {
                let el = e_linpred[ix];
                let row = self.exog_s.row(ix);
                for a in 0..self.k {
                    xp1f[a] += el * row[a];
                    for b in 0..self.k {
                        xp2f[[a, b]] += el * row[a] * row[b];
                    }
                }
            }
            let m = fail.len();
            // hess += xp2 * sum(1/c0) - xp2f * sum(J/c0)
            //       - sum_j outer(mat_j, mat_j), mat_j = (xp1 - J_j*xp1f)/c0_j
            let mut sum_inv = 0.0;
            let mut sum_jinv = 0.0;
            // Per-tie "mat" rows: accumulate outer product sum directly.
            let mut outer_acc = Array2::<f64>::zeros((self.k, self.k));
            for jj in 0..m {
                let jf = jj as f64 / m as f64;
                let c0 = xp0 - jf * xp0f;
                sum_inv += 1.0 / c0;
                sum_jinv += jf / c0;
                // mat_j vector
                let mut matj = Array1::<f64>::zeros(self.k);
                for a in 0..self.k {
                    matj[a] = (xp1[a] - jf * xp1f[a]) / c0;
                }
                for a in 0..self.k {
                    for b in 0..self.k {
                        outer_acc[[a, b]] += matj[a] * matj[b];
                    }
                }
            }
            for a in 0..self.k {
                for b in 0..self.k {
                    hess[[a, b]] +=
                        xp2[[a, b]] * sum_inv - xp2f[[a, b]] * sum_jinv - outer_acc[[a, b]];
                }
            }
        }
        hess.mapv_inplace(|v| -v);
        hess
    }

    /// Hessian under the configured tie-handling method.
    pub fn hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        match self.ties {
            Ties::Breslow => self.breslow_hessian(params),
            Ties::Efron => self.efron_hessian(params),
        }
    }

    /// Estimate the model by Newton iteration on the partial likelihood.
    pub fn fit(&self) -> Result<PHRegTiesResults> {
        let start = Array1::<f64>::zeros(self.k);
        let opt = newton_stationary(
            &start,
            |b| {
                let f = self.loglike(b);
                let g = self.gradient(b);
                let h = self.hessian(b);
                (f, g, h)
            },
            self.maxiter,
            self.gtol,
        )?;

        let params = opt.x;
        // `hessian` returns the second derivative of the log partial likelihood
        // (negative definite at the maximum). The observed information is its
        // negative, and the coefficient covariance is the inverse information.
        let d2l = self.hessian(&params);
        let info = d2l.mapv(|v| -v);
        let cov = inv(&info)?;

        let bse = Array1::from_iter((0..self.k).map(|i| cov[[i, i]].sqrt()));
        let tvalues = Array1::from_iter((0..self.k).map(|i| params[i] / bse[i]));
        let pvalues = Array1::from_iter((0..self.k).map(|i| 2.0 * norm_sf(tvalues[i].abs())));
        let llf = self.loglike(&params);

        Ok(PHRegTiesResults {
            params,
            bse,
            tvalues,
            pvalues,
            cov_params: cov,
            llf,
            converged: opt.converged,
            ties: self.ties,
        })
    }
}

/// Build the per-failure-time risk-set indices for a single stratum with no
/// left truncation (entry time 0).
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

/// Results of a fitted Cox proportional-hazards model with selectable ties.
#[derive(Clone, Debug)]
pub struct PHRegTiesResults {
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
    /// Maximized partial log-likelihood.
    pub llf: f64,
    /// Whether the Newton iteration converged to the gradient tolerance.
    pub converged: bool,
    /// The tie-handling method used.
    pub ties: Ties,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn efron_gradient_zero_at_optimum() {
        let time = [4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let status = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let exog = array![[0.5_f64], [1.2], [-0.3], [0.8], [0.1], [-1.0], [0.4]];
        let model = PHRegTies::new(&time, &exog, &status, Ties::Efron).unwrap();
        let res = model.fit().unwrap();
        assert!(res.converged);
        let g = model.efron_gradient(&res.params);
        assert!(g.iter().all(|&v| v.abs() < 1e-8));
    }

    #[test]
    fn efron_hessian_matches_numeric_gradient_diff() {
        // Tied failure times exercise the Efron correction.
        let time = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let status = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let exog = array![[0.2_f64], [-0.5], [1.0], [0.3], [-0.8], [0.6]];
        let model = PHRegTies::new(&time, &exog, &status, Ties::Efron).unwrap();
        let b = array![0.15_f64];
        let h = model.efron_hessian(&b)[[0, 0]];
        let eps = 1e-6;
        let gp = model.efron_gradient(&array![0.15 + eps])[0];
        let gm = model.efron_gradient(&array![0.15 - eps])[0];
        let num_d2l = (gp - gm) / (2.0 * eps);
        assert!((h - num_d2l).abs() < 1e-4, "h={h}, num={num_d2l}");
    }

    #[test]
    fn breslow_matches_existing_phreg() {
        // Without ties, Breslow and Efron coincide; with the same data the new
        // Breslow path should reproduce the canonical Cox estimate.
        let time = [1.0, 2.0, 3.0, 4.0, 5.0];
        let status = [1.0, 1.0, 0.0, 1.0, 1.0];
        let exog = array![[0.2_f64], [-0.5], [1.0], [0.3], [-0.8]];
        let breslow = PHRegTies::new(&time, &exog, &status, Ties::Breslow)
            .unwrap()
            .fit()
            .unwrap();
        let efron = PHRegTies::new(&time, &exog, &status, Ties::Efron)
            .unwrap()
            .fit()
            .unwrap();
        // No tied event times -> identical results.
        assert!((breslow.params[0] - efron.params[0]).abs() < 1e-10);
    }
}
