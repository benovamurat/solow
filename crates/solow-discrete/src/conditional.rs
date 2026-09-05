//! Conditional (fixed-effects) maximum-likelihood count / choice models.
//!
//! These estimators condition the group-specific intercept out of the
//! likelihood, so the within-group fixed effects never have to be estimated.
//! What remains is a *conditional* likelihood that depends only on the slope
//! coefficients `β` — there is **no intercept** in the design.
//!
//! * [`ConditionalLogit`] — conditional logistic regression. Within each group
//!   the conditional likelihood is the Cox/Chamberlain form: given that the
//!   group has `n₁` successes, the probability of the observed success pattern
//!   is `exp(Σ_{i:yᵢ=1} xᵢ·β) / Σ_{S} exp(Σ_{i∈S} xᵢ·β)`, where `S` ranges over
//!   all size-`n₁` subsets of the group. The denominator is evaluated by the
//!   standard `O(n·n₁)` Howard recursion.
//! * [`ConditionalPoisson`] — conditional Poisson regression. Conditioning on
//!   the group total `Σ yᵢ` turns the group contribution into a multinomial,
//!   giving the group log-likelihood
//!   `Σ yᵢ (xᵢ·β) − (Σ yᵢ) log Σ exp(xᵢ·β)`.
//!
//! ## Grouping (matches the reference exactly)
//!
//! Observations are bucketed by group code **in first-appearance order**. A
//! group with no within-group variation in the response (`std(y) == 0`) carries
//! no information about `β` and is dropped, exactly as the reference does; the
//! reported `nobs` counts only the retained observations.
//!
//! ## Estimation
//!
//! The conditional log-likelihood is concave, so a full Newton iteration on the
//! analytic score and a finite-difference Hessian of that score converges to the
//! same optimum as the reference's BFGS fit. The covariance is `(−H)⁻¹` with the
//! Hessian formed as the (central-difference) Jacobian of the analytic score —
//! the same object the reference inverts for its standard errors.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// The conditional family selected for estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Logit,
    Poisson,
}

/// A single retained group: its response, design rows, and sufficient stats.
#[derive(Clone, Debug)]
struct Group {
    /// Response values for the group's rows.
    y: Array1<f64>,
    /// Design rows for the group, `gᵢ × k`.
    x: Array2<f64>,
    /// `xᵀy` for the group (length `k`) — a sufficient statistic.
    xy: Array1<f64>,
    /// Number of successes / total count `Σ yᵢ`.
    n1: f64,
}

/// A conditional fixed-effects model awaiting estimation.
#[derive(Clone, Debug)]
pub struct ConditionalModel {
    groups: Vec<Group>,
    kind: Kind,
    k_params: usize,
    nobs: usize,
    n_groups: usize,
    maxiter: usize,
    gtol: f64,
}

/// Conditional logistic regression for grouped binary data.
#[derive(Clone, Debug)]
pub struct ConditionalLogit(ConditionalModel);

/// Conditional Poisson regression for grouped count data.
#[derive(Clone, Debug)]
pub struct ConditionalPoisson(ConditionalModel);

macro_rules! wrapper {
    ($name:ident, $kind:expr, $doc:literal) => {
        impl $name {
            #[doc = $doc]
            ///
            /// `exog` must **not** contain an intercept column (group effects are
            /// conditioned out). `groups` assigns each row to a group code.
            /// Returns an error on shape mismatch or if a constant column is
            /// present.
            pub fn new(endog: Array1<f64>, exog: Array2<f64>, groups: &[i64]) -> Result<Self> {
                Ok($name(ConditionalModel::new(endog, exog, groups, $kind)?))
            }

            /// Estimate by Newton's method and return the fitted results.
            pub fn fit(&self) -> Result<ConditionalResults> {
                self.0.fit()
            }

            /// The underlying generic conditional model.
            pub fn model(&self) -> &ConditionalModel {
                &self.0
            }
        }
    };
}

wrapper!(
    ConditionalLogit,
    Kind::Logit,
    "Build a conditional logistic regression model from grouped binary data."
);
wrapper!(
    ConditionalPoisson,
    Kind::Poisson,
    "Build a conditional Poisson regression model from grouped count data."
);

impl ConditionalModel {
    fn new(endog: Array1<f64>, exog: Array2<f64>, groups: &[i64], kind: Kind) -> Result<Self> {
        let n = endog.len();
        if exog.nrows() != n {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if groups.len() != n {
            return Err(Error::Shape("endog length != groups length".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        let k = exog.ncols();
        // Reject an intercept column — conditional models must not have one.
        for j in 0..k {
            let col = exog.column(j);
            let Some(&first) = col.iter().next() else {
                continue;
            };
            if first != 0.0 && col.iter().all(|&v| v == first) {
                return Err(Error::Shape(
                    "conditional models should not have an intercept in exog".into(),
                ));
            }
        }

        // Bucket rows by group code in first-appearance order.
        let mut order: Vec<i64> = Vec::new();
        let mut row_ix: std::collections::HashMap<i64, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            row_ix.entry(g).or_insert_with(|| {
                order.push(g);
                Vec::new()
            });
            row_ix.get_mut(&g).unwrap().push(i);
        }

        let mut grps: Vec<Group> = Vec::new();
        let mut nobs = 0usize;
        for g in order {
            let ix = &row_ix[&g];
            // Drop groups with no within-group variance in the response.
            let y0 = endog[ix[0]];
            if ix.iter().all(|&i| endog[i] == y0) {
                continue;
            }
            let gi = ix.len();
            let mut y = Array1::<f64>::zeros(gi);
            let mut x = Array2::<f64>::zeros((gi, k));
            for (r, &i) in ix.iter().enumerate() {
                y[r] = endog[i];
                for c in 0..k {
                    x[[r, c]] = exog[[i, c]];
                }
            }
            let xy = x.t().dot(&y);
            let n1 = y.sum();
            nobs += gi;
            grps.push(Group { y, x, xy, n1 });
        }

        if grps.is_empty() {
            return Err(Error::Shape(
                "no groups with within-group variation remain".into(),
            ));
        }

        let n_groups = grps.len();
        Ok(ConditionalModel {
            groups: grps,
            kind,
            k_params: k,
            nobs,
            n_groups,
            maxiter: 100,
            gtol: 1e-12,
        })
    }

    /// Number of observations retained after dropping invariant groups.
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    /// Number of groups retained.
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// Conditional log-likelihood at `params`.
    fn loglike(&self, params: &Array1<f64>) -> f64 {
        match self.kind {
            Kind::Logit => self.groups.iter().map(|g| logit_grp_ll(g, params)).sum(),
            Kind::Poisson => self.groups.iter().map(|g| poisson_grp_ll(g, params)).sum(),
        }
    }

    /// Analytic conditional score (gradient) at `params`.
    fn score(&self, params: &Array1<f64>) -> Array1<f64> {
        let mut s = Array1::<f64>::zeros(self.k_params);
        match self.kind {
            Kind::Logit => {
                for g in &self.groups {
                    s += &logit_grp_score(g, params);
                }
            }
            Kind::Poisson => {
                for g in &self.groups {
                    s += &poisson_grp_score(g, params);
                }
            }
        }
        s
    }

    /// Hessian as the central-difference Jacobian of the analytic score.
    ///
    /// This is exactly the object the reference inverts to form standard errors
    /// (it differentiates the same analytic score). Because the score is exact
    /// the finite-difference Jacobian agrees with the true Hessian to ~1e-9.
    fn hessian(&self, params: &Array1<f64>) -> Array2<f64> {
        let k = self.k_params;
        // Central-difference Jacobian of the (vector) score.
        let mut h = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let step = 1e-6 * (1.0 + params[j].abs());
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[j] += step;
            pm[j] -= step;
            let sp = self.score(&pp);
            let sm = self.score(&pm);
            for i in 0..k {
                h[[i, j]] = (sp[i] - sm[i]) / (2.0 * step);
            }
        }
        // Symmetrize (the true Hessian is symmetric).
        for i in 0..k {
            for j in (i + 1)..k {
                let v = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = v;
                h[[j, i]] = v;
            }
        }
        h
    }

    fn start(&self) -> Array1<f64> {
        Array1::<f64>::zeros(self.k_params)
    }

    /// Estimate by full Newton steps and assemble [`ConditionalResults`].
    pub fn fit(&self) -> Result<ConditionalResults> {
        let fgh = |p: &Array1<f64>| {
            let f = -self.loglike(p);
            let g = self.score(p).mapv(|v| -v);
            let h = self.hessian(p).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&self.start(), fgh, self.maxiter, self.gtol)?;
        let params = opt.x;
        let h = self.hessian(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;
        Ok(ConditionalResults::new(self, params, cov, opt.converged))
    }
}

// --------------------------------------------------------------------------- //
//  Per-group conditional-logit pieces (Howard recursion)                      //
// --------------------------------------------------------------------------- //

/// Denominator `Σ_{|S|=n₁} Π_{i∈S} exp(xᵢ·β)` for a group, via the recursion
/// `f(t,k) = f(t−1,k) + f(t−1,k−1)·e_{t}` with `f(t,0)=1`, `f(t,k)=0` for `t<k`.
fn logit_denom(exb: &[f64], n1: usize) -> f64 {
    let t = exb.len();
    // dp[k] holds f(·, k); iterate t from 1..=t.
    let mut dp = vec![0.0f64; n1 + 1];
    dp[0] = 1.0;
    for (i, &e) in exb.iter().enumerate().take(t) {
        // Update in decreasing k to reuse f(t-1, k-1).
        let kmax = (i + 1).min(n1);
        for k in (1..=kmax).rev() {
            dp[k] += dp[k - 1] * e;
        }
    }
    dp[n1]
}

/// Group log-likelihood for conditional logit: `xy·β − log(denom)`.
fn logit_grp_ll(g: &Group, params: &Array1<f64>) -> f64 {
    let exb: Vec<f64> = g.x.dot(params).iter().map(|v| v.exp()).collect();
    let n1 = g.n1.round() as usize;
    let denom = logit_denom(&exb, n1);
    g.xy.dot(params) - denom.ln()
}

/// Group score for conditional logit: `xy − (∇denom)/denom`.
///
/// `∇denom` is accumulated by the value/gradient recursion: with
/// `s(t,k) = (value, grad)`, `s(t,k) = s(t−1,k) + e_t · (s(t−1,k−1) shifted)`,
/// where the gradient also gains `value(t−1,k−1) · e_t · x_t`.
fn logit_grp_score(g: &Group, params: &Array1<f64>) -> Array1<f64> {
    let k = params.len();
    let xb = g.x.dot(params);
    let exb: Vec<f64> = xb.iter().map(|v| v.exp()).collect();
    let n1 = g.n1.round() as usize;
    let t = exb.len();

    // val[kk] = f(·, kk); grad[kk] = ∇ f(·, kk).
    let mut val = vec![0.0f64; n1 + 1];
    let mut grad: Vec<Array1<f64>> = (0..=n1).map(|_| Array1::<f64>::zeros(k)).collect();
    val[0] = 1.0;
    for (i, &e) in exb.iter().enumerate().take(t) {
        let xi = g.x.row(i);
        let kmax = (i + 1).min(n1);
        for kk in (1..=kmax).rev() {
            // new grad[kk] += e * grad[kk-1] + (val[kk-1] * e) * x_i
            let prev_val = val[kk - 1];
            // grad[kk] += e * grad[kk-1]
            let add = &grad[kk - 1] * e;
            grad[kk] = &grad[kk] + &add;
            // grad[kk] += val[kk-1] * e * x_i
            for c in 0..k {
                grad[kk][c] += prev_val * e * xi[c];
            }
            // val[kk] += e * val[kk-1]
            val[kk] += e * prev_val;
        }
    }
    let denom = val[n1];
    let gdenom = &grad[n1];
    &g.xy - &(gdenom / denom)
}

// --------------------------------------------------------------------------- //
//  Per-group conditional-Poisson pieces                                       //
// --------------------------------------------------------------------------- //

/// Group log-likelihood for conditional Poisson: `Σ yᵢ xᵢβ − (Σy) log Σ e^{xᵢβ}`.
fn poisson_grp_ll(g: &Group, params: &Array1<f64>) -> f64 {
    let xb = g.x.dot(params);
    let mut ll = g.y.dot(&xb);
    let s: f64 = xb.iter().map(|v| v.exp()).sum();
    ll -= g.n1 * s.ln();
    ll
}

/// Group score for conditional Poisson: `xᵀy − (Σy) (Σ e^{xᵢβ} xᵢ)/(Σ e^{xᵢβ})`.
fn poisson_grp_score(g: &Group, params: &Array1<f64>) -> Array1<f64> {
    let xb = g.x.dot(params);
    let exb = xb.mapv(f64::exp);
    let s: f64 = exb.sum();
    // weighted = Σ e^{xᵢβ} xᵢ
    let weighted = g.x.t().dot(&exb);
    &g.xy - &(&weighted * (g.n1 / s))
}

/// The fitted result of a conditional fixed-effects model.
#[derive(Clone, Debug)]
pub struct ConditionalResults {
    /// Estimated slope coefficients `β̂` (no intercept).
    pub params: Array1<f64>,
    /// Standard errors `√diag((−H)⁻¹)`.
    pub bse: Array1<f64>,
    /// z-statistics `params / bse`.
    pub tvalues: Array1<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`.
    pub pvalues: Array1<f64>,

    /// Number of observations retained.
    pub nobs: f64,
    /// Number of groups retained.
    pub n_groups: usize,

    /// Maximized conditional log-likelihood.
    pub llf: f64,

    /// Coefficient covariance `(−H)⁻¹`.
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,
}

impl ConditionalResults {
    fn new(
        model: &ConditionalModel,
        params: Array1<f64>,
        cov_params: Array2<f64>,
        converged: bool,
    ) -> ConditionalResults {
        let k = params.len();
        let mut bse = Array1::<f64>::zeros(k);
        for i in 0..k {
            bse[i] = cov_params[[i, i]].sqrt();
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));
        let llf = model.loglike(&params);

        ConditionalResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs: model.nobs as f64,
            n_groups: model.n_groups,
            llf,
            cov_params,
            converged,
        }
    }

    /// Confidence interval for each coefficient at level `1 − alpha` (normal).
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let q = norm_ppf(1.0 - alpha / 2.0);
        let k = self.params.len();
        let mut out = Array2::<f64>::zeros((k, 2));
        for i in 0..k {
            out[[i, 0]] = self.params[i] - q * self.bse[i];
            out[[i, 1]] = self.params[i] + q * self.bse[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use solow_optimize::approx_fprime;

    fn data() -> (Array1<f64>, Array2<f64>, Vec<i64>) {
        // 6 groups of 3; binary-ish response with within-group variation.
        let y = array![
            0.0, 1.0, 1.0, // g0
            1.0, 0.0, 1.0, // g1
            0.0, 0.0, 1.0, // g2
            1.0, 1.0, 0.0, // g3
            0.0, 1.0, 0.0, // g4
            1.0, 0.0, 0.0, // g5
        ];
        let xcol1 = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3,
            0.2, -0.6
        ];
        let xcol2 = array![
            -0.3, 0.7, -1.1, 0.4, 1.2, -0.5, 0.8, -0.2, 0.6, -1.3, 0.9, 0.1, -0.7, 0.5, -0.4, 1.0,
            -0.8, 0.3
        ];
        let mut x = Array2::<f64>::zeros((18, 2));
        x.column_mut(0).assign(&xcol1);
        x.column_mut(1).assign(&xcol2);
        let groups: Vec<i64> = (0..6).flat_map(|g| [g, g, g]).collect();
        (y, x, groups)
    }

    #[test]
    fn logit_score_zero_at_optimum() {
        let (y, x, g) = data();
        let m = ConditionalLogit::new(y, x, &g).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let s = m.0.score(&res.params);
        assert!(s.dot(&s).sqrt() < 1e-8, "score {}", s.dot(&s).sqrt());
    }

    #[test]
    fn logit_analytic_score_matches_fd() {
        let (y, x, g) = data();
        let m = ConditionalLogit::new(y, x, &g).unwrap();
        let p = array![0.2, -0.3];
        let s = m.0.score(&p);
        let fd = approx_fprime(&p, |q| m.0.loglike(q));
        for i in 0..p.len() {
            assert_abs_diff_eq!(s[i], fd[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn poisson_score_zero_at_optimum() {
        let y = array![
            2.0, 0.0, 3.0, // g0
            1.0, 4.0, 0.0, // g1
            5.0, 1.0, 2.0, // g2
            0.0, 3.0, 1.0, // g3
            2.0, 6.0, 1.0, // g4
            1.0, 0.0, 4.0, // g5
        ];
        let xcol1 = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3,
            0.2, -0.6
        ];
        let xcol2 = array![
            -0.3, 0.7, -1.1, 0.4, 1.2, -0.5, 0.8, -0.2, 0.6, -1.3, 0.9, 0.1, -0.7, 0.5, -0.4, 1.0,
            -0.8, 0.3
        ];
        let mut x = Array2::<f64>::zeros((18, 2));
        x.column_mut(0).assign(&xcol1);
        x.column_mut(1).assign(&xcol2);
        let groups: Vec<i64> = (0..6).flat_map(|g| [g, g, g]).collect();
        let m = ConditionalPoisson::new(y, x, &groups).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let s = m.0.score(&res.params);
        assert!(s.dot(&s).sqrt() < 1e-8, "score {}", s.dot(&s).sqrt());
    }

    #[test]
    fn poisson_analytic_score_matches_fd() {
        let y = array![
            2.0, 0.0, 3.0, 1.0, 4.0, 0.0, 5.0, 1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 6.0, 1.0, 1.0, 0.0,
            4.0
        ];
        let xcol1 = array![
            0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4, 1.1, -0.9, 0.8, -0.3,
            0.2, -0.6
        ];
        let mut x = Array2::<f64>::zeros((18, 1));
        x.column_mut(0).assign(&xcol1);
        let groups: Vec<i64> = (0..6).flat_map(|g| [g, g, g]).collect();
        let m = ConditionalPoisson::new(y, x, &groups).unwrap();
        let p = array![0.25];
        let s = m.0.score(&p);
        let fd = approx_fprime(&p, |q| m.0.loglike(q));
        assert_abs_diff_eq!(s[0], fd[0], epsilon = 1e-6);
    }

    #[test]
    fn rejects_intercept() {
        let (y, _x, g) = data();
        let xc = Array2::<f64>::ones((18, 1));
        assert!(ConditionalLogit::new(y, xc, &g).is_err());
    }
}
