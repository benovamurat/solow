//! Multinomial logistic regression (softmax) fit by Newton's method.
//!
//! [`MNLogit`] models an unordered categorical response with `K` levels
//! `0, 1, …, K−1`. Level `0` is the baseline: its linear predictor is fixed at
//! zero, and one coefficient vector is estimated for each remaining level. With
//! `k_exog` regressors the coefficient matrix is therefore `k_exog × (K−1)`,
//! oriented exactly like the canonical reference (column `j` holds the
//! coefficients for response level `j + 1` relative to the baseline).
//!
//! For observation `i` with regressors `xᵢ` the choice probabilities are the
//! softmax
//!
//! ```text
//! P(yᵢ = 0) = 1 / D,   P(yᵢ = k) = exp(xᵢ · β_k) / D   (k = 1 … K−1)
//! ```
//!
//! with `D = 1 + Σ_{k≥1} exp(xᵢ · β_k)`. The log-likelihood, its score, and its
//! (block) Hessian are available in closed form, so a full Newton iteration
//! converges to the maximum-likelihood estimate to machine precision.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_core::tools::{ensure_all_finite, ensure_all_finite_2d};
use solow_distributions::{chi2_sf, norm_ppf, norm_sf};
use solow_linalg::inv;
use solow_optimize::newton_stationary;

/// A multinomial logit model awaiting estimation.
///
/// Construct with [`MNLogit::new`]; estimate with [`MNLogit::fit`].
#[derive(Clone, Debug)]
pub struct MNLogit {
    /// Integer-coded response in `0 … K−1`, stored as `f64`.
    endog: Array1<f64>,
    /// Design matrix `n × k_exog` (include an intercept column if wanted).
    exog: Array2<f64>,
    /// Number of response levels `K`.
    k_levels: usize,
    /// Whether `exog` contains a constant column (0 or 1).
    k_constant: usize,
    maxiter: usize,
    gtol: f64,
}

impl MNLogit {
    /// Build a multinomial logit model.
    ///
    /// `endog` must contain integer category codes `0 … K−1` (as `f64`); `K` is
    /// inferred as `max(endog) + 1`. `exog` should already contain an intercept
    /// column if one is wanted (see `solow_core::tools::add_constant`). Returns
    /// an error when the shapes are inconsistent or the response is degenerate.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        ensure_all_finite(&endog.view(), "endog")?;
        ensure_all_finite_2d(&exog.view(), "exog")?;
        // Determine K and validate the codes are non-negative integers.
        let mut kmax = 0usize;
        for &v in endog.iter() {
            if v < 0.0 || v.fract() != 0.0 {
                return Err(Error::Shape(
                    "endog must hold integer category codes".into(),
                ));
            }
            let c = v as usize;
            if c > kmax {
                kmax = c;
            }
        }
        let k_levels = kmax + 1;
        if k_levels < 2 {
            return Err(Error::Shape("MNLogit needs at least two categories".into()));
        }
        let k_constant = detect_k_constant(&exog);
        Ok(MNLogit {
            endog,
            exog,
            k_levels,
            k_constant,
            maxiter: 100,
            gtol: 1e-12,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// Number of response levels `K`.
    pub fn k_levels(&self) -> usize {
        self.k_levels
    }

    /// Per-observation choice-probability matrix `n × K` at coefficient matrix
    /// `beta` (shape `k_exog × (K−1)`), computed in a numerically stable way.
    fn probs(&self, beta: &Array2<f64>) -> Array2<f64> {
        let n = self.endog.len();
        let k = self.k_levels;
        let xb = self.exog.dot(beta); // n × (K−1)
        let mut p = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            // Build the full eta row including the baseline 0.
            let mut mx = 0.0; // baseline level contributes eta = 0
            for j in 0..k - 1 {
                if xb[[i, j]] > mx {
                    mx = xb[[i, j]];
                }
            }
            let mut denom = (-mx).exp(); // baseline: exp(0 - mx)
            p[[i, 0]] = denom;
            for j in 0..k - 1 {
                let e = (xb[[i, j]] - mx).exp();
                p[[i, j + 1]] = e;
                denom += e;
            }
            for c in 0..k {
                p[[i, c]] /= denom;
            }
        }
        p
    }

    /// Log-likelihood at coefficient matrix `beta`.
    fn loglike_mat(&self, beta: &Array2<f64>) -> f64 {
        let p = self.probs(beta);
        let mut ll = 0.0;
        for i in 0..self.endog.len() {
            let c = self.endog[i] as usize;
            ll += p[[i, c]].max(1e-300).ln();
        }
        ll
    }

    /// Score with respect to the flattened (category-major) parameter vector.
    ///
    /// The flat layout is `[β_1; β_2; …; β_{K−1}]`, each `β_k` a length-`k_exog`
    /// block — matching the reference's Fortran-order ravel of the `k_exog ×
    /// (K−1)` matrix.
    fn score_flat(&self, beta: &Array2<f64>) -> Array1<f64> {
        let p = self.probs(beta);
        let ke = self.exog.ncols();
        let k = self.k_levels;
        let mut g = Array1::<f64>::zeros(ke * (k - 1));
        for kk in 1..k {
            // residual r_i = d_{i,kk} - P_{i,kk}
            let mut r = Array1::<f64>::zeros(self.endog.len());
            for i in 0..self.endog.len() {
                let d = if self.endog[i] as usize == kk {
                    1.0
                } else {
                    0.0
                };
                r[i] = d - p[[i, kk]];
            }
            let block = self.exog.t().dot(&r); // length k_exog
            let off = (kk - 1) * ke;
            for a in 0..ke {
                g[off + a] = block[a];
            }
        }
        g
    }

    /// Hessian with respect to the flattened (category-major) parameter vector.
    ///
    /// Block `(k, l)` (each `1 … K−1`) equals `−Xᵀ diag(P_k(δ_{kl} − P_l)) X`.
    fn hessian_flat(&self, beta: &Array2<f64>) -> Array2<f64> {
        let p = self.probs(beta);
        let ke = self.exog.ncols();
        let k = self.k_levels;
        let n = self.endog.len();
        let dim = ke * (k - 1);
        let mut h = Array2::<f64>::zeros((dim, dim));
        for kk in 1..k {
            for ll in 1..k {
                let delta = if kk == ll { 1.0 } else { 0.0 };
                // weights w_i = P_{i,kk} (δ - P_{i,ll})
                let mut w = Array1::<f64>::zeros(n);
                for i in 0..n {
                    w[i] = p[[i, kk]] * (delta - p[[i, ll]]);
                }
                let roff = (kk - 1) * ke;
                let coff = (ll - 1) * ke;
                for a in 0..ke {
                    for b in 0..ke {
                        let mut s = 0.0;
                        for i in 0..n {
                            s += self.exog[[i, a]] * w[i] * self.exog[[i, b]];
                        }
                        h[[roff + a, coff + b]] = -s;
                    }
                }
            }
        }
        h
    }

    /// Reshape a flat (category-major) parameter vector into the `k_exog ×
    /// (K−1)` coefficient matrix.
    fn unflatten(&self, flat: &Array1<f64>) -> Array2<f64> {
        let ke = self.exog.ncols();
        let k = self.k_levels;
        let mut b = Array2::<f64>::zeros((ke, k - 1));
        for kk in 0..k - 1 {
            for a in 0..ke {
                b[[a, kk]] = flat[kk * ke + a];
            }
        }
        b
    }

    /// Intercept-only (null) log-likelihood, `Σ_k n_k log(n_k / n)`.
    fn llnull(&self) -> f64 {
        let n = self.endog.len() as f64;
        let mut counts = vec![0.0f64; self.k_levels];
        for &v in self.endog.iter() {
            counts[v as usize] += 1.0;
        }
        let mut ll = 0.0;
        for c in counts {
            if c > 0.0 {
                ll += c * (c / n).ln();
            }
        }
        ll
    }

    /// Estimate by full Newton steps and assemble [`MNLogitResults`].
    pub fn fit(&self) -> Result<MNLogitResults> {
        let ke = self.exog.ncols();
        let dim = ke * (self.k_levels - 1);
        let start = Array1::<f64>::zeros(dim);
        let fgh = |flat: &Array1<f64>| {
            let b = self.unflatten(flat);
            // Newton drives the gradient to zero; pass the *negative*
            // log-likelihood's value/score/hessian so the located stationary
            // point is the maximizer.
            let f = -self.loglike_mat(&b);
            let g = self.score_flat(&b).mapv(|v| -v);
            let h = self.hessian_flat(&b).mapv(|v| -v);
            (f, g, h)
        };
        let opt = newton_stationary(&start, fgh, self.maxiter, self.gtol)?;
        let params = self.unflatten(&opt.x);

        // Covariance = (−H)^{-1} at the optimum (observed information), in the
        // flat category-major ordering.
        let h = self.hessian_flat(&params);
        let neg_h = h.mapv(|v| -v);
        let cov = inv(&neg_h)?;

        Ok(MNLogitResults::new(self, params, cov, opt.converged))
    }
}

fn detect_k_constant(exog: &Array2<f64>) -> usize {
    let (_, k) = exog.dim();
    for j in 0..k {
        let col = exog.column(j);
        let Some(&first) = col.iter().next() else {
            continue;
        };
        if first != 0.0 && col.iter().all(|&v| v == first) {
            return 1;
        }
    }
    0
}

/// The fitted result of a multinomial logit model.
#[derive(Clone, Debug)]
pub struct MNLogitResults {
    /// Estimated coefficients, shape `k_exog × (K−1)` (baseline level 0).
    pub params: Array2<f64>,
    /// Standard errors, same shape as `params`.
    pub bse: Array2<f64>,
    /// z-statistics `params / bse`, same shape as `params`.
    pub tvalues: Array2<f64>,
    /// Two-sided p-values `2·Φ̄(|z|)`, same shape as `params`.
    pub pvalues: Array2<f64>,

    /// Number of observations.
    pub nobs: f64,
    /// Number of response levels `K`.
    pub k_levels: usize,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom, `(k_exog − k_constant)·(K−1)`.
    pub df_model: f64,
    /// Residual degrees of freedom, `nobs − (df_model + (K−1))`.
    pub df_resid: f64,

    /// Maximized log-likelihood.
    pub llf: f64,
    /// Intercept-only (null) log-likelihood.
    pub llnull: f64,
    /// Likelihood-ratio statistic `2(llf − llnull)`.
    pub llr: f64,
    /// p-value of the LR statistic, `χ²_{df_model}` survival.
    pub llr_pvalue: f64,
    /// McFadden's pseudo-R², `1 − llf/llnull`.
    pub prsquared: f64,

    /// Akaike information criterion `−2 llf + 2·k_exog·(K−1)`.
    pub aic: f64,
    /// Bayesian information criterion `−2 llf + k_exog·(K−1)·ln n`.
    pub bic: f64,

    /// Predicted choice probabilities, shape `n × K`.
    pub predicted: Array2<f64>,

    /// Coefficient covariance in flat category-major order, shape
    /// `(k_exog·(K−1)) × (k_exog·(K−1))`.
    pub cov_params: Array2<f64>,

    /// Whether Newton converged.
    pub converged: bool,
}

impl MNLogitResults {
    fn new(
        model: &MNLogit,
        params: Array2<f64>,
        cov_params: Array2<f64>,
        converged: bool,
    ) -> MNLogitResults {
        let nobs = model.endog.len() as f64;
        let ke = model.exog.ncols();
        let k = model.k_levels;
        let kc = model.k_constant;

        // Standard errors: sqrt of the diagonal of cov, reshaped category-major.
        let mut bse = Array2::<f64>::zeros((ke, k - 1));
        for kk in 0..k - 1 {
            for a in 0..ke {
                let idx = kk * ke + a;
                bse[[a, kk]] = cov_params[[idx, idx]].sqrt();
            }
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|z| 2.0 * norm_sf(z.abs()));

        let df_model = (ke as f64 - kc as f64) * (k as f64 - 1.0);
        let df_resid = nobs - (df_model + (k as f64 - 1.0));

        let llf = model.loglike_mat(&params);
        let llnull = model.llnull();
        let llr = 2.0 * (llf - llnull);
        let llr_pvalue = chi2_sf(llr, df_model);
        let prsquared = 1.0 - llf / llnull;

        let k_params = (ke * (k - 1)) as f64;
        let aic = -2.0 * llf + 2.0 * k_params;
        let bic = -2.0 * llf + k_params * nobs.ln();

        let predicted = model.probs(&params);

        MNLogitResults {
            params,
            bse,
            tvalues,
            pvalues,
            nobs,
            k_levels: k,
            k_constant: kc,
            df_model,
            df_resid,
            llf,
            llnull,
            llr,
            llr_pvalue,
            prsquared,
            aic,
            bic,
            predicted,
            cov_params,
            converged,
        }
    }

    /// Confidence intervals for each coefficient at level `1 − alpha` (normal).
    ///
    /// Returns a `(k_exog·(K−1)) × 2` matrix in flat category-major order to
    /// pair with [`Self::cov_params`].
    pub fn conf_int(&self, alpha: f64) -> Array2<f64> {
        let q = norm_ppf(1.0 - alpha / 2.0);
        let (ke, km1) = self.params.dim();
        let mut out = Array2::<f64>::zeros((ke * km1, 2));
        for kk in 0..km1 {
            for a in 0..ke {
                let idx = kk * ke + a;
                out[[idx, 0]] = self.params[[a, kk]] - q * self.bse[[a, kk]];
                out[[idx, 1]] = self.params[[a, kk]] + q * self.bse[[a, kk]];
            }
        }
        out
    }

    /// A full results table in the canonical reference MNLogit layout: the
    /// "MNLogit Regression Results" header block followed by one coefficient
    /// table per non-baseline category (each headed `y=k`), with `z`, `P>|z|`,
    /// and the 95 % confidence interval.
    ///
    /// `names`, if given, labels the `k_exog` regressors (the same labels are
    /// reused for every equation); otherwise `x0, x1, …` are used. The output
    /// matches the reference field-for-field (the only volatile fields are the
    /// `Date:`/`Time:` stamps).
    pub fn summary(&self, names: Option<&[&str]>) -> String {
        self.summary_titled("y", names)
    }

    /// As [`summary`](Self::summary), with the dependent-variable label of your
    /// choosing.
    pub fn summary_titled(&self, dep: &str, names: Option<&[&str]>) -> String {
        use crate::summary::{
            centered, coef_header_labeled, coef_row, fmt_g, header_row, utc_now_strings,
        };
        use std::fmt::Write as _;
        const W: usize = 78;
        let bar = "=".repeat(W);
        let dash = "-".repeat(W);
        let (ke, km1) = self.params.dim();
        let ci = self.conf_int(0.05);
        let (date, time) = utc_now_strings();

        let mut s = String::new();
        let _ = writeln!(s, "{}", centered("MNLogit Regression Results", W));
        let _ = writeln!(s, "{bar}");

        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Dep. Variable:",
                dep,
                "No. Observations:",
                &format!("{:.0}", self.nobs)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Model:",
                "MNLogit",
                "Df Residuals:",
                &format!("{:.0}", self.df_resid)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Method:",
                "MLE",
                "Df Model:",
                &format!("{:.0}", self.df_model)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row("Date:", &date, "Pseudo R-squ.:", &fmt_g(self.prsquared, 4))
        );
        let _ = writeln!(
            s,
            "{}",
            header_row("Time:", &time, "Log-Likelihood:", &fmt_g(self.llf, 5))
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "converged:",
                if self.converged { "True" } else { "False" },
                "LL-Null:",
                &fmt_g(self.llnull, 5)
            )
        );
        let _ = writeln!(
            s,
            "{}",
            header_row(
                "Covariance Type:",
                "nonrobust",
                "LLR p-value:",
                &fmt_g(self.llr_pvalue, 4)
            )
        );
        let _ = writeln!(s, "{bar}");

        // One coefficient table per non-baseline category (level 0 is baseline,
        // so equations are labeled y=1 … y=(K-1)).
        for kk in 0..km1 {
            let _ = writeln!(s, "{}", coef_header_labeled(&format!("{}={}", dep, kk + 1)));
            let _ = writeln!(s, "{dash}");
            for a in 0..ke {
                let name = names
                    .and_then(|n| n.get(a).copied())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("x{a}"));
                let idx = kk * ke + a;
                let _ = writeln!(
                    s,
                    "{}",
                    coef_row(
                        &name,
                        self.params[[a, kk]],
                        self.bse[[a, kk]],
                        self.tvalues[[a, kk]],
                        self.pvalues[[a, kk]],
                        ci[[idx, 0]],
                        ci[[idx, 1]]
                    )
                );
            }
            // Inter-equation rows use a dashed separator; the final one a bar.
            if kk + 1 < km1 {
                let _ = writeln!(s, "{dash}");
            } else {
                let _ = write!(s, "{bar}");
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn design() -> (Array1<f64>, Array2<f64>) {
        let xcol = array![0.5, -1.2, 0.3, 2.1, -0.7, 1.4, -0.2, 0.9, -1.5, 0.6, 0.1, -0.4];
        let mut x = Array2::<f64>::ones((12, 2));
        x.column_mut(1).assign(&xcol);
        let y = array![0., 1., 2., 1., 0., 2., 0., 1., 2., 1., 0., 2.];
        (y, x)
    }

    #[test]
    fn score_zero_at_optimum() {
        let (y, x) = design();
        let m = MNLogit::new(y, x).unwrap();
        let res = m.fit().unwrap();
        assert!(res.converged);
        let g = m.score_flat(&res.params);
        assert!(g.dot(&g).sqrt() < 1e-9, "score norm {}", g.dot(&g).sqrt());
    }

    #[test]
    fn probabilities_sum_to_one() {
        let (y, x) = design();
        let res = MNLogit::new(y, x).unwrap().fit().unwrap();
        for i in 0..res.predicted.nrows() {
            let s: f64 = res.predicted.row(i).sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn params_orientation_and_shape() {
        let (y, x) = design();
        let res = MNLogit::new(y, x).unwrap().fit().unwrap();
        assert_eq!(res.params.dim(), (2, 2)); // k_exog=2, K-1=2
        assert_eq!(res.k_levels, 3);
    }

    #[test]
    fn analytic_hessian_matches_finite_difference() {
        let (y, x) = design();
        let m = MNLogit::new(y, x).unwrap();
        let b = array![0.1, -0.2, 0.3, 0.05];
        let bm = m.unflatten(&b);
        let h_an = m.hessian_flat(&bm);
        let eps = 1e-6;
        let dim = b.len();
        for j in 0..dim {
            let mut bp = b.clone();
            let mut bn = b.clone();
            bp[j] += eps;
            bn[j] -= eps;
            let gp = m.score_flat(&m.unflatten(&bp));
            let gn = m.score_flat(&m.unflatten(&bn));
            for i in 0..dim {
                let fd = (gp[i] - gn[i]) / (2.0 * eps);
                assert_abs_diff_eq!(h_an[[i, j]], fd, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn llr_nonnegative_prsquared_unit_interval() {
        let (y, x) = design();
        let res = MNLogit::new(y, x).unwrap().fit().unwrap();
        assert!(res.llr >= -1e-9);
        assert!(res.prsquared >= 0.0 && res.prsquared <= 1.0);
    }

    /// The summary must carry the discrete header block plus one coefficient
    /// table per non-baseline category (each headed `y=k`).
    #[test]
    fn summary_has_header_and_per_category_tables() {
        let (y, x) = design(); // 3 categories → equations y=1, y=2
        let res = MNLogit::new(y, x).unwrap().fit().unwrap();
        let s = res.summary(Some(&["const", "x1"]));
        assert!(s.contains("MNLogit Regression Results"));
        assert!(s.contains("Method:") && s.contains("MLE"));
        assert!(s.contains("No. Observations:"));
        assert!(s.contains("Df Residuals:") && s.contains("Df Model:"));
        assert!(s.contains("Pseudo R-squ.:") && s.contains("LL-Null:"));
        assert!(s.contains("LLR p-value:"));
        // One equation header per non-baseline level.
        assert!(s.contains("y=1") && s.contains("y=2"));
        // Each equation repeats the regressor labels and the z-inference columns.
        assert!(s.contains("P>|z|") && !s.contains("P>|t|"));
        assert!(s.contains("[0.025") && s.contains("0.975]"));
        assert_eq!(s.matches("const").count(), 2);
        assert_eq!(s.matches("x1").count(), 2);
    }
}
