//! Quantile regression by iteratively reweighted least squares (IRLS).
//!
//! The conditional-quantile estimator minimizes the asymmetric *check loss*
//! `ρ_q(u) = u·(q − 1{u<0})`. It is solved here by the IRLS scheme of the
//! canonical reference: each iteration reweights the design rows by the inverse
//! of the (check-weighted) absolute residual and re-solves the normal equations.
//!
//! The reported standard errors use the reference's default *robust*
//! sparsity/kernel sandwich,
//!
//! ```text
//!   V = (XᵀX)⁻¹ · (Xᵀ D X) · (XᵀX)⁻¹,
//!   D_ii = (q / f̂₀)²   if eᵢ > 0,   ((1−q) / f̂₀)²   otherwise,
//! ```
//!
//! where `f̂₀` is a kernel density estimate of the residual density at zero,
//! computed with the Epanechnikov kernel and the Hall–Sheather bandwidth — the
//! reference defaults (`kernel='epa'`, `bandwidth='hsheather'`, `vcov='robust'`).

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::{norm_pdf, norm_ppf};
use solow_linalg::pinv;

/// A quantile-regression model awaiting a `fit(q)`.
#[derive(Clone, Debug)]
pub struct QuantReg {
    endog: Array1<f64>,
    exog: Array2<f64>,
    k_constant: usize,
}

impl QuantReg {
    /// Build a quantile-regression model from response `endog` and design `exog`.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if exog.nrows() == 0 {
            return Err(Error::Value("empty design matrix".into()));
        }
        let k_constant = detect_k_constant(&exog);
        Ok(QuantReg {
            endog,
            exog,
            k_constant,
        })
    }

    /// Fit the `q`-th conditional quantile (`0 < q < 1`) with the reference
    /// defaults: convergence tolerance `1e-9` on the max coefficient change and
    /// up to `5000` IRLS iterations.
    pub fn fit(&self, q: f64) -> Result<QuantRegResults> {
        self.fit_with(q, 1e-9, 5000)
    }

    /// Fit the `q`-th conditional quantile with explicit IRLS controls.
    pub fn fit_with(&self, q: f64, p_tol: f64, max_iter: usize) -> Result<QuantRegResults> {
        if q <= 0.0 || q >= 1.0 {
            return Err(Error::Value("q must be strictly between 0 and 1".into()));
        }
        let (n, p) = self.exog.dim();
        let nobs = n as f64;

        // IRLS: start from beta = 1 (used only for the convergence check; the
        // first solve is the unweighted least-squares fit, exactly as in the
        // reference).
        let mut beta = Array1::<f64>::ones(p);
        // xstar starts equal to exog (unweighted), so iteration 1 solves OLS.
        let mut xstar = self.exog.clone();
        let mut diff = 10.0_f64;
        let mut n_iter = 0usize;

        while n_iter < max_iter && diff > p_tol {
            n_iter += 1;
            let beta0 = beta.clone();

            // beta = pinv(xstarᵀ X) · (xstarᵀ y)
            let xtx = xstar.t().dot(&self.exog);
            let xty = xstar.t().dot(&self.endog);
            let (xtx_pinv, _) = pinv(&xtx)?;
            beta = xtx_pinv.dot(&xty);

            // resid = y − X beta, floored away from zero, then check-weighted.
            let fitted = self.exog.dot(&beta);
            let mut resid = &self.endog - &fitted;
            for r in resid.iter_mut() {
                if r.abs() < 1e-6 {
                    *r = if *r >= 0.0 { 1e-6 } else { -1e-6 };
                }
                // check-function weight, then absolute value
                let w = if *r < 0.0 { q * *r } else { (1.0 - q) * *r };
                *r = w.abs();
            }

            // xstar = exog / resid (row-wise).
            let mut new_xstar = self.exog.clone();
            for i in 0..n {
                let inv = 1.0 / resid[i];
                for j in 0..p {
                    new_xstar[[i, j]] *= inv;
                }
            }
            xstar = new_xstar;

            diff = (&beta - &beta0)
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()));
        }

        // Residuals at the solution.
        let e = &self.endog - &self.exog.dot(&beta);

        // Bandwidth (Hall–Sheather, alpha = 0.05) and sparsity scale.
        let hs = hall_sheather(nobs, q, 0.05);
        let iqre = percentile(&e, 75.0) - percentile(&e, 25.0);
        let std_endog = std(&self.endog);
        let h = std_endog.min(iqre / 1.34) * (norm_ppf(q + hs) - norm_ppf(q - hs));

        // f̂₀ = (1 / (n h)) Σ K(eᵢ / h), Epanechnikov kernel.
        let mut fhat0 = 0.0;
        for &ei in e.iter() {
            fhat0 += epanechnikov(ei / h);
        }
        fhat0 /= nobs * h;
        let sparsity = 1.0 / fhat0;

        // Robust sandwich vcov.
        let xtx = self.exog.t().dot(&self.exog);
        let (xtxi, _) = pinv(&xtx)?;
        // d_i and Xᵀ D X.
        let mut xtdx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let d = if e[i] > 0.0 {
                (q / fhat0).powi(2)
            } else {
                ((1.0 - q) / fhat0).powi(2)
            };
            let row = self.exog.row(i);
            for a in 0..p {
                let ra = row[a] * d;
                for b in 0..p {
                    xtdx[[a, b]] += ra * row[b];
                }
            }
        }
        let vcov = xtxi.dot(&xtdx).dot(&xtxi);

        let mut bse = Array1::<f64>::zeros(p);
        for j in 0..p {
            bse[j] = vcov[[j, j]].sqrt();
        }

        let rank = matrix_rank(&self.exog);
        let k_constant = self.k_constant;
        Ok(QuantRegResults {
            params: beta,
            bse,
            vcov,
            q,
            sparsity,
            bandwidth: h,
            iterations: n_iter,
            resid: e,
            nobs,
            rank,
            k_constant,
            df_model: rank as f64 - k_constant as f64,
            df_resid: nobs - rank as f64,
        })
    }

    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.endog.len()
    }
}

/// The fitted result of a [`QuantReg`].
#[derive(Clone, Debug)]
pub struct QuantRegResults {
    /// Estimated coefficients for the `q`-th conditional quantile.
    pub params: Array1<f64>,
    /// Robust (sparsity/kernel sandwich) standard errors.
    pub bse: Array1<f64>,
    /// Full robust coefficient covariance matrix.
    pub vcov: Array2<f64>,
    /// The fitted quantile level.
    pub q: f64,
    /// Estimated sparsity `1 / f̂₀`.
    pub sparsity: f64,
    /// Kernel-density bandwidth `h`.
    pub bandwidth: f64,
    /// Number of IRLS iterations performed.
    pub iterations: usize,
    /// Residuals `y − X·params`.
    pub resid: Array1<f64>,
    /// Number of observations.
    pub nobs: f64,
    /// Rank of the design.
    pub rank: usize,
    /// Whether a constant is present (0/1).
    pub k_constant: usize,
    /// Model degrees of freedom (`rank − k_constant`).
    pub df_model: f64,
    /// Residual degrees of freedom (`nobs − rank`).
    pub df_resid: f64,
}

/// Epanechnikov kernel `¾(1 − u²)·1{|u| ≤ 1}` (reference `kernels['epa']`).
fn epanechnikov(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Hall–Sheather (1988) plug-in bandwidth rule (reference `hall_sheather`).
fn hall_sheather(n: f64, q: f64, alpha: f64) -> f64 {
    let z = norm_ppf(q);
    let num = 1.5 * norm_pdf(z).powi(2);
    let den = 2.0 * z * z + 1.0;
    n.powf(-1.0 / 3.0) * norm_ppf(1.0 - alpha / 2.0).powf(2.0 / 3.0) * (num / den).powf(1.0 / 3.0)
}

/// Sample standard deviation (population, `ddof = 0`), matching `np.std`.
fn std(v: &Array1<f64>) -> f64 {
    let n = v.len() as f64;
    let mean = v.sum() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// `scipy.stats.scoreatpercentile` with the default linear ("fraction")
/// interpolation: sort, then linearly interpolate at index `per/100·(n−1)`.
fn percentile(v: &Array1<f64>, per: f64) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let idx = per / 100.0 * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let frac = idx - lo as f64;
    if lo + 1 >= n {
        s[n - 1]
    } else {
        s[lo] + (s[lo + 1] - s[lo]) * frac
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

/// Numerical rank of `x` via singular values of its pseudoinverse-producing
/// decomposition, with the reference's `tol = max(s)·max(m,n)·eps`.
fn matrix_rank(x: &Array2<f64>) -> usize {
    solow_linalg::matrix_rank(x).unwrap_or_else(|_| x.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn median_regression_is_robust_to_an_outlier() {
        // y = 1 + 2x with one gross outlier; the q=0.5 fit must ignore it.
        // (The data are otherwise an exact fit, so — exactly as in the reference —
        // the residual-density bandwidth degenerates and bse is NaN; the point
        // estimate is what matters here.)
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        let mut y = array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        y[2] = 100.0; // outlier
        let res = QuantReg::new(y, x).unwrap().fit(0.5).unwrap();
        assert!(
            (res.params[0] - 1.0).abs() < 1e-6,
            "intercept {}",
            res.params[0]
        );
        assert!(
            (res.params[1] - 2.0).abs() < 1e-6,
            "slope {}",
            res.params[1]
        );
    }

    #[test]
    fn median_regression_bse_well_defined_on_noisy_data() {
        // With genuine noise the sandwich SE is finite and positive.
        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
            [1.0, 2.5]
        ];
        let y = array![0.9, 1.2, 2.1, 2.0, 3.3, 3.1, 4.4, 4.0];
        let res = QuantReg::new(y, x).unwrap().fit(0.5).unwrap();
        assert!(
            res.bse.iter().all(|&b| b.is_finite() && b >= 0.0),
            "bse {:?}",
            res.bse
        );
        assert!(res.sparsity.is_finite() && res.sparsity > 0.0);
    }

    #[test]
    fn percentile_matches_linear_interpolation() {
        let v = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!((percentile(&v, 75.0) - 5.5).abs() < 1e-12);
        assert!((percentile(&v, 25.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&v, 50.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_out_of_range_quantile() {
        let x = array![[1.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0];
        let m = QuantReg::new(y, x).unwrap();
        assert!(m.fit(0.0).is_err());
        assert!(m.fit(1.0).is_err());
    }
}
