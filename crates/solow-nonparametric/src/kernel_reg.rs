//! Multivariate kernel regression and multivariate kernel density estimation
//! with a product Gaussian kernel and a fixed, user-supplied bandwidth.
//!
//! Two estimators are provided, both mirroring the canonical Python reference
//! (`nonparametric.kernel_regression.KernelReg` and
//! `nonparametric.kernel_density.KDEMultivariate`) for the all-continuous
//! (`var_type = "c…c"`) case with an explicit bandwidth:
//!
//! * [`KernelReg`] — Nadaraya–Watson kernel regression of a scalar response on
//!   `d` continuous predictors, with a local-constant ([`RegType::LocalConstant`])
//!   or local-linear ([`RegType::LocalLinear`]) fit.
//! * [`KdeMultivariate`] — a product-Gaussian multivariate kernel density
//!   estimator for `d` continuous variables.
//!
//! # Kernel
//!
//! The continuous kernel is the standard Gaussian
//! `k(z) = exp(−z²/2) / √(2π)`. The generalized product kernel for an
//! observation `Xᵢ ∈ ℝ^d` evaluated at a point `x` with bandwidth vector
//! `h ∈ ℝ^d` is
//!
//! ```text
//! K_h(Xᵢ, x) = (1 / ∏_s h_s) · ∏_{s=1}^{d} k((X_{is} − x_s) / h_s),
//! ```
//!
//! exactly the (non-normalized) generalized product kernel estimator used by
//! the reference's `gpke`. The leading `1/∏ h_s` factor cancels in the
//! Nadaraya–Watson ratio and in the local-linear normal equations, so the
//! regression mean is insensitive to it; it matters for the density.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::pinv;
use std::f64::consts::PI;

/// Local polynomial order for [`KernelReg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegType {
    /// Local-constant (Nadaraya–Watson) fit, the reference's `reg_type='lc'`.
    LocalConstant,
    /// Local-linear fit, the reference's `reg_type='ll'`.
    LocalLinear,
}

/// One-dimensional Gaussian kernel `k(z) = exp(−z²/2)/√(2π)`.
#[inline]
fn gaussian(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Per-observation generalized product kernel values `K_h(Xᵢ, x)` for every
/// training row `Xᵢ` against a single evaluation point `x`.
///
/// Returns a length-`n` vector whose `i`-th entry is
/// `(1/∏ h_s) · ∏_s k((X_{is} − x_s)/h_s)`, matching `gpke(..., tosum=False)`
/// for an all-continuous Gaussian kernel.
fn gpke_rows(bw: &[f64], data: &Array2<f64>, point: &[f64]) -> Array1<f64> {
    let (n, d) = data.dim();
    let mut prod_h = 1.0;
    for &h in bw.iter() {
        prod_h *= h;
    }
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut prod = 1.0;
        for s in 0..d {
            prod *= gaussian((data[[i, s]] - point[s]) / bw[s]);
        }
        out[i] = prod / prod_h;
    }
    out
}

/// Validate a feature matrix / bandwidth pair shared by both estimators.
fn validate(data: &Array2<f64>, bw: &[f64]) -> Result<()> {
    let (n, d) = data.dim();
    if n == 0 {
        return Err(Error::Value(
            "kernel estimator requires at least one observation".into(),
        ));
    }
    if d == 0 {
        return Err(Error::Value(
            "kernel estimator requires at least one variable".into(),
        ));
    }
    if bw.len() != d {
        return Err(Error::Shape(format!(
            "bandwidth length {} does not match number of variables {d}",
            bw.len()
        )));
    }
    for &h in bw.iter() {
        if !h.is_finite() || h <= 0.0 {
            return Err(Error::Value(
                "every bandwidth must be positive and finite".into(),
            ));
        }
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error::Value("kernel estimator data must be finite".into()));
    }
    Ok(())
}

/// Multivariate Nadaraya–Watson kernel regression with a product Gaussian
/// kernel and a fixed bandwidth.
///
/// The model is `y = g(x) + ε` with `x ∈ ℝ^d`. [`KernelReg::fit`] evaluates the
/// estimated conditional mean `ĝ` on a grid of query points.
///
/// # Example
/// ```
/// use ndarray::{array, Array2};
/// use solow_nonparametric::{KernelReg, RegType};
///
/// let exog = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let endog = array![0.0, 1.0, 4.0, 9.0, 16.0];
/// let kr = KernelReg::new(endog, exog, vec![0.8], RegType::LocalLinear).unwrap();
/// let grid = Array2::from_shape_vec((2, 1), vec![1.5, 2.5]).unwrap();
/// let mean = kr.fit(&grid).unwrap();
/// assert_eq!(mean.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KernelReg {
    endog: Array1<f64>,
    exog: Array2<f64>,
    bw: Vec<f64>,
    reg_type: RegType,
}

impl KernelReg {
    /// Construct a kernel regression for response `endog` on predictors `exog`
    /// (`n × d`) with bandwidth vector `bw` (length `d`) and the given fit order.
    ///
    /// # Errors
    /// Returns an error on empty data, an `endog`/`exog` row-count mismatch, a
    /// bandwidth-length mismatch, a non-positive/non-finite bandwidth, or
    /// non-finite inputs.
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        bw: Vec<f64>,
        reg_type: RegType,
    ) -> Result<Self> {
        validate(&exog, &bw)?;
        if endog.len() != exog.dim().0 {
            return Err(Error::Shape(format!(
                "endog length {} does not match exog rows {}",
                endog.len(),
                exog.dim().0
            )));
        }
        if endog.iter().any(|v| !v.is_finite()) {
            return Err(Error::Value("endog must be finite".into()));
        }
        Ok(KernelReg {
            endog,
            exog,
            bw,
            reg_type,
        })
    }

    /// Number of predictor variables `d`.
    pub fn k_vars(&self) -> usize {
        self.exog.dim().1
    }

    /// The bandwidth vector in use.
    pub fn bw(&self) -> &[f64] {
        &self.bw
    }

    /// Evaluate the estimated conditional mean at each row of `data_predict`
    /// (`m × d`).
    ///
    /// Returns the length-`m` vector of fitted means, matching the first return
    /// value of the reference's `KernelReg.fit`.
    ///
    /// # Errors
    /// Returns an error if `data_predict` does not have `d` columns.
    pub fn fit(&self, data_predict: &Array2<f64>) -> Result<Array1<f64>> {
        let (m, d) = data_predict.dim();
        if d != self.k_vars() {
            return Err(Error::Shape(format!(
                "data_predict has {d} columns, expected {}",
                self.k_vars()
            )));
        }
        let mut mean = Array1::<f64>::zeros(m);
        let mut row = vec![0.0; d];
        for i in 0..m {
            for s in 0..d {
                row[s] = data_predict[[i, s]];
            }
            mean[i] = match self.reg_type {
                RegType::LocalConstant => self.est_loc_constant(&row),
                RegType::LocalLinear => self.est_loc_linear(&row)?,
            };
        }
        Ok(mean)
    }

    /// Local-constant (Nadaraya–Watson) estimate at one point.
    ///
    /// `ĝ(x) = Σᵢ K_h(Xᵢ,x) yᵢ / Σᵢ K_h(Xᵢ,x)`. The `1/∏ h_s` factor cancels.
    fn est_loc_constant(&self, point: &[f64]) -> f64 {
        let ker = gpke_rows(&self.bw, &self.exog, point);
        let mut numer = 0.0;
        let mut denom = 0.0;
        for (k, &w) in ker.iter().enumerate() {
            numer += w * self.endog[k];
            denom += w;
        }
        numer / denom
    }

    /// Local-linear estimate at one point.
    ///
    /// Builds the weighted least-squares normal-equation system on p. 81 of
    /// Li & Racine and solves it with the Moore–Penrose pseudoinverse, exactly
    /// as the reference's `_est_loc_linear` (which divides the kernel weights by
    /// `nobs`; that scale cancels and is omitted here). Returns the intercept
    /// component, i.e. the fitted mean at `point`.
    fn est_loc_linear(&self, point: &[f64]) -> Result<f64> {
        let (n, d) = self.exog.dim();
        let ker = gpke_rows(&self.bw, &self.exog, point);

        // M is (d+1)×(d+1); V is (d+1).
        let dim = d + 1;
        let mut m = Array2::<f64>::zeros((dim, dim));
        let mut v = Array1::<f64>::zeros(dim);

        // Centered predictors Δᵢ = Xᵢ − x.
        // M[0,0] = Σ kᵢ
        // M[0,1+s] = M[1+s,0] = Σ kᵢ Δ_{is}
        // M[1+s,1+t] = Σ kᵢ Δ_{is} Δ_{it}
        // V[0] = Σ kᵢ yᵢ
        // V[1+s] = Σ kᵢ Δ_{is} yᵢ
        let mut sum_k = 0.0;
        let mut m12 = vec![0.0; d];
        let mut m22 = vec![0.0; d * d];
        let mut v0 = 0.0;
        let mut v_rest = vec![0.0; d];
        for i in 0..n {
            let ki = ker[i];
            sum_k += ki;
            let yi = self.endog[i];
            v0 += ki * yi;
            for s in 0..d {
                let ds = self.exog[[i, s]] - point[s];
                let kds = ki * ds;
                m12[s] += kds;
                v_rest[s] += kds * yi;
                for t in 0..d {
                    let dt = self.exog[[i, t]] - point[t];
                    m22[s * d + t] += kds * dt;
                }
            }
        }
        m[[0, 0]] = sum_k;
        v[0] = v0;
        for s in 0..d {
            m[[0, s + 1]] = m12[s];
            m[[s + 1, 0]] = m12[s];
            v[s + 1] = v_rest[s];
            for t in 0..d {
                m[[s + 1, t + 1]] = m22[s * d + t];
            }
        }

        let (mp, _s) = pinv(&m)?;
        let coef = mp.dot(&v);
        Ok(coef[0])
    }
}

/// Product-Gaussian multivariate kernel density estimator with a fixed
/// bandwidth, mirroring the reference's `KDEMultivariate` for all-continuous
/// variables.
///
/// The density at a point `x ∈ ℝ^d` is
///
/// ```text
/// f̂(x) = (1/n) · Σᵢ K_h(Xᵢ, x)
///      = 1 / (n · ∏_s h_s) · Σᵢ ∏_s k((X_{is} − x_s)/h_s).
/// ```
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use solow_nonparametric::KdeMultivariate;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.5, -0.5, 1.0, 0.3, -0.2,
/// ]).unwrap();
/// let kde = KdeMultivariate::new(data, vec![0.5, 0.5]).unwrap();
/// let pts = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
/// let dens = kde.pdf(&pts).unwrap();
/// assert!(dens[0] > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct KdeMultivariate {
    data: Array2<f64>,
    bw: Vec<f64>,
}

impl KdeMultivariate {
    /// Construct an estimator over `data` (`n × d`) with bandwidth vector `bw`
    /// of length `d`.
    ///
    /// # Errors
    /// Returns an error on empty data, a bandwidth-length mismatch, a
    /// non-positive/non-finite bandwidth, or non-finite data.
    pub fn new(data: Array2<f64>, bw: Vec<f64>) -> Result<Self> {
        validate(&data, &bw)?;
        Ok(KdeMultivariate { data, bw })
    }

    /// Number of variables `d`.
    pub fn k_vars(&self) -> usize {
        self.data.dim().1
    }

    /// The bandwidth vector in use.
    pub fn bw(&self) -> &[f64] {
        &self.bw
    }

    /// Evaluate the density at each row of `data_predict` (`m × d`).
    ///
    /// # Errors
    /// Returns an error if `data_predict` does not have `d` columns.
    pub fn pdf(&self, data_predict: &Array2<f64>) -> Result<Array1<f64>> {
        let (m, d) = data_predict.dim();
        if d != self.k_vars() {
            return Err(Error::Shape(format!(
                "data_predict has {d} columns, expected {}",
                self.k_vars()
            )));
        }
        let nobs = self.data.dim().0 as f64;
        let mut out = Array1::<f64>::zeros(m);
        let mut row = vec![0.0; d];
        for i in 0..m {
            for s in 0..d {
                row[s] = data_predict[[i, s]];
            }
            let ker = gpke_rows(&self.bw, &self.data, &row);
            out[i] = ker.sum() / nobs;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn col(v: Vec<f64>) -> Array2<f64> {
        let n = v.len();
        Array2::from_shape_vec((n, 1), v).unwrap()
    }

    #[test]
    fn loc_constant_matches_manual_nw() {
        let exog = col(vec![0.0, 1.0, 2.0, 3.0]);
        let endog = array![1.0, 2.0, 1.0, 3.0];
        let h = 0.7;
        let kr =
            KernelReg::new(endog.clone(), exog.clone(), vec![h], RegType::LocalConstant).unwrap();
        let grid = col(vec![1.5]);
        let got = kr.fit(&grid).unwrap()[0];
        // Manual Nadaraya-Watson.
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..4 {
            let w = gaussian((exog[[i, 0]] - 1.5) / h);
            num += w * endog[i];
            den += w;
        }
        assert!((got - num / den).abs() < 1e-12);
    }

    #[test]
    fn loc_linear_reproduces_exact_line() {
        // On exactly-linear data the local-linear estimator is exact.
        let exog = col(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let endog: Array1<f64> = exog.column(0).mapv(|x| 2.0 * x - 1.0);
        let kr = KernelReg::new(endog, exog, vec![0.6], RegType::LocalLinear).unwrap();
        let grid = col(vec![1.3, 2.7, 4.1]);
        let got = kr.fit(&grid).unwrap();
        for (k, &t) in [1.3, 2.7, 4.1].iter().enumerate() {
            assert!(
                (got[k] - (2.0 * t - 1.0)).abs() < 1e-8,
                "k={k} got={}",
                got[k]
            );
        }
    }

    #[test]
    fn pdf_integrates_to_one_2d() {
        // Coarse trapezoid over a grid should integrate near 1.
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.5, -0.5, 1.0, 0.3, -0.2, -0.8, -0.6],
        )
        .unwrap();
        let kde = KdeMultivariate::new(data, vec![0.6, 0.6]).unwrap();
        let lo = -4.0;
        let hi = 4.0;
        let ng = 80;
        let step = (hi - lo) / (ng as f64 - 1.0);
        let axis: Vec<f64> = (0..ng).map(|i| lo + step * i as f64).collect();
        let mut grid_pts = Vec::with_capacity(ng * ng * 2);
        for &a in &axis {
            for &b in &axis {
                grid_pts.push(a);
                grid_pts.push(b);
            }
        }
        let pts = Array2::from_shape_vec((ng * ng, 2), grid_pts).unwrap();
        let dens = kde.pdf(&pts).unwrap();
        let area: f64 = dens.sum() * step * step;
        assert!((area - 1.0).abs() < 1e-2, "area = {area}");
    }

    #[test]
    fn bad_bandwidth_errors() {
        let exog = col(vec![0.0, 1.0, 2.0]);
        let endog = array![0.0, 1.0, 2.0];
        assert!(
            KernelReg::new(endog.clone(), exog.clone(), vec![0.0], RegType::LocalLinear).is_err()
        );
        assert!(
            KernelReg::new(endog, exog, vec![0.5, 0.5], RegType::LocalLinear).is_err(),
            "bandwidth length mismatch should error"
        );
    }
}
