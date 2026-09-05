//! Robust ("sandwich") covariance estimators for ordinary-least-squares fits.
//!
//! Given the design matrix `exog` (`n × k`) and the OLS residuals `u` (`n`)
//! these functions return the `k × k` robust covariance of the coefficient
//! estimates. The heteroskedasticity-consistent family (`cov_hc0` … `cov_hc3`)
//! differs only in the per-observation scaling applied to `u²`; the
//! Newey–West HAC estimator ([`cov_hac`]) adds Bartlett-weighted autocovariance
//! terms; and the one-way clustered estimator ([`cov_cluster`]) sums the score
//! contributions within clusters. [`robust_bse`] returns the square root of the
//! diagonal of any of these.
//!
//! The implementations mirror the reference
//! `…stats.sandwich_covariance` (`cov_hc0`, `cov_hc1`, `cov_hc2`, `cov_hc3`,
//! `cov_hac_simple`, `cov_cluster`), which are equivalent to the covariance
//! matrices produced by `OLS(...).get_robustcov_results(cov_type=...)`.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::inv;

/// `(X'X)⁻¹` ("normalized cov params" in the reference); the unscaled hessian
/// inverse for an OLS fit.
fn xtx_inv(exog: &Array2<f64>) -> Result<Array2<f64>> {
    let xtx = exog.t().dot(exog);
    inv(&xtx).map_err(|_| Error::Value("design matrix is singular".into()))
}

/// `pinv(X) = (X'X)⁻¹ X'`, the Moore–Penrose pseudo-inverse of a full-column-rank
/// design (`k × n`).
fn pinv(exog: &Array2<f64>, xtxi: &Array2<f64>) -> Array2<f64> {
    xtxi.dot(&exog.t())
}

/// Leverage (hat-matrix diagonal) `hᵢ = xᵢ (X'X)⁻¹ xᵢ'`.
///
/// Exposed because HC2/HC3 need it; callers may also reuse it directly.
pub fn hat_diag(exog: &Array2<f64>) -> Result<Array1<f64>> {
    let xtxi = xtx_inv(exog)?;
    Ok(hat_diag_with(exog, &xtxi))
}

fn hat_diag_with(exog: &Array2<f64>, xtxi: &Array2<f64>) -> Array1<f64> {
    let n = exog.nrows();
    let mut h = Array1::<f64>::zeros(n);
    // hᵢ = xᵢ · (X'X)⁻¹ · xᵢ'
    let tmp = exog.dot(xtxi); // n × k
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..exog.ncols() {
            s += tmp[[i, j]] * exog[[i, j]];
        }
        h[i] = s;
    }
    h
}

/// `pinv(X) · diag(scale) · pinv(X)'` — the HCCM sandwich for a per-observation
/// scale vector. This is the reference `_HCCM`.
fn hccm(pinv_x: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let k = pinv_x.nrows();
    let n = pinv_x.ncols();
    // out = P · diag(scale) · P'   where P is k × n.
    let mut out = Array2::<f64>::zeros((k, k));
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for t in 0..n {
                s += pinv_x[[a, t]] * scale[t] * pinv_x[[b, t]];
            }
            out[[a, b]] = s;
        }
    }
    out
}

fn validate(exog: &Array2<f64>, resid: &Array1<f64>) -> Result<()> {
    if exog.nrows() != resid.len() {
        return Err(Error::Shape("exog rows must equal residual length".into()));
    }
    if exog.nrows() <= exog.ncols() {
        return Err(Error::Value(
            "need more observations than parameters".into(),
        ));
    }
    Ok(())
}

/// White's HC0 heteroskedasticity-consistent covariance:
/// `(X'X)⁻¹ X' diag(uᵢ²) X (X'X)⁻¹`.
pub fn cov_hc0(exog: &Array2<f64>, resid: &Array1<f64>) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let xtxi = xtx_inv(exog)?;
    let p = pinv(exog, &xtxi);
    let scale = resid.mapv(|u| u * u);
    Ok(hccm(&p, &scale))
}

/// HC1: HC0 scaled by `n / (n − k)` (a degrees-of-freedom correction).
pub fn cov_hc1(exog: &Array2<f64>, resid: &Array1<f64>) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let n = exog.nrows() as f64;
    let k = exog.ncols() as f64;
    let xtxi = xtx_inv(exog)?;
    let p = pinv(exog, &xtxi);
    let factor = n / (n - k);
    let scale = resid.mapv(|u| factor * u * u);
    Ok(hccm(&p, &scale))
}

/// HC2: per-observation scale `uᵢ² / (1 − hᵢ)` with leverages `hᵢ`.
pub fn cov_hc2(exog: &Array2<f64>, resid: &Array1<f64>) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let xtxi = xtx_inv(exog)?;
    let p = pinv(exog, &xtxi);
    let h = hat_diag_with(exog, &xtxi);
    let mut scale = Array1::<f64>::zeros(resid.len());
    for i in 0..resid.len() {
        scale[i] = resid[i] * resid[i] / (1.0 - h[i]);
    }
    Ok(hccm(&p, &scale))
}

/// HC3: per-observation scale `(uᵢ / (1 − hᵢ))²` with leverages `hᵢ`.
pub fn cov_hc3(exog: &Array2<f64>, resid: &Array1<f64>) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let xtxi = xtx_inv(exog)?;
    let p = pinv(exog, &xtxi);
    let h = hat_diag_with(exog, &xtxi);
    let mut scale = Array1::<f64>::zeros(resid.len());
    for i in 0..resid.len() {
        let r = resid[i] / (1.0 - h[i]);
        scale[i] = r * r;
    }
    Ok(hccm(&p, &scale))
}

/// Bartlett kernel weights `1 − l / (maxlags + 1)` for lags `l = 0 … maxlags`.
fn weights_bartlett(maxlags: usize) -> Vec<f64> {
    (0..=maxlags)
        .map(|l| 1.0 - l as f64 / (maxlags as f64 + 1.0))
        .collect()
}

/// Newey–West heteroskedasticity- and autocorrelation-consistent (HAC)
/// covariance with Bartlett weights and a window of `maxlags` lags.
///
/// Forms the score array `xuᵢ = xᵢ · uᵢ`, builds the weighted long-run inner
/// matrix `S = Γ₀ + Σ_{l=1}^{maxlags} w_l (Γ_l + Γ_l')` (with `Γ_l = Σ_t
/// xu_t xu_{t−l}'`), then sandwiches it as `(X'X)⁻¹ S (X'X)⁻¹`. When
/// `use_correction` is `true` the result is scaled by `n / (n − k)` (the
/// reference default). The observations are assumed to be ordered in time.
/// Mirrors the reference `cov_hac_simple`.
pub fn cov_hac(
    exog: &Array2<f64>,
    resid: &Array1<f64>,
    maxlags: usize,
    use_correction: bool,
) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let n = exog.nrows();
    let k = exog.ncols();
    if maxlags >= n {
        return Err(Error::Value("maxlags too large for sample".into()));
    }
    // Score array xu (n × k): each row is xᵢ scaled by the residual uᵢ.
    let mut xu = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            xu[[i, j]] = exog[[i, j]] * resid[i];
        }
    }
    let weights = weights_bartlett(maxlags);

    // S = w0 · (xu' xu) + Σ_{lag≥1} w_lag · (Γ_lag + Γ_lag')
    let mut s = xu.t().dot(&xu); // Γ₀, w0 == 1
    for (lag, &w) in weights.iter().enumerate().skip(1) {
        // Γ_lag = Σ_t xu[t] · xu[t−lag]'  =  xu[lag..]' · xu[..n−lag]
        let upper = xu.slice(ndarray::s![lag.., ..]);
        let lower = xu.slice(ndarray::s![..n - lag, ..]);
        let g = upper.t().dot(&lower); // k × k
        let gt = g.t().to_owned();
        s = s + (&g + &gt) * w;
    }

    let xtxi = xtx_inv(exog)?;
    let mut cov = xtxi.dot(&s).dot(&xtxi);
    if use_correction {
        let factor = n as f64 / (n as f64 - k as f64);
        cov.mapv_inplace(|v| v * factor);
    }
    Ok(cov)
}

/// One-way cluster-robust covariance.
///
/// `groups` assigns each observation to a cluster (any integer labels). The
/// inner matrix sums the cluster-aggregated scores
/// `S = Σ_g (Σ_{i∈g} xuᵢ)(Σ_{i∈g} xuᵢ)'`, then sandwiches it as
/// `(X'X)⁻¹ S (X'X)⁻¹`. With `use_correction = true` (the reference default) the
/// small-sample factor `G/(G−1) · (n−1)/(n−k)` is applied, where `G` is the
/// number of clusters. Mirrors the reference `cov_cluster`.
pub fn cov_cluster(
    exog: &Array2<f64>,
    resid: &Array1<f64>,
    groups: &[i64],
    use_correction: bool,
) -> Result<Array2<f64>> {
    validate(exog, resid)?;
    let n = exog.nrows();
    let k = exog.ncols();
    if groups.len() != n {
        return Err(Error::Shape("groups length must equal sample size".into()));
    }
    // Distinct cluster labels (sorted for determinism).
    let mut labels: Vec<i64> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let n_groups = labels.len();
    if n_groups < 2 {
        return Err(Error::Value("need at least two clusters".into()));
    }
    let index = |g: i64| labels.binary_search(&g).unwrap();

    // Per-cluster sum of scores xuᵢ = xᵢ uᵢ.
    let mut group_sums = Array2::<f64>::zeros((n_groups, k));
    for i in 0..n {
        let gi = index(groups[i]);
        for j in 0..k {
            group_sums[[gi, j]] += exog[[i, j]] * resid[i];
        }
    }
    // S = Σ_g s_g s_g'  = group_sums' · group_sums.
    let s = group_sums.t().dot(&group_sums);

    let xtxi = xtx_inv(exog)?;
    let mut cov = xtxi.dot(&s).dot(&xtxi);
    if use_correction {
        let g = n_groups as f64;
        let nn = n as f64;
        let kk = k as f64;
        let factor = g / (g - 1.0) * ((nn - 1.0) / (nn - kk));
        cov.mapv_inplace(|v| v * factor);
    }
    Ok(cov)
}

/// Robust standard errors: `sqrt(diag(cov))` of any covariance matrix.
pub fn robust_bse(cov: &Array2<f64>) -> Array1<f64> {
    let k = cov.nrows();
    let mut bse = Array1::<f64>::zeros(k);
    for i in 0..k {
        bse[i] = cov[[i, i]].sqrt();
    }
    bse
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use solow_regression::LinearModel;

    fn small() -> (Array1<f64>, Array2<f64>) {
        let x = array![
            [1.0, 0.2, -0.5],
            [1.0, -0.1, 0.3],
            [1.0, 0.4, 0.1],
            [1.0, -0.3, -0.2],
            [1.0, 0.5, 0.6],
            [1.0, -0.2, -0.4],
            [1.0, 0.1, 0.2],
            [1.0, 0.3, -0.1],
            [1.0, -0.4, 0.5],
            [1.0, 0.0, -0.3],
            [1.0, 0.25, 0.15],
            [1.0, -0.35, 0.05],
        ];
        let y = array![0.9, 1.1, 1.4, 0.7, 1.8, 0.6, 1.2, 1.0, 0.8, 1.05, 1.3, 0.95];
        (y, x)
    }

    #[test]
    fn hc_family_symmetric_and_psd_diag() {
        let (y, x) = small();
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();
        for cov in [
            cov_hc0(&x, &res.resid).unwrap(),
            cov_hc1(&x, &res.resid).unwrap(),
            cov_hc2(&x, &res.resid).unwrap(),
            cov_hc3(&x, &res.resid).unwrap(),
        ] {
            for i in 0..cov.nrows() {
                assert!(cov[[i, i]] > 0.0);
                for j in 0..cov.ncols() {
                    assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn hc1_is_hc0_scaled() {
        let (y, x) = small();
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();
        let c0 = cov_hc0(&x, &res.resid).unwrap();
        let c1 = cov_hc1(&x, &res.resid).unwrap();
        let n = x.nrows() as f64;
        let k = x.ncols() as f64;
        let f = n / (n - k);
        for i in 0..c0.nrows() {
            for j in 0..c0.ncols() {
                assert!((c1[[i, j]] - f * c0[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hac_zero_lag_equals_hc0() {
        let (y, x) = small();
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();
        // HAC with maxlags=0 and no correction is exactly HC0.
        let hac = cov_hac(&x, &res.resid, 0, false).unwrap();
        let hc0 = cov_hc0(&x, &res.resid).unwrap();
        for i in 0..hac.nrows() {
            for j in 0..hac.ncols() {
                assert!((hac[[i, j]] - hc0[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cluster_runs() {
        let (y, x) = small();
        let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();
        let groups: Vec<i64> = (0..12).map(|i| (i % 3) as i64).collect();
        let cov = cov_cluster(&x, &res.resid, &groups, true).unwrap();
        let bse = robust_bse(&cov);
        assert_eq!(bse.len(), 3);
        assert!(bse.iter().all(|&b| b > 0.0));
    }

    #[test]
    fn hat_diag_sums_to_rank() {
        let (_, x) = small();
        let h = hat_diag(&x).unwrap();
        let s: f64 = h.sum();
        // trace of hat matrix equals rank (here k = 3).
        assert!((s - 3.0).abs() < 1e-9);
    }
}
