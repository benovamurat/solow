//! Vector autoregression (VAR) estimated by equation-by-equation OLS on the
//! stacked lag design, mirroring the reference `VAR(endog).fit(p)`.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_linalg::{cholesky, inv};

/// Deterministic trend term included in the VAR design.
///
/// Only the constant (`C`) is currently exposed, matching the default of the
/// reference estimator and the scope of this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// No deterministic term.
    N,
    /// A single constant (intercept) term. This is the reference default.
    C,
}

impl Trend {
    /// Number of deterministic columns prepended to the design matrix.
    fn order(self) -> usize {
        match self {
            Trend::N => 0,
            Trend::C => 1,
        }
    }
}

/// A vector autoregression of fixed lag order `p` with an optional constant.
///
/// For `K`-dimensional data `y_t` and lag order `p`, the model is
///
/// ```text
/// y_t = nu + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t
/// ```
///
/// estimated by ordinary least squares applied independently to each of the
/// `K` equations on the shared design
/// `Z_t = [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]` (for `trend = C`). The lag blocks
/// are ordered from the most recent (`y_{t-1}`) to the oldest (`y_{t-p}`),
/// matching Lütkepohl's convention and the reference implementation.
#[derive(Debug, Clone)]
pub struct Var {
    endog: Array2<f64>,
    trend: Trend,
}

/// Fitted results of a [`Var`] model.
#[derive(Debug, Clone)]
pub struct VarResults {
    /// Lag order `p`.
    pub k_ar: usize,
    /// Number of equations / series `K`.
    pub neqs: usize,
    /// Number of observations used in estimation, `T = n_totobs - p`.
    pub nobs: usize,
    /// Number of regressors per equation, `df_model = K p + k_trend`.
    pub df_model: usize,
    /// Residual degrees of freedom, `df_resid = T - df_model`.
    pub df_resid: usize,
    /// Number of deterministic columns (`k_trend`, 1 for `trend = C`).
    pub k_trend: usize,
    /// Estimated coefficients, shape `(df_model, K)`. Row blocks are ordered
    /// `[deterministic; A_1; ...; A_p]` with each `A_i` occupying `K` rows;
    /// column `j` holds the parameters of equation `j`.
    pub params: Array2<f64>,
    /// Intercept vector `nu`, length `K` (empty if `trend = N`).
    pub intercept: Array1<f64>,
    /// Autoregressive coefficient matrices `A_1, ..., A_p`, each `K x K`, with
    /// `coefs[i][[r, c]]` the effect of `y_{t-1-i, c}` on `y_{t, r}`.
    pub coefs: Vec<Array2<f64>>,
    /// In-sample residuals, shape `(T, K)`.
    pub resid: Array2<f64>,
    /// Fitted values, shape `(T, K)`.
    pub fittedvalues: Array2<f64>,
    /// Degrees-of-freedom-adjusted residual covariance `SSE / df_resid`.
    pub sigma_u: Array2<f64>,
    /// Maximum-likelihood residual covariance `SSE / T`.
    pub sigma_u_mle: Array2<f64>,
    /// Gaussian log-likelihood at the estimates.
    pub llf: f64,
    /// Akaike information criterion (Lütkepohl convention).
    pub aic: f64,
    /// Bayesian (Schwarz) information criterion.
    pub bic: f64,
    /// Hannan-Quinn information criterion.
    pub hqic: f64,
    /// Final prediction error.
    pub fpe: f64,
    /// Standard errors of the coefficients, shape `(df_model, K)`.
    pub bse: Array1Of2,
    /// t-statistics `params / bse`, shape `(df_model, K)`.
    pub tvalues: Array2<f64>,
    /// Two-sided p-values from the standard normal, shape `(df_model, K)`.
    pub pvalues: Array2<f64>,
}

// The reference returns `bse`/`stderr` as a 2-D array. We keep the same shape.
type Array1Of2 = Array2<f64>;

impl Var {
    /// Create a VAR model from a `(n_totobs, K)` matrix of observations.
    ///
    /// Returns an error if the series has fewer than one column or row.
    pub fn new(endog: Array2<f64>) -> Result<Self> {
        Self::with_trend(endog, Trend::C)
    }

    /// Create a VAR model with an explicit deterministic [`Trend`].
    pub fn with_trend(endog: Array2<f64>, trend: Trend) -> Result<Self> {
        let (n, k) = endog.dim();
        if k == 0 || n == 0 {
            return Err(Error::Value("endog must be non-empty".into()));
        }
        Ok(Self { endog, trend })
    }

    /// Build the design matrix `Z` (shape `(T, df_model)`) and the sample target
    /// `Y` (shape `(T, K)`) for lag order `p`.
    fn build(&self, p: usize) -> (Array2<f64>, Array2<f64>) {
        let (n, k) = self.endog.dim();
        let t = n - p;
        let ktrend = self.trend.order();
        let df_model = k * p + ktrend;

        let mut z = Array2::<f64>::zeros((t, df_model));
        let mut y = Array2::<f64>::zeros((t, k));
        for i in 0..t {
            let row = p + i; // index of the dependent observation in endog
            for c in 0..k {
                y[[i, c]] = self.endog[[row, c]];
            }
            let mut col = 0;
            if ktrend == 1 {
                z[[i, col]] = 1.0;
                col += 1;
            }
            // Lag blocks, most recent first: y_{t-1}, y_{t-2}, ..., y_{t-p}.
            for lag in 1..=p {
                let src = row - lag;
                for c in 0..k {
                    z[[i, col]] = self.endog[[src, c]];
                    col += 1;
                }
            }
        }
        (z, y)
    }

    /// Estimate a VAR(`p`) model by equation-by-equation OLS.
    pub fn fit(&self, p: usize) -> Result<VarResults> {
        let (n, k) = self.endog.dim();
        if p == 0 {
            return Err(Error::Value("lag order p must be >= 1".into()));
        }
        if p >= n {
            return Err(Error::Value("lag order p must be smaller than nobs".into()));
        }
        let ktrend = self.trend.order();
        let (z, y) = self.build(p);
        let (t, df_model) = z.dim();
        if t <= df_model {
            return Err(Error::Value(
                "not enough observations for the requested lag order".into(),
            ));
        }
        let df_resid = t - df_model;

        // OLS via normal equations: params = (Z'Z)^{-1} Z'Y.
        let ztz = z.t().dot(&z);
        let ztz_inv = inv(&ztz)?;
        let zty = z.t().dot(&y);
        let params = ztz_inv.dot(&zty); // (df_model, K)

        let fitted = z.dot(&params); // (T, K)
        let resid = &y - &fitted;

        // Residual covariances.
        let sse = resid.t().dot(&resid); // (K, K)
        let tf = t as f64;
        let dfr = df_resid as f64;
        let sigma_u = sse.mapv(|v| v / dfr);
        let sigma_u_mle = sse.mapv(|v| v / tf);

        // Log-likelihood uses the ML covariance (divisor T).
        let logdet_mle = logdet_symm(&sigma_u_mle)?;
        let kf = k as f64;
        let llf =
            -(tf * kf / 2.0) * (2.0 * std::f64::consts::PI).ln() - (tf / 2.0) * (logdet_mle + kf);

        // Information criteria (Lütkepohl pp. 146-150), using logdet of the ML
        // covariance and the free-parameter count K p + K * k_trend.
        let free_params = (p * k * k + k * ktrend) as f64;
        let aic = logdet_mle + (2.0 / tf) * free_params;
        let bic = logdet_mle + (tf.ln() / tf) * free_params;
        let hqic = logdet_mle + (2.0 * tf.ln().ln() / tf) * free_params;
        let fpe = ((tf + df_model as f64) / dfr).powf(kf) * logdet_mle.exp();

        // Coefficient standard errors: cov(vec(B)) = kron(inv(Z'Z), sigma_u).
        // Reshaped to (df_model, K) row-major, the (r, c) standard error is
        // sqrt(inv(Z'Z)[r, r] * sigma_u[c, c]).
        let mut bse = Array2::<f64>::zeros((df_model, k));
        for r in 0..df_model {
            for c in 0..k {
                bse[[r, c]] = (ztz_inv[[r, r]] * sigma_u[[c, c]]).sqrt();
            }
        }
        let tvalues = &params / &bse;
        let pvalues = tvalues.mapv(|tv| 2.0 * norm_sf(tv.abs()));

        // Split params into intercept and AR coefficient matrices.
        let intercept = if ktrend == 1 {
            params.slice(s![0, ..]).to_owned()
        } else {
            Array1::<f64>::zeros(0)
        };
        let mut coefs = Vec::with_capacity(p);
        for lag in 0..p {
            let start = ktrend + lag * k;
            // params[start..start+k, :] holds A_lag^T (lag block of Z is the
            // regressor, columns index equations). A_i[[r, c]] is the effect of
            // y_{t-1-lag, c} on equation r, so transpose the block.
            let block = params.slice(s![start..start + k, ..]); // (K, K), rows=regressor, cols=equation
            let mut a = Array2::<f64>::zeros((k, k));
            for r in 0..k {
                for c in 0..k {
                    // equation r, regressor variable c
                    a[[r, c]] = block[[c, r]];
                }
            }
            coefs.push(a);
        }

        Ok(VarResults {
            k_ar: p,
            neqs: k,
            nobs: t,
            df_model,
            df_resid,
            k_trend: ktrend,
            params,
            intercept,
            coefs,
            resid,
            fittedvalues: fitted,
            sigma_u,
            sigma_u_mle,
            llf,
            aic,
            bic,
            hqic,
            fpe,
            bse,
            tvalues,
            pvalues,
        })
    }
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky,
/// `2 * sum(log(diag(L)))`, matching the reference `logdet_symm`.
fn logdet_symm(m: &Array2<f64>) -> Result<f64> {
    let l = cholesky(m)?;
    let n = m.dim().0;
    let mut s = 0.0;
    for i in 0..n {
        s += l[[i, i]].ln();
    }
    Ok(2.0 * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn small_series() -> Array2<f64> {
        // A short, well-conditioned bivariate series.
        array![
            [0.5, 1.0],
            [0.7, 0.8],
            [0.4, 1.2],
            [0.9, 0.6],
            [0.6, 1.1],
            [1.0, 0.5],
            [0.7, 0.9],
            [1.1, 0.4],
            [0.8, 1.0],
            [1.2, 0.3],
            [0.9, 0.8],
            [1.3, 0.5],
        ]
    }

    #[test]
    fn shapes_and_residual_orthogonality() {
        let m = Var::new(small_series()).unwrap();
        let res = m.fit(2).unwrap();
        assert_eq!(res.neqs, 2);
        assert_eq!(res.k_ar, 2);
        assert_eq!(res.nobs, 10);
        assert_eq!(res.df_model, 5); // const + 2 lags * 2 vars
        assert_eq!(res.df_resid, 5);
        assert_eq!(res.params.dim(), (5, 2));
        assert_eq!(res.coefs.len(), 2);
        assert_eq!(res.coefs[0].dim(), (2, 2));
        // OLS residuals are orthogonal to the design's constant column, i.e.
        // each equation's residuals sum to (numerically) zero.
        let col_sums = res.resid.sum_axis(ndarray::Axis(0));
        for v in col_sums.iter() {
            assert!(v.abs() < 1e-10, "residual sum not zero: {v}");
        }
    }

    #[test]
    fn sigma_u_scaling_consistent() {
        let m = Var::new(small_series()).unwrap();
        let res = m.fit(1).unwrap();
        let ratio = res.df_resid as f64 / res.nobs as f64;
        for i in 0..res.neqs {
            for j in 0..res.neqs {
                let expect = res.sigma_u[[i, j]] * ratio;
                assert!((res.sigma_u_mle[[i, j]] - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn n_trend_rejects_zero_lag() {
        let m = Var::with_trend(small_series(), Trend::N).unwrap();
        assert!(m.fit(0).is_err());
        let res = m.fit(1).unwrap();
        assert_eq!(res.k_trend, 0);
        assert_eq!(res.intercept.len(), 0);
    }
}
