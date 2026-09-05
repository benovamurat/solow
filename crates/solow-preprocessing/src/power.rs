//! Power transforms — [`PowerTransformer`] with the Yeo-Johnson
//! (2000) and Box-Cox (1964) families, and [`QuantileTransformer`]
//! for empirical-CDF mapping to a uniform or standard-normal target.
//!
//! Both are staple "make-your-data-more-Gaussian" preprocessors —
//! the reference `PowerTransformer` and `QuantileTransformer` respectively.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use solow_core::{Error, Result};

/// Power-transform family.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PowerMethod {
    /// Yeo-Johnson (works for real-valued inputs).
    YeoJohnson,
    /// Box-Cox (requires strictly positive inputs).
    BoxCox,
}

/// Column-wise power transform.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PowerTransformer {
    /// Per-column optimal λ (maximum-likelihood).
    pub lambdas: Array1<f64>,
    /// Family used.
    pub method: PowerMethod,
    /// Whether to standardise (zero mean, unit variance) after the transform.
    pub standardize: bool,
    /// Post-transform mean (0 when `standardize = false`).
    pub post_mean: Array1<f64>,
    /// Post-transform std (1 when `standardize = false`).
    pub post_std: Array1<f64>,
}

impl PowerTransformer {
    /// Fit with Yeo-Johnson and standardisation (the reference default).
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, PowerMethod::YeoJohnson, true)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        method: PowerMethod,
        standardize: bool,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "PowerTransformer::fit_with: x must be non-empty".into(),
            ));
        }
        if matches!(method, PowerMethod::BoxCox) {
            for &v in x.iter() {
                if v <= 0.0 {
                    return Err(Error::Value(
                        "PowerTransformer::fit_with: Box-Cox requires strictly positive inputs"
                            .into(),
                    ));
                }
            }
        }
        let d = x.ncols();
        let mut lambdas = Array1::<f64>::zeros(d);
        for j in 0..d {
            let col: Vec<f64> = x.column(j).iter().copied().collect();
            lambdas[j] = fit_lambda(&col, method);
        }
        // Transform, then optionally standardise on the transformed columns.
        let tr = transform_matrix(x, &lambdas, method);
        let (post_mean, post_std) = if standardize {
            column_stats(&tr)
        } else {
            (Array1::<f64>::zeros(d), Array1::<f64>::ones(d))
        };
        Ok(Self {
            lambdas,
            method,
            standardize,
            post_mean,
            post_std,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.lambdas.len() {
            return Err(Error::Shape(format!(
                "PowerTransformer::transform: expected {} columns, got {}",
                self.lambdas.len(),
                x.ncols()
            )));
        }
        let mut tr = transform_matrix(x, &self.lambdas, self.method);
        if self.standardize {
            for (j, mut col) in tr.axis_iter_mut(Axis(1)).enumerate() {
                for v in col.iter_mut() {
                    *v = (*v - self.post_mean[j]) / self.post_std[j];
                }
            }
        }
        Ok(tr)
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let pt = Self::fit(x)?;
        let tr = pt.transform(x)?;
        Ok((pt, tr))
    }
}

fn transform_matrix(
    x: ArrayView2<'_, f64>,
    lambdas: &Array1<f64>,
    method: PowerMethod,
) -> Array2<f64> {
    let (n, d) = (x.nrows(), x.ncols());
    let mut out = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            out[[i, j]] = apply(x[[i, j]], lambdas[j], method);
        }
    }
    out
}

fn apply(v: f64, lambda: f64, method: PowerMethod) -> f64 {
    match method {
        PowerMethod::YeoJohnson => {
            if v >= 0.0 {
                if lambda.abs() < 1e-9 {
                    (1.0 + v).ln()
                } else {
                    ((1.0 + v).powf(lambda) - 1.0) / lambda
                }
            } else if (lambda - 2.0).abs() < 1e-9 {
                -(1.0 - v).ln()
            } else {
                -(((1.0 - v).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
            }
        }
        PowerMethod::BoxCox => {
            if lambda.abs() < 1e-9 {
                v.ln()
            } else {
                (v.powf(lambda) - 1.0) / lambda
            }
        }
    }
}

fn fit_lambda(col: &[f64], method: PowerMethod) -> f64 {
    // Maximum-likelihood over a coarse then fine grid — Brent's method
    // is overkill for a one-parameter search on a smooth objective.
    let grid_coarse: Vec<f64> = (-40..=40).map(|i| i as f64 * 0.1).collect(); // [-4, 4]
    let (l_coarse, _) = grid_search(col, &grid_coarse, method);
    let grid_fine: Vec<f64> = (0..=200)
        .map(|i| l_coarse - 0.2 + i as f64 * 0.002)
        .collect();
    let (best_l, _) = grid_search(col, &grid_fine, method);
    best_l
}

fn grid_search(col: &[f64], grid: &[f64], method: PowerMethod) -> (f64, f64) {
    let mut best_l = 0.0_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for &l in grid {
        let ll = yeo_johnson_log_likelihood(col, l, method);
        if ll > best_ll {
            best_ll = ll;
            best_l = l;
        }
    }
    (best_l, best_ll)
}

fn yeo_johnson_log_likelihood(col: &[f64], lambda: f64, method: PowerMethod) -> f64 {
    let n = col.len() as f64;
    // Log-Jacobian contribution.
    let mut log_j = 0.0_f64;
    let mut transformed = Vec::with_capacity(col.len());
    for &v in col {
        transformed.push(apply(v, lambda, method));
        log_j += log_derivative(v, lambda, method);
    }
    // Post-transform variance.
    let mean: f64 = transformed.iter().sum::<f64>() / n;
    let var: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * n * var.ln() + log_j
}

fn log_derivative(v: f64, lambda: f64, method: PowerMethod) -> f64 {
    match method {
        PowerMethod::YeoJohnson => {
            if v >= 0.0 {
                (lambda - 1.0) * (1.0 + v).ln()
            } else {
                (1.0 - lambda) * (1.0 - v).ln()
            }
        }
        PowerMethod::BoxCox => (lambda - 1.0) * v.ln(),
    }
}

fn column_stats(x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n, d) = x.dim();
    let mut mean = Array1::<f64>::zeros(d);
    for j in 0..d {
        for i in 0..n {
            mean[j] += x[[i, j]];
        }
        mean[j] /= n as f64;
    }
    let mut std = Array1::<f64>::zeros(d);
    for j in 0..d {
        for i in 0..n {
            std[j] += (x[[i, j]] - mean[j]).powi(2);
        }
        std[j] = (std[j] / (n as f64 - 1.0).max(1.0)).sqrt().max(1e-12);
    }
    (mean, std)
}

// ---------------------------------------------------------------------------
// QuantileTransformer
// ---------------------------------------------------------------------------

/// Output distribution for [`QuantileTransformer`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuantileOutput {
    /// Uniform on `[0, 1]`.
    Uniform,
    /// Standard normal.
    Normal,
}

/// Empirical-CDF quantile transformer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileTransformer {
    /// Per-column sorted quantile knots.
    pub quantiles: Vec<Vec<f64>>,
    /// Sorted reference-distribution quantile knots (shared across
    /// columns — depends only on `n_quantiles` and `output`).
    pub references: Vec<f64>,
    /// Target output distribution.
    pub output: QuantileOutput,
    /// Number of quantile knots.
    pub n_quantiles: usize,
}

impl QuantileTransformer {
    /// Fit with `n_quantiles = min(1000, n)` and `Uniform` output
    /// (the reference defaults).
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, 1000.min(x.nrows()), QuantileOutput::Uniform)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_quantiles: usize,
        output: QuantileOutput,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "QuantileTransformer::fit_with: x must be non-empty".into(),
            ));
        }
        if n_quantiles < 2 {
            return Err(Error::Value(format!(
                "QuantileTransformer::fit_with: n_quantiles must be ≥ 2 (got {n_quantiles})"
            )));
        }
        let d = x.ncols();
        // Per-column sorted values at the reference probabilities.
        let mut quantiles = Vec::with_capacity(d);
        for j in 0..d {
            let mut col: Vec<f64> = x.column(j).iter().copied().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut q = Vec::with_capacity(n_quantiles);
            for k in 0..n_quantiles {
                let p = k as f64 / (n_quantiles - 1) as f64;
                q.push(quantile_sorted(&col, p));
            }
            quantiles.push(q);
        }
        // Reference probabilities → output-distribution values.
        let references: Vec<f64> = (0..n_quantiles)
            .map(|k| {
                let p = k as f64 / (n_quantiles - 1) as f64;
                match output {
                    QuantileOutput::Uniform => p,
                    QuantileOutput::Normal => normal_inverse_cdf(p.clamp(1e-6, 1.0 - 1e-6)),
                }
            })
            .collect();
        Ok(Self {
            quantiles,
            references,
            output,
            n_quantiles,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.quantiles.len() {
            return Err(Error::Shape(format!(
                "QuantileTransformer::transform: expected {} cols, got {}",
                self.quantiles.len(),
                x.ncols()
            )));
        }
        let (n, d) = (x.nrows(), x.ncols());
        let mut out = Array2::<f64>::zeros((n, d));
        for j in 0..d {
            let q = &self.quantiles[j];
            for i in 0..n {
                out[[i, j]] = interp(x[[i, j]], q, &self.references);
            }
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let qt = Self::fit(x)?;
        let tr = qt.transform(x)?;
        Ok((qt, tr))
    }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    (1.0 - frac) * sorted[lo] + frac * sorted[hi]
}

fn interp(v: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if v <= xs[0] {
        return ys[0];
    }
    if v >= xs[n - 1] {
        return ys[n - 1];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (x0, x1) = (xs[lo], xs[hi]);
    let (y0, y1) = (ys[lo], ys[hi]);
    if (x1 - x0).abs() < 1e-300 {
        return y1;
    }
    y0 + (y1 - y0) * (v - x0) / (x1 - x0)
}

fn normal_inverse_cdf(p: f64) -> f64 {
    // Acklam's approximation — max relative error ~1.15e-9 in the central
    // region; adequate for the QuantileTransformer's normal output.
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn yeo_johnson_reduces_skew() {
        // Heavily skewed positive data — Yeo-Johnson should whiten the mean.
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [10.0],
            [20.0],
            [50.0],
            [100.0],
            [200.0]
        ];
        let (pt, tr) = PowerTransformer::fit_transform(x.view()).unwrap();
        // Sanity: fitted λ finite, transformed column has ~0 mean when standardised.
        assert!(pt.lambdas[0].is_finite());
        let mean: f64 = tr.column(0).iter().sum::<f64>() / tr.nrows() as f64;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn quantile_transformer_uniform_maps_to_grid() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let (_, tr) = QuantileTransformer::fit_transform(x.view()).unwrap();
        // Transform of the training data lands in [0, 1] and the range is spanned.
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &v in tr.iter() {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        assert!(mn >= 0.0);
        assert!(mx <= 1.0);
        assert!(mx - mn > 0.5);
    }
}
