//! RANSACRegressor and TheilSenRegressor — robust linear-model
//! estimators.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

/// Fitted RANSAC regressor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RansacRegressor {
    /// Inlier coefficients (`d + 1`; last = intercept).
    pub coef: Array1<f64>,
    /// Inlier mask.
    pub inlier_mask: Vec<bool>,
    /// Iterations required.
    pub n_iter: usize,
    /// Seed used.
    pub seed: u64,
}

impl RansacRegressor {
    /// Fit with the reference defaults `min_samples = d + 1`,
    /// `residual_threshold = MAD(y)`, `max_trials = 100`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, seed: u64) -> Result<Self> {
        Self::fit_with(x, y, x.ncols() + 1, None, 100, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        min_samples: usize,
        residual_threshold: Option<f64>,
        max_trials: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("RansacRegressor: y/x row mismatch".into()));
        }
        if min_samples < d + 1 || min_samples > n {
            return Err(Error::Value(format!(
                "RansacRegressor: min_samples must be in [{}, {n}] (got {min_samples})",
                d + 1
            )));
        }
        // Residual threshold from MAD if not provided.
        let threshold = residual_threshold.unwrap_or_else(|| {
            let mean = y.iter().sum::<f64>() / n as f64;
            let mut dev: Vec<f64> = y.iter().map(|v| (v - mean).abs()).collect();
            dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = dev[dev.len() / 2];
            (1.4826 * m).max(1e-6)
        });
        let mut state = seed.wrapping_add(0xC0DE_F00D);
        let mut best_coef = Array1::<f64>::zeros(d + 1);
        let mut best_mask = vec![false; n];
        let mut best_count = 0_usize;
        let mut iters = 0_usize;
        for trial in 0..max_trials {
            iters = trial + 1;
            // Sample `min_samples` unique row indices.
            let mut sample = Vec::with_capacity(min_samples);
            let mut seen = std::collections::HashSet::new();
            while sample.len() < min_samples {
                let idx = uniform_index(&mut state, n as u64);
                if seen.insert(idx) {
                    sample.push(idx);
                }
            }
            let (xs, ys) = row_subset(x, y, &sample);
            let Ok(coef) = ols_fit(xs.view(), ys.view()) else {
                continue;
            };
            let mut mask = vec![false; n];
            let mut count = 0_usize;
            for i in 0..n {
                let mut pred = coef[d];
                for j in 0..d {
                    pred += coef[j] * x[[i, j]];
                }
                if (y[i] - pred).abs() < threshold {
                    mask[i] = true;
                    count += 1;
                }
            }
            if count > best_count {
                best_count = count;
                best_mask = mask;
                best_coef = coef;
            }
        }
        // Refit on the inliers.
        let inlier_rows: Vec<usize> = (0..n).filter(|&i| best_mask[i]).collect();
        if inlier_rows.len() >= d + 1 {
            let (xs, ys) = row_subset(x, y, &inlier_rows);
            if let Ok(coef) = ols_fit(xs.view(), ys.view()) {
                best_coef = coef;
            }
        }
        Ok(Self {
            coef: best_coef,
            inlier_mask: best_mask,
            n_iter: iters,
            seed,
        })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.coef.len() != d + 1 {
            return Err(Error::Shape("RansacRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += self.coef[j] * x[[i, j]];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

/// Fitted TheilSen regressor (median-based linear model).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct TheilSenRegressor {
    /// Coefficients (`d + 1`; last = intercept).
    pub coef: Array1<f64>,
    /// Seed used for the deterministic random subsampling.
    pub seed: u64,
}

impl TheilSenRegressor {
    /// Fit with `n_subsamples = min(200, C(n, d + 1))`.
    pub fn fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, seed: u64) -> Result<Self> {
        Self::fit_with(x, y, 200, seed)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        n_subsamples: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if y.len() != n {
            return Err(Error::Shape("TheilSenRegressor: y/x row mismatch".into()));
        }
        let min_samples = d + 1;
        if n < min_samples {
            return Err(Error::Value(format!(
                "TheilSenRegressor: need ≥ {} samples", min_samples
            )));
        }
        let mut state = seed.wrapping_add(0xC0DE_F00D);
        let mut per_coef: Vec<Vec<f64>> = vec![Vec::with_capacity(n_subsamples); d + 1];
        for _ in 0..n_subsamples {
            let mut sample = Vec::with_capacity(min_samples);
            let mut seen = std::collections::HashSet::new();
            while sample.len() < min_samples {
                let idx = uniform_index(&mut state, n as u64);
                if seen.insert(idx) {
                    sample.push(idx);
                }
            }
            let (xs, ys) = row_subset(x, y, &sample);
            if let Ok(coef) = ols_fit(xs.view(), ys.view()) {
                for j in 0..=d {
                    per_coef[j].push(coef[j]);
                }
            }
        }
        let mut coef = Array1::<f64>::zeros(d + 1);
        for j in 0..=d {
            per_coef[j].sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = per_coef[j].len();
            coef[j] = if m == 0 { 0.0 } else { per_coef[j][m / 2] };
        }
        Ok(Self { coef, seed })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let d = x.ncols();
        if self.coef.len() != d + 1 {
            return Err(Error::Shape("TheilSenRegressor::predict: shape mismatch".into()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.coef[d];
            for j in 0..d {
                s += self.coef[j] * x[[i, j]];
            }
            out[i] = s;
        }
        Ok(out)
    }
}

fn row_subset(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    rows: &[usize],
) -> (Array2<f64>, Array1<f64>) {
    let d = x.ncols();
    let mut xs = Array2::<f64>::zeros((rows.len(), d));
    let mut ys = Array1::<f64>::zeros(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..d {
            xs[[r, j]] = x[[i, j]];
        }
        ys[r] = y[i];
    }
    (xs, ys)
}

fn ols_fit(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Array1<f64>> {
    // Add intercept column, then solve normal equations.
    let n = x.nrows();
    let d = x.ncols();
    let mut xd = Array2::<f64>::zeros((n, d + 1));
    for i in 0..n {
        for j in 0..d {
            xd[[i, j]] = x[[i, j]];
        }
        xd[[i, d]] = 1.0;
    }
    let mut xtx = Array2::<f64>::zeros((d + 1, d + 1));
    for i in 0..(d + 1) {
        for j in 0..(d + 1) {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += xd[[r, i]] * xd[[r, j]];
            }
            xtx[[i, j]] = s;
        }
    }
    let mut xty = Array1::<f64>::zeros(d + 1);
    for i in 0..(d + 1) {
        let mut s = 0.0_f64;
        for r in 0..n {
            s += xd[[r, i]] * y[r];
        }
        xty[i] = s;
    }
    let inv = invert(&xtx)?;
    Ok(matvec(&inv, &xty))
}

fn matvec(m: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let n = m.nrows();
    let p = m.ncols();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += m[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
}

fn invert(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Value("robust_regression::invert: singular".into()));
        }
        if piv != i {
            a.swap(i, piv);
        }
        let d = a[i][i];
        for c in 0..(2 * n) {
            a[i][c] /= d;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in 0..(2 * n) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = a[i][n + j];
        }
    }
    Ok(inv)
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        (state.wrapping_mul(3) % n) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ransac_recovers_line_despite_outliers() {
        // y = 2x + 1, with 2 outliers.
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![3.0_f64, 5.0, 7.0, 9.0, 100.0, 13.0, 15.0, -50.0];
        let m = RansacRegressor::fit(x.view(), y.view(), 42).unwrap();
        assert!((m.coef[0] - 2.0).abs() < 0.5);
        assert!((m.coef[1] - 1.0).abs() < 1.0);
    }

    #[test]
    fn theil_sen_recovers_line_from_a_noisy_dataset() {
        let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.1_f64, 4.0, 5.9, 8.1, 10.0, 11.9];
        let m = TheilSenRegressor::fit_with(x.view(), y.view(), 30, 7).unwrap();
        assert!((m.coef[0] - 2.0).abs() < 0.5);
    }
}
