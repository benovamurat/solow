//! Feature scalers.
//!
//! Five classical scalers ship here, each with a `fit` + `transform` +
//! `fit_transform` + `inverse_transform` shape identical to
//! `preprocessing`:
//!
//! * [`StandardScaler`] — `(x - μ) / σ`, zero mean and unit variance
//!   per column. Uses Welford's one-pass algorithm for numerically
//!   stable statistics; the sample variance is the unbiased estimator
//!   (`n - 1` denominator) by default and can be switched to the
//!   population version (`n`) by [`StandardScaler::population`].
//! * [`MinMaxScaler`] — `(x - min) / (max - min) · (b - a) + a`,
//!   projecting each column onto an arbitrary `[a, b]` range
//!   (default `[0, 1]`).
//! * [`RobustScaler`] — `(x - median) / IQR`, the robust analogue that
//!   is invariant to outliers. Uses the R type-7 quantile (matches
//!   the reference and numpy) so the fit is exactly reproducible.
//! * [`MaxAbsScaler`] — `x / max(|x|)`, projecting each column onto
//!   `[-1, 1]` while preserving sparsity and sign.
//! * [`Normalizer`] — per-**row** rescaling to unit L¹, L², or L∞
//!   norm. This is the only scaler that operates row-wise, following
//!   `preprocessing.Normalizer`.
//!
//! ## Numerical stability
//!
//! Column variance uses Welford's recurrence (Chan-Golub-LeVeque);
//! this eliminates the catastrophic cancellation of the naive
//! "sum-of-squares minus mean-squared" identity when the feature has a
//! large mean and a small variance. See
//! [`solow_core::numeric::compensated_mean_and_var`] for the primitive.
//!
//! ## Edge cases
//!
//! When a column has zero variance / range / max-abs, the corresponding
//! scale is set to `1.0` and reported in [`StandardScaler::scale`] etc.
//! This makes the scaler idempotent on constant columns (matches
//! the reference behaviour with `with_std=True`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use solow_core::{numeric::compensated_mean_and_var, Error, Result};

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/// Zero-mean unit-variance scaler.
///
/// Fits per-column mean `μⱼ` and standard deviation `σⱼ`, then
/// transforms each row via `x ← (x - μ) / σ`. On the inverse
/// transform, `x ← x · σ + μ`.
///
/// # Complexity
///
/// * `fit`: `O(n · d)` time, `O(d)` space.
/// * `transform` / `inverse_transform`: `O(n · d)` time.
///
/// # Reference
///
/// Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). *Algorithms for
/// computing the sample variance: analysis and recommendations.*
/// The American Statistician, 37(3), 242-247.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct StandardScaler {
    /// Per-column mean.
    pub mean: Array1<f64>,
    /// Per-column standard deviation (with `n - 1` denominator by default;
    /// `1.0` for zero-variance columns to keep the transform well-defined).
    pub scale: Array1<f64>,
    /// Whether variance uses the biased `n` denominator (default is `n - 1`).
    biased: bool,
    /// Whether to center (subtract the mean) on transform.
    with_mean: bool,
    /// Whether to scale (divide by the std) on transform.
    with_std: bool,
}

impl StandardScaler {
    /// Fit the scaler on `x` (rows = samples, cols = features).
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, true, true, false)
    }

    /// Full-configuration fit.
    ///
    /// * `with_mean` — subtract the mean on transform.
    /// * `with_std` — divide by the std on transform.
    /// * `biased` — use the population variance (`n` denominator) rather
    ///   than the sample variance (`n - 1`).
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        with_mean: bool,
        with_std: bool,
        biased: bool,
    ) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "StandardScaler::fit: x must have at least one row and one column".into(),
            ));
        }
        let d = x.ncols();
        let n = x.nrows();
        let mut mean = Array1::<f64>::zeros(d);
        let mut scale = Array1::<f64>::ones(d);
        for j in 0..d {
            let (m, v) = compensated_mean_and_var(x.column(j).iter().copied());
            mean[j] = m;
            // Convert unbiased (n-1) to biased (n) or vice versa if requested.
            let var = if biased && n > 1 {
                v * (n as f64 - 1.0) / n as f64
            } else {
                v
            };
            let s = var.sqrt();
            scale[j] = if s > 0.0 && s.is_finite() { s } else { 1.0 };
        }
        Ok(Self {
            mean,
            scale,
            biased,
            with_mean,
            with_std,
        })
    }

    /// Switch to the biased (population) variance.
    pub fn population(mut self) -> Self {
        self.biased = true;
        self
    }

    /// Fit + transform in one call — returns a freshly-allocated matrix.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let s = Self::fit(x)?;
        let t = s.transform(x)?;
        Ok((s, t))
    }

    /// Apply the scaler to `x`. Returns a new matrix.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.mean.len() {
            return Err(Error::Shape(format!(
                "StandardScaler::transform: expected {} columns, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                if self.with_mean {
                    *v -= self.mean[j];
                }
                if self.with_std {
                    *v /= self.scale[j];
                }
            }
        }
        Ok(out)
    }

    /// Invert the transform: `x ← x · σ + μ`.
    pub fn inverse_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.mean.len() {
            return Err(Error::Shape(format!(
                "StandardScaler::inverse_transform: expected {} columns, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                if self.with_std {
                    *v *= self.scale[j];
                }
                if self.with_mean {
                    *v += self.mean[j];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/// Range-scaler mapping each column to a target `[low, high]` interval.
///
/// Transform: `x ← (x - min_j) / (max_j - min_j) · (high - low) + low`.
/// Constant columns are mapped to the midpoint `(low + high) / 2`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MinMaxScaler {
    /// Per-column minimum observed during fit.
    pub data_min: Array1<f64>,
    /// Per-column maximum observed during fit.
    pub data_max: Array1<f64>,
    /// Per-column scale factor `(high - low) / (max - min)` (or `0` for
    /// constant columns).
    pub scale: Array1<f64>,
    /// Per-column translation.
    pub min: Array1<f64>,
    /// Target lower bound.
    pub low: f64,
    /// Target upper bound.
    pub high: f64,
}

impl MinMaxScaler {
    /// Fit onto `x` with the default target range `[0, 1]`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_range(x, 0.0, 1.0)
    }

    /// Fit onto `x` with a custom target `[low, high]`.
    pub fn fit_range(x: ArrayView2<'_, f64>, low: f64, high: f64) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "MinMaxScaler::fit_range: x must have at least one row and one column".into(),
            ));
        }
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(Error::Value(format!(
                "MinMaxScaler::fit_range: require low < high (got low={low}, high={high})"
            )));
        }
        let d = x.ncols();
        let mut data_min = Array1::<f64>::zeros(d);
        let mut data_max = Array1::<f64>::zeros(d);
        let mut scale = Array1::<f64>::ones(d);
        let mut min = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &v in x.column(j).iter() {
                if !v.is_finite() {
                    return Err(Error::Value(format!(
                        "MinMaxScaler::fit_range: column {j} contains non-finite values"
                    )));
                }
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            data_min[j] = mn;
            data_max[j] = mx;
            let range = mx - mn;
            if range > 0.0 {
                scale[j] = (high - low) / range;
                min[j] = low - mn * scale[j];
            } else {
                // Constant column → map to midpoint.
                scale[j] = 0.0;
                min[j] = 0.5 * (low + high);
            }
        }
        Ok(Self {
            data_min,
            data_max,
            scale,
            min,
            low,
            high,
        })
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let s = Self::fit(x)?;
        let t = s.transform(x)?;
        Ok((s, t))
    }

    /// Apply the scaler to `x`.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.scale.len() {
            return Err(Error::Shape(format!(
                "MinMaxScaler::transform: expected {} columns, got {}",
                self.scale.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                *v = *v * self.scale[j] + self.min[j];
            }
        }
        Ok(out)
    }

    /// Invert the transform.
    pub fn inverse_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.scale.len() {
            return Err(Error::Shape(format!(
                "MinMaxScaler::inverse_transform: expected {} columns, got {}",
                self.scale.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            if self.scale[j] > 0.0 {
                for v in col.iter_mut() {
                    *v = (*v - self.min[j]) / self.scale[j];
                }
            } else {
                // Constant column — recover the original constant value.
                for v in col.iter_mut() {
                    *v = self.data_min[j];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

/// Median / IQR scaler robust to outliers.
///
/// Transform: `x ← (x - median_j) / IQR_j`. The interquartile range
/// (IQR) defaults to the 25th-to-75th-percentile spread and can be
/// customised via [`RobustScaler::quantile_range`]. Quantiles use R
/// type-7 linear interpolation (matches `numpy.quantile` /
/// `preprocessing.RobustScaler`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RobustScaler {
    /// Per-column median (50th percentile).
    pub center: Array1<f64>,
    /// Per-column IQR (or `1.0` for degenerate columns).
    pub scale: Array1<f64>,
    /// Lower quantile fraction (default 0.25).
    pub q_low: f64,
    /// Upper quantile fraction (default 0.75).
    pub q_high: f64,
    /// Whether to subtract the median on transform.
    with_center: bool,
    /// Whether to divide by the IQR on transform.
    with_scale: bool,
}

impl RobustScaler {
    /// Fit with the default `[0.25, 0.75]` quantile range.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_range(x, 0.25, 0.75, true, true)
    }

    /// Full-configuration fit.
    pub fn fit_range(
        x: ArrayView2<'_, f64>,
        q_low: f64,
        q_high: f64,
        with_center: bool,
        with_scale: bool,
    ) -> Result<Self> {
        if !(0.0 <= q_low && q_low < q_high && q_high <= 1.0) {
            return Err(Error::Value(format!(
                "RobustScaler::fit_range: require 0 ≤ q_low < q_high ≤ 1 (got {q_low}, {q_high})"
            )));
        }
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "RobustScaler::fit_range: x must have at least one row and one column".into(),
            ));
        }
        let d = x.ncols();
        let mut center = Array1::<f64>::zeros(d);
        let mut scale = Array1::<f64>::ones(d);
        for j in 0..d {
            let mut col: Vec<f64> = x.column(j).iter().copied().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            center[j] = quantile_sorted(&col, 0.5);
            let iqr = quantile_sorted(&col, q_high) - quantile_sorted(&col, q_low);
            scale[j] = if iqr > 0.0 && iqr.is_finite() {
                iqr
            } else {
                1.0
            };
        }
        Ok(Self {
            center,
            scale,
            q_low,
            q_high,
            with_center,
            with_scale,
        })
    }

    /// Set a custom `quantile_range`, e.g. `(0.10, 0.90)` for a broader spread.
    pub fn quantile_range(mut self, q_low: f64, q_high: f64) -> Self {
        self.q_low = q_low;
        self.q_high = q_high;
        self
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let s = Self::fit(x)?;
        let t = s.transform(x)?;
        Ok((s, t))
    }

    /// Apply the scaler.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.center.len() {
            return Err(Error::Shape(format!(
                "RobustScaler::transform: expected {} columns, got {}",
                self.center.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                if self.with_center {
                    *v -= self.center[j];
                }
                if self.with_scale {
                    *v /= self.scale[j];
                }
            }
        }
        Ok(out)
    }

    /// Invert the transform.
    pub fn inverse_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.center.len() {
            return Err(Error::Shape(format!(
                "RobustScaler::inverse_transform: expected {} columns, got {}",
                self.center.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                if self.with_scale {
                    *v *= self.scale[j];
                }
                if self.with_center {
                    *v += self.center[j];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MaxAbsScaler
// ---------------------------------------------------------------------------

/// Scaler that divides each column by its maximum absolute value.
///
/// Transform: `x ← x / max|x_j|`. Preserves zero entries and sign,
/// making it the sparsity-preserving analogue of [`MinMaxScaler`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MaxAbsScaler {
    /// Per-column maximum absolute value observed during fit.
    pub max_abs: Array1<f64>,
    /// Per-column scale (`max_abs`, with `1.0` fallback for zero columns).
    pub scale: Array1<f64>,
}

impl MaxAbsScaler {
    /// Fit on `x`.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "MaxAbsScaler::fit: x must have at least one row and one column".into(),
            ));
        }
        let d = x.ncols();
        let mut max_abs = Array1::<f64>::zeros(d);
        let mut scale = Array1::<f64>::ones(d);
        for j in 0..d {
            let m = x
                .column(j)
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max);
            max_abs[j] = m;
            scale[j] = if m > 0.0 { m } else { 1.0 };
        }
        Ok(Self { max_abs, scale })
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let s = Self::fit(x)?;
        let t = s.transform(x)?;
        Ok((s, t))
    }

    /// Apply the scaler.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.scale.len() {
            return Err(Error::Shape(format!(
                "MaxAbsScaler::transform: expected {} columns, got {}",
                self.scale.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                *v /= self.scale[j];
            }
        }
        Ok(out)
    }

    /// Invert the transform.
    pub fn inverse_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.scale.len() {
            return Err(Error::Shape(format!(
                "MaxAbsScaler::inverse_transform: expected {} columns, got {}",
                self.scale.len(),
                x.ncols()
            )));
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            for v in col.iter_mut() {
                *v *= self.scale[j];
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Normalizer
// ---------------------------------------------------------------------------

/// Norm the [`Normalizer`] rescales each row by.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// L¹: sum of absolute values.
    L1,
    /// L²: Euclidean length (default).
    L2,
    /// L∞: max absolute value.
    Max,
}

/// Per-row rescaling to unit norm.
///
/// Transform: for each row, `x ← x / ‖x‖_p` where `p` is one of
/// [`NormKind::L1`], [`NormKind::L2`], [`NormKind::Max`]. Zero rows are
/// left unchanged (matches `preprocessing.Normalizer`).
///
/// Unlike the column-wise scalers, [`Normalizer`] does not need a fit —
/// the per-row norm is computed at transform time. It's exposed as a
/// struct anyway to fit into the common preprocessing API.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Normalizer {
    /// The norm to rescale by.
    pub kind: NormKind,
}

impl Normalizer {
    /// Build a normalizer with the given [`NormKind`].
    pub fn new(kind: NormKind) -> Self {
        Self { kind }
    }

    /// L² by default (matches the reference).
    pub fn default() -> Self {
        Self { kind: NormKind::L2 }
    }

    /// Apply the normalizer.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut out = x.to_owned();
        for mut row in out.axis_iter_mut(Axis(0)) {
            let n = match self.kind {
                NormKind::L1 => row.iter().map(|v| v.abs()).sum::<f64>(),
                NormKind::L2 => row.iter().map(|v| v * v).sum::<f64>().sqrt(),
                NormKind::Max => row.iter().map(|v| v.abs()).fold(0.0, f64::max),
            };
            if n > 0.0 {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
        }
        Ok(out)
    }

    /// Row-wise fit-transform is identical to transform (no state).
    pub fn fit_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// Suppress unused-import lint when tests don't use every helper.
#[allow(dead_code)]
fn _unused(_v: ArrayView1<'_, f64>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn standard_scaler_zeroes_mean_and_units_variance() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let scaler = StandardScaler::fit(x.view()).unwrap();
        assert_abs_diff_eq!(scaler.mean[0], 2.5, epsilon = 1e-12);
        assert_abs_diff_eq!(scaler.mean[1], 25.0, epsilon = 1e-12);
        let t = scaler.transform(x.view()).unwrap();
        // Sample mean of every column of `t` should be ~0.
        for j in 0..2 {
            let m: f64 = t.column(j).iter().sum::<f64>() / t.nrows() as f64;
            assert_abs_diff_eq!(m, 0.0, epsilon = 1e-12);
        }
        // Inverse round-trip.
        let back = scaler.inverse_transform(t.view()).unwrap();
        for (a, b) in back.iter().zip(x.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn minmax_scaler_maps_to_target_range() {
        let x = array![[-1.0, 100.0], [0.0, 0.0], [1.0, -100.0]];
        let scaler = MinMaxScaler::fit_range(x.view(), 0.0, 1.0).unwrap();
        let t = scaler.transform(x.view()).unwrap();
        for j in 0..2 {
            let mn = t.column(j).iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = t
                .column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert_abs_diff_eq!(mn, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(mx, 1.0, epsilon = 1e-12);
        }
        // Round-trip.
        let back = scaler.inverse_transform(t.view()).unwrap();
        for (a, b) in back.iter().zip(x.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn robust_scaler_is_immune_to_a_lone_outlier() {
        // A big outlier at row 0 col 0 pulls MinMaxScaler badly, but
        // RobustScaler stays close to the "clean" range.
        // Column 0 sorted: [1, 2, 3, 4, 1e6]; median = 3, Q1 = 2, Q3 = 4,
        // IQR = 2. None of these are driven by the outlier.
        let x = array![[1e6, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let robust = RobustScaler::fit(x.view()).unwrap();
        assert_abs_diff_eq!(robust.center[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(robust.scale[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn maxabs_scaler_preserves_sign() {
        let x = array![[2.0, -4.0], [-1.0, 2.0], [0.0, 0.0]];
        let scaler = MaxAbsScaler::fit(x.view()).unwrap();
        let t = scaler.transform(x.view()).unwrap();
        assert_abs_diff_eq!(t[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[[1, 0]], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(t[[0, 1]], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn normalizer_l2_gives_unit_norm_rows() {
        let x = array![[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]];
        let n = Normalizer::default();
        let t = n.transform(x.view()).unwrap();
        assert_abs_diff_eq!(
            (t.row(0).iter().map(|v| v * v).sum::<f64>()).sqrt(),
            1.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(t[[2, 0]], 0.0, epsilon = 1e-12); // zero row untouched
    }

    #[test]
    fn standard_scaler_handles_constant_columns() {
        let x = array![[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]];
        let scaler = StandardScaler::fit(x.view()).unwrap();
        assert_eq!(scaler.scale[0], 1.0); // no divide-by-zero
        let t = scaler.transform(x.view()).unwrap();
        assert_abs_diff_eq!(t[[0, 0]], 0.0, epsilon = 1e-12);
    }
}
