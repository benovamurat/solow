//! Model-agnostic feature-importance diagnostics.
//!
//! Right now this module ships a single but essential building block:
//! [`permutation_importance`], the classical Breiman-Fisher score used by
//! `inspection.permutation_importance`. It measures how much a
//! trained model's out-of-sample score drops when the values in one
//! column of the feature matrix are randomly shuffled. A column that a
//! model does not use won't move the score; a column the model relies on
//! will show a large drop.
//!
//! The API is deliberately estimator-free: the caller supplies a closure
//! that maps a **feature matrix** to a **scalar score** — typically by
//! predicting with an already-fit model and running a
//! [`crate::regression`] or [`crate::classification`] metric — and this
//! function repeats the shuffle-and-score procedure `n_repeats` times per
//! column with a deterministic seed.
//!
//! The returned importance is `baseline_score - permuted_score`, using
//! the reference sign convention: **larger means the column mattered
//! more**. This is correct when the score is "higher is better"
//! (accuracy, R², AUC). If you pass a "lower is better" score
//! (MSE, log-loss) the sign flips — either negate your score inside the
//! closure or read the resulting importances as `-Δloss`.

use ndarray::{Array2, ArrayView2, Axis};
use solow_core::{Error, Result};

/// Per-feature importance summary from [`permutation_importance`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureImportance {
    /// Column index in the input matrix.
    pub feature: usize,
    /// Mean importance across the `n_repeats` shuffled evaluations.
    pub importance_mean: f64,
    /// Sample standard deviation of the per-repeat importances (`n - 1`).
    /// `0.0` when `n_repeats == 1`.
    pub importance_std: f64,
    /// The full vector of `n_repeats` per-shuffle importances, in order.
    pub importances: Vec<f64>,
}

/// Permutation feature importance.
///
/// * `x` — the feature matrix (rows = samples, cols = features). It is
///   used only to source column values; nothing is written back into it.
/// * `scorer` — a closure that, given a possibly-permuted feature matrix,
///   returns the model's score on it. This is where the caller ties in a
///   fitted estimator and a metric: `|xp| Ok(r2_score(y.view(), model.predict(&xp)?, None)?)`.
/// * `n_repeats` — number of shuffles per column. Bigger reduces variance.
///   `10` is the reference default.
/// * `seed` — deterministic seed for the permutation PRNG (a portable
///   64-bit MMIX-LCG).
///
/// Returns one [`FeatureImportance`] per column of `x`, in column order.
pub fn permutation_importance<F>(
    x: ArrayView2<'_, f64>,
    scorer: F,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<FeatureImportance>>
where
    F: Fn(ArrayView2<'_, f64>) -> Result<f64>,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(Error::Value(
            "permutation_importance: x must have at least one row and one column".into(),
        ));
    }
    if n_repeats == 0 {
        return Err(Error::Value(
            "permutation_importance: n_repeats must be ≥ 1".into(),
        ));
    }

    let baseline = scorer(x)?;
    if !baseline.is_finite() {
        return Err(Error::Value(
            "permutation_importance: scorer returned a non-finite value on the original matrix"
                .into(),
        ));
    }

    let mut state = seed.wrapping_add(0xA0B1_C2D3_E4F5_0617);
    let n_features = x.ncols();
    let n = x.nrows();
    let mut out = Vec::with_capacity(n_features);

    // Working buffer big enough to hold a permuted copy of `x`.
    let mut work: Array2<f64> = x.to_owned();

    for j in 0..n_features {
        let original_col: Vec<f64> = x.column(j).iter().copied().collect();
        let mut importances = Vec::with_capacity(n_repeats);

        for _ in 0..n_repeats {
            // Fisher-Yates shuffle a fresh permutation of 0..n.
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let k = uniform_index(&mut state, (i + 1) as u64);
                perm.swap(i, k);
            }
            // Write the permuted column into the working matrix.
            {
                let mut col = work.column_mut(j);
                for (row, &src) in perm.iter().enumerate() {
                    col[row] = original_col[src];
                }
            }
            let permuted_score = scorer(work.view())?;
            if !permuted_score.is_finite() {
                // Restore the column before returning so the caller's matrix stays consistent.
                let mut col = work.column_mut(j);
                for (row, &v) in original_col.iter().enumerate() {
                    col[row] = v;
                }
                return Err(Error::Value(
                    "permutation_importance: scorer returned a non-finite value on a permuted matrix".into(),
                ));
            }
            importances.push(baseline - permuted_score);
        }

        // Restore the original column before moving on to the next feature.
        {
            let mut col = work.column_mut(j);
            for (row, &v) in original_col.iter().enumerate() {
                col[row] = v;
            }
        }

        let mean: f64 = importances.iter().sum::<f64>() / importances.len() as f64;
        let std = if importances.len() < 2 {
            0.0
        } else {
            let s2: f64 = importances.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (importances.len() as f64 - 1.0);
            s2.sqrt()
        };
        out.push(FeatureImportance {
            feature: j,
            importance_mean: mean,
            importance_std: std,
            importances,
        });
    }
    // Suppress an unused-import warning when `n_features == 0` is impossible above.
    let _ = Axis(0);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (same MMIX constants as solow-cv)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    let max = u64::MAX - (u64::MAX % n);
    loop {
        let r = lcg_next(state);
        if r < max {
            return (r % n) as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// Partial dependence and Accumulated Local Effects
// ---------------------------------------------------------------------------

use ndarray::Array1;

/// One-way partial-dependence curve.
///
/// * `grid` — the sequence of values the feature is set to.
/// * `values` — the marginalised prediction at each grid point,
///   `E_x[f(x_-j, gᵢ)]` over the sample (a plain average of the
///   `predictor` outputs at each grid point over rows of `x`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartialDependence {
    /// Column of `x` this curve was computed on.
    pub feature: usize,
    /// Grid values the feature was set to.
    pub grid: Vec<f64>,
    /// Marginalised prediction at each `grid` point.
    pub values: Vec<f64>,
}

/// One-way partial-dependence curve for one column of `x` on a fitted
/// prediction callback.
///
/// * `x` — the reference sample (rows = observations, cols = features).
/// * `feature` — the column index to vary.
/// * `grid` — values to set the feature to (typically `n_grid` evenly-
///   spaced quantiles of `x[:, feature]`).
/// * `predictor` — a closure that maps a feature matrix to a per-row
///   prediction vector. Usually `|xp| Ok(model.predict(xp)?)`.
///
/// The returned `values[i]` is the plain mean over rows of
/// `predictor(x with column `feature` set to grid[i])`.
pub fn partial_dependence<F>(
    x: ArrayView2<'_, f64>,
    feature: usize,
    grid: &[f64],
    predictor: F,
) -> Result<PartialDependence>
where
    F: Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(Error::Value(
            "partial_dependence: x must have at least one row and one column".into(),
        ));
    }
    if feature >= x.ncols() {
        return Err(Error::Value(format!(
            "partial_dependence: feature {feature} out of range for {} columns",
            x.ncols()
        )));
    }
    if grid.is_empty() {
        return Err(Error::Value(
            "partial_dependence: grid must be non-empty".into(),
        ));
    }
    let mut work = x.to_owned();
    let n = x.nrows();
    let mut values = Vec::with_capacity(grid.len());
    for &g in grid {
        {
            let mut col = work.column_mut(feature);
            for i in 0..n {
                col[i] = g;
            }
        }
        let preds = predictor(work.view())?;
        if preds.len() != n {
            return Err(Error::Shape(format!(
                "partial_dependence: predictor returned {} values for {n} rows",
                preds.len()
            )));
        }
        let mean: f64 = preds.iter().copied().sum::<f64>() / n as f64;
        if !mean.is_finite() {
            return Err(Error::Value(
                "partial_dependence: predictor returned non-finite values".into(),
            ));
        }
        values.push(mean);
    }
    Ok(PartialDependence {
        feature,
        grid: grid.to_vec(),
        values,
    })
}

/// One-way accumulated local effects (Apley & Zhu, 2020).
///
/// ALE removes the extrapolation and correlated-features bias that
/// partial dependence suffers from by:
///
/// 1. binning `x[:, feature]` into `n_bins` bins by quantile,
/// 2. within each bin, evaluating the model twice — with the feature
///    set to the bin's upper edge and to its lower edge — and averaging
///    the per-row *difference*,
/// 3. accumulating these local effects into a cumulative curve, and
/// 4. centering the result to have zero mean over the sample.
///
/// The returned `values[i]` is the ALE at the bin midpoint. ALE is the
/// preferred interpretability plot when features are correlated because
/// it never averages over combinations that never appear in the data.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct AccumulatedLocalEffects {
    /// Column of `x` the ALE was computed for.
    pub feature: usize,
    /// Bin edges (length `n_bins + 1`).
    pub edges: Vec<f64>,
    /// ALE curve values at each bin midpoint (length `n_bins`).
    pub values: Vec<f64>,
}

/// One-way ALE (Apley & Zhu, 2020) for a fitted prediction callback.
pub fn accumulated_local_effects<F>(
    x: ArrayView2<'_, f64>,
    feature: usize,
    n_bins: usize,
    predictor: F,
) -> Result<AccumulatedLocalEffects>
where
    F: Fn(ArrayView2<'_, f64>) -> Result<Array1<f64>>,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(Error::Value(
            "accumulated_local_effects: x must have at least one row and one column".into(),
        ));
    }
    if feature >= x.ncols() {
        return Err(Error::Value(format!(
            "accumulated_local_effects: feature {feature} out of range for {} columns",
            x.ncols()
        )));
    }
    if n_bins < 2 {
        return Err(Error::Value(format!(
            "accumulated_local_effects: n_bins must be ≥ 2 (got {n_bins})"
        )));
    }
    let n = x.nrows();
    let col = x.column(feature);
    // Quantile-spaced edges: unique quantiles at `k / n_bins`.
    let mut sorted: Vec<f64> = col.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut edges = Vec::with_capacity(n_bins + 1);
    for k in 0..=n_bins {
        let q = k as f64 / n_bins as f64;
        let h = (n as f64 - 1.0) * q;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = h - lo as f64;
        edges.push((1.0 - frac) * sorted[lo] + frac * sorted[hi]);
    }
    if edges.first() == edges.last() {
        return Err(Error::Value(
            "accumulated_local_effects: feature has zero range in the sample".into(),
        ));
    }
    // Assign rows to bins using the (possibly non-unique) edges: bin b covers
    // `[edges[b], edges[b + 1]]`. The last bin includes its right endpoint.
    let mut bin_of = vec![0usize; n];
    for i in 0..n {
        let v = col[i];
        let mut b = 0usize;
        for k in 0..n_bins {
            if v >= edges[k] {
                b = k;
            }
        }
        bin_of[i] = b;
    }
    let work = x.to_owned();
    let mut local = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    // For each bin, evaluate the model at the bin's upper and lower edges on the
    // subset of rows in that bin, and average the difference.
    for b in 0..n_bins {
        // Collect the rows in bin b.
        let rows: Vec<usize> = (0..n).filter(|&i| bin_of[i] == b).collect();
        if rows.is_empty() {
            continue;
        }
        let mut sub = Array2::<f64>::zeros((rows.len(), x.ncols()));
        for (ri, &row) in rows.iter().enumerate() {
            for j in 0..x.ncols() {
                sub[[ri, j]] = work[[row, j]];
            }
        }
        // Upper-edge prediction.
        for ri in 0..rows.len() {
            sub[[ri, feature]] = edges[b + 1];
        }
        let up = predictor(sub.view())?;
        for ri in 0..rows.len() {
            sub[[ri, feature]] = edges[b];
        }
        let lo = predictor(sub.view())?;
        if up.len() != rows.len() || lo.len() != rows.len() {
            return Err(Error::Shape(
                "accumulated_local_effects: predictor returned wrong-length vector".into(),
            ));
        }
        let mut diff = 0.0_f64;
        for ri in 0..rows.len() {
            diff += up[ri] - lo[ri];
        }
        local[b] = diff / rows.len() as f64;
        counts[b] = rows.len();
    }
    // Cumulative sum → uncentred curve.
    let mut acc = vec![0.0_f64; n_bins];
    let mut running = 0.0_f64;
    for b in 0..n_bins {
        running += local[b];
        acc[b] = running;
    }
    // Centre so that the sample-weighted mean of the ALE is zero.
    let total: usize = counts.iter().sum();
    let mut mean = 0.0_f64;
    if total > 0 {
        for b in 0..n_bins {
            mean += (counts[b] as f64 / total as f64) * acc[b];
        }
    }
    for v in acc.iter_mut() {
        *v -= mean;
    }
    Ok(AccumulatedLocalEffects {
        feature,
        edges,
        values: acc,
    })
}
