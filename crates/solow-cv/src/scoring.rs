//! Higher-level helpers for scoring an estimator through a splitter.
//!
//! [`cross_val_score`] runs a `fit_and_score` callback across every fold a
//! [`Splitter`](crate::Splitter) produces and returns the per-fold scores.
//! Because the callback receives fold indices rather than a fitted model,
//! this helper composes with every Solow estimator and every custom scoring
//! function without the crate needing to know either type.

use solow_core::{Error, Result};

use crate::{Split, Splitter};

/// Per-fold scores plus their summary.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CrossValScores {
    /// One score per fold, in fold order.
    pub scores: Vec<f64>,
    /// Number of folds.
    pub n_folds: usize,
}

impl CrossValScores {
    /// Arithmetic mean of the per-fold scores.
    pub fn mean(&self) -> f64 {
        self.scores.iter().sum::<f64>() / self.n_folds as f64
    }

    /// Standard deviation of the per-fold scores (unbiased, `n - 1`).
    pub fn std(&self) -> f64 {
        if self.n_folds < 2 {
            return 0.0;
        }
        let m = self.mean();
        let s2: f64 =
            self.scores.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (self.n_folds as f64 - 1.0);
        s2.sqrt()
    }

    /// Standard error of the mean, `std() / sqrt(n_folds)`.
    pub fn standard_error(&self) -> f64 {
        self.std() / (self.n_folds as f64).sqrt()
    }
}

/// Run `fit_and_score` on every fold a `splitter` produces and collect the
/// per-fold scores.
///
/// `fit_and_score` takes the fold's train and test index slices and returns
/// a scalar score for that fold (typically a metric from
/// [`solow_metrics`](https://docs.rs/solow-metrics)). Any callback error is
/// propagated.
pub fn cross_val_score<S, F>(splitter: &S, n: usize, fit_and_score: F) -> Result<CrossValScores>
where
    S: Splitter,
    F: FnMut(&[usize], &[usize]) -> Result<f64>,
{
    let folds = splitter.split(n)?;
    fold_scores(&folds, fit_and_score)
}

/// Variant of [`cross_val_score`] that takes a pre-computed list of folds
/// directly — useful when the caller wants to share the same folds between
/// multiple scoring passes (e.g. hyperparameter sweeps).
pub fn cross_val_score_from_folds<F>(folds: &[Split], fit_and_score: F) -> Result<CrossValScores>
where
    F: FnMut(&[usize], &[usize]) -> Result<f64>,
{
    fold_scores(folds, fit_and_score)
}

/// Parallel counterpart of [`cross_val_score`] — evaluates every fold on a
/// rayon thread pool and returns the scores in fold order.
///
/// `fit_and_score` must be `Fn + Send + Sync` because it is invoked from
/// multiple worker threads concurrently. The scores are collected in fold
/// order regardless of the order in which the workers finish. Any callback
/// error is propagated; if two folds fail concurrently the returned error
/// is unspecified but always one of them.
///
/// Only available with the `parallel` cargo feature.
#[cfg(feature = "parallel")]
pub fn cross_val_score_parallel<S, F>(
    splitter: &S,
    n: usize,
    fit_and_score: F,
) -> Result<CrossValScores>
where
    S: Splitter,
    F: Fn(&[usize], &[usize]) -> Result<f64> + Send + Sync,
{
    use rayon::prelude::*;
    let folds = splitter.split(n)?;
    if folds.is_empty() {
        return Err(solow_core::Error::Value(
            "cross_val_score_parallel: splitter produced zero folds".into(),
        ));
    }
    let scores: Result<Vec<f64>> = folds
        .par_iter()
        .map(|fold| {
            let s = fit_and_score(&fold.train, &fold.test)?;
            if !s.is_finite() {
                return Err(solow_core::Error::Value(
                    "cross_val_score_parallel: fit_and_score returned a non-finite score".into(),
                ));
            }
            Ok(s)
        })
        .collect();
    let scores = scores?;
    let n_folds = scores.len();
    Ok(CrossValScores { scores, n_folds })
}

fn fold_scores<F>(folds: &[Split], mut fit_and_score: F) -> Result<CrossValScores>
where
    F: FnMut(&[usize], &[usize]) -> Result<f64>,
{
    if folds.is_empty() {
        return Err(Error::Value(
            "cross_val_score: splitter produced zero folds".into(),
        ));
    }
    let mut scores = Vec::with_capacity(folds.len());
    for fold in folds {
        let s = fit_and_score(&fold.train, &fold.test)?;
        if !s.is_finite() {
            return Err(Error::Value(
                "cross_val_score: fit_and_score returned a non-finite score".into(),
            ));
        }
        scores.push(s);
    }
    let n_folds = scores.len();
    Ok(CrossValScores { scores, n_folds })
}
