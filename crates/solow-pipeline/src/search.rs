//! Grid and randomised hyperparameter search over any Solow CV
//! splitter.
//!
//! Both searchers take a `score` callback that receives a fold's train
//! and test index slices *plus* the current parameter dictionary and
//! returns a scalar CV score. The callback is invoked
//! `n_folds · n_params` times for [`GridSearchCV`] and
//! `n_folds · n_iter` times for [`RandomizedSearchCV`]. The best
//! parameter set is the one with the highest mean CV score.

use std::collections::BTreeMap;

use solow_core::{Error, Result};
use solow_cv::Splitter;

/// One dimension of the search grid.
#[derive(Clone, Debug, PartialEq)]
struct GridAxis {
    name: String,
    values: Vec<f64>,
}

/// A grid of parameter values.
#[derive(Clone, Debug, Default)]
pub struct ParamGrid {
    axes: Vec<GridAxis>,
}

impl ParamGrid {
    /// Add a parameter dimension.
    pub fn add(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.axes.push(GridAxis {
            name: name.into(),
            values,
        });
        self
    }

    /// Enumerate every point in the grid (Cartesian product).
    pub fn enumerate(&self) -> Vec<BTreeMap<String, f64>> {
        let mut out: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new()];
        for axis in &self.axes {
            let mut next = Vec::with_capacity(out.len() * axis.values.len());
            for point in &out {
                for &v in &axis.values {
                    let mut p = point.clone();
                    p.insert(axis.name.clone(), v);
                    next.push(p);
                }
            }
            out = next;
        }
        out
    }
}

/// One row of a hyperparameter search's report.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Parameter dictionary.
    pub params: BTreeMap<String, f64>,
    /// Mean CV score across folds.
    pub mean_score: f64,
    /// Unbiased sample standard deviation across folds.
    pub std_score: f64,
}

// ---------------------------------------------------------------------------
// GridSearchCV
// ---------------------------------------------------------------------------

/// Full-factorial grid search over a `CV × ParamGrid` product.
#[derive(Clone, Debug)]
pub struct GridSearchCV {
    /// Every grid point evaluated, ordered by grid enumeration.
    pub cv_results: Vec<SearchResult>,
    /// Best-mean-score parameter dictionary.
    pub best_params: BTreeMap<String, f64>,
    /// Mean score of the best parameter dictionary.
    pub best_score: f64,
}

impl GridSearchCV {
    /// Run the search.
    pub fn run<S, F>(splitter: &S, n: usize, grid: ParamGrid, mut score: F) -> Result<Self>
    where
        S: Splitter,
        F: FnMut(&[usize], &[usize], &BTreeMap<String, f64>) -> Result<f64>,
    {
        let folds = splitter.split(n)?;
        if folds.is_empty() {
            return Err(Error::Value(
                "GridSearchCV: splitter produced zero folds".into(),
            ));
        }
        let points = grid.enumerate();
        if points.is_empty() {
            return Err(Error::Value(
                "GridSearchCV: parameter grid was empty".into(),
            ));
        }
        let mut results = Vec::with_capacity(points.len());
        for params in points {
            let mut scores = Vec::with_capacity(folds.len());
            for f in &folds {
                scores.push(score(&f.train, &f.test, &params)?);
            }
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let std_score = if scores.len() < 2 {
                0.0
            } else {
                let s2: f64 = scores.iter().map(|v| (v - mean_score).powi(2)).sum::<f64>()
                    / (scores.len() as f64 - 1.0);
                s2.sqrt()
            };
            results.push(SearchResult {
                params,
                mean_score,
                std_score,
            });
        }
        let best = results
            .iter()
            .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
            .cloned()
            .unwrap();
        Ok(Self {
            cv_results: results,
            best_params: best.params,
            best_score: best.mean_score,
        })
    }
}

// ---------------------------------------------------------------------------
// RandomizedSearchCV
// ---------------------------------------------------------------------------

/// Randomised search over the same `CV × BTreeMap` interface. Samples
/// `n_iter` parameter dictionaries by drawing each parameter
/// independently from a per-name sampler closure.
#[derive(Clone, Debug)]
pub struct RandomizedSearchCV {
    /// Every draw evaluated, in draw order.
    pub cv_results: Vec<SearchResult>,
    /// Best-mean-score draw.
    pub best_params: BTreeMap<String, f64>,
    /// Its mean score.
    pub best_score: f64,
}

impl RandomizedSearchCV {
    /// Run the search.
    ///
    /// * `samplers` — a list of `(name, boxed sampler)` where the
    ///   sampler takes a mutable PRNG state and returns a scalar.
    /// * `n_iter` — how many parameter dictionaries to draw.
    /// * `seed` — deterministic MMIX-LCG seed.
    pub fn run<S, F>(
        splitter: &S,
        n: usize,
        samplers: Vec<(String, Box<dyn Fn(&mut u64) -> f64>)>,
        n_iter: usize,
        seed: u64,
        mut score: F,
    ) -> Result<Self>
    where
        S: Splitter,
        F: FnMut(&[usize], &[usize], &BTreeMap<String, f64>) -> Result<f64>,
    {
        let folds = splitter.split(n)?;
        if folds.is_empty() {
            return Err(Error::Value(
                "RandomizedSearchCV: splitter produced zero folds".into(),
            ));
        }
        if n_iter == 0 {
            return Err(Error::Value(
                "RandomizedSearchCV: n_iter must be ≥ 1".into(),
            ));
        }
        let mut state = seed.wrapping_add(0x1357_9BDF_2468_ACE0);
        let mut results = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let mut params = BTreeMap::new();
            for (name, sampler) in &samplers {
                params.insert(name.clone(), sampler(&mut state));
            }
            let mut scores = Vec::with_capacity(folds.len());
            for f in &folds {
                scores.push(score(&f.train, &f.test, &params)?);
            }
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let std_score = if scores.len() < 2 {
                0.0
            } else {
                let s2: f64 = scores.iter().map(|v| (v - mean_score).powi(2)).sum::<f64>()
                    / (scores.len() as f64 - 1.0);
                s2.sqrt()
            };
            results.push(SearchResult {
                params,
                mean_score,
                std_score,
            });
        }
        let best = results
            .iter()
            .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
            .cloned()
            .unwrap();
        Ok(Self {
            cv_results: results,
            best_params: best.params,
            best_score: best.mean_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solow_cv::KFold;

    #[test]
    fn grid_search_picks_the_best_point() {
        let kf = KFold::new(3).unwrap();
        let grid = ParamGrid::default().add("alpha", vec![0.1, 1.0, 10.0]);
        // A monotonic score that increases in alpha — best = 10.0.
        let gs = GridSearchCV::run(&kf, 9, grid, |_train, _test, p| {
            Ok(*p.get("alpha").unwrap())
        })
        .unwrap();
        assert!((gs.best_score - 10.0).abs() < 1e-12);
        assert_eq!(gs.best_params["alpha"], 10.0);
        assert_eq!(gs.cv_results.len(), 3);
    }

    #[test]
    fn randomised_search_is_deterministic_by_seed() {
        let kf = KFold::new(3).unwrap();
        let samplers: Vec<(String, Box<dyn Fn(&mut u64) -> f64>)> = vec![(
            "alpha".to_string(),
            Box::new(|state: &mut u64| {
                // 53-bit uniform on [0, 1).
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (*state >> 11) as f64 / ((1u64 << 53) as f64)
            }),
        )];
        let a = RandomizedSearchCV::run(&kf, 9, samplers, 4, 42, |_t, _te, p| {
            Ok(*p.get("alpha").unwrap())
        })
        .unwrap();
        // Build a matching second sampler list — closures are stateless.
        let samplers2: Vec<(String, Box<dyn Fn(&mut u64) -> f64>)> = vec![(
            "alpha".to_string(),
            Box::new(|state: &mut u64| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (*state >> 11) as f64 / ((1u64 << 53) as f64)
            }),
        )];
        let b = RandomizedSearchCV::run(&kf, 9, samplers2, 4, 42, |_t, _te, p| {
            Ok(*p.get("alpha").unwrap())
        })
        .unwrap();
        assert_eq!(a.best_params, b.best_params);
    }
}
