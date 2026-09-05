//! HalvingGridSearchCV and HalvingRandomSearchCV — the successive-
//! halving hyperparameter search (Karnin-Koren-Somekh 2013; Li et al.
//! 2018 as adopted by the reference).
//!
//! Starts with a small budget over many candidate configurations, keeps
//! only the top `1/factor` after each round, and multiplies the budget
//! by `factor`. Converges when a single candidate remains or when the
//! full budget is reached.

use std::collections::BTreeMap;

use solow_core::{Error, Result};

use crate::search::{ParamGrid, SearchResult};

/// Configuration knobs shared by both halving searchers.
#[derive(Clone, Copy, Debug)]
pub struct HalvingConfig {
    /// The reduction factor between successive rounds (the reference default: 3).
    pub factor: usize,
    /// Minimum resource per candidate at round 0 (the reference default: 'exhaust').
    pub min_resources: usize,
    /// Maximum resource per candidate (also the training-set size).
    pub max_resources: usize,
}

impl HalvingConfig {
    /// Default with `factor = 3`.
    pub fn new(min_resources: usize, max_resources: usize) -> Result<Self> {
        if min_resources == 0 || max_resources < min_resources {
            return Err(Error::Value(
                "HalvingConfig: need 0 < min_resources ≤ max_resources".into(),
            ));
        }
        Ok(Self {
            factor: 3,
            min_resources,
            max_resources,
        })
    }
}

/// HalvingGridSearchCV.
#[derive(Clone, Debug)]
pub struct HalvingGridSearchCV;

impl HalvingGridSearchCV {
    /// Run successive halving over the grid.
    ///
    /// `score` is invoked as `score(resource_budget, &params) -> f64`. Higher
    /// scores are better.
    pub fn run<F>(grid: ParamGrid, config: HalvingConfig, mut score: F) -> Result<SearchResult>
    where
        F: FnMut(usize, &BTreeMap<String, f64>) -> Result<f64>,
    {
        let candidates = grid.enumerate();
        run_halving(candidates, config, |budget, p| score(budget, p))
    }
}

/// HalvingRandomSearchCV.
#[derive(Clone, Debug)]
pub struct HalvingRandomSearchCV;

impl HalvingRandomSearchCV {
    /// Run successive halving over `n_candidates` random draws.
    pub fn run<F>(
        grid: ParamGrid,
        n_candidates: usize,
        config: HalvingConfig,
        seed: u64,
        mut score: F,
    ) -> Result<SearchResult>
    where
        F: FnMut(usize, &BTreeMap<String, f64>) -> Result<f64>,
    {
        let full = grid.enumerate();
        if full.is_empty() {
            return Err(Error::Value("HalvingRandomSearchCV: empty grid".into()));
        }
        let mut state = seed.wrapping_add(0xC0DE_F00D);
        let mut chosen: Vec<BTreeMap<String, f64>> = Vec::with_capacity(n_candidates);
        for _ in 0..n_candidates {
            let idx = uniform_index(&mut state, full.len() as u64);
            chosen.push(full[idx].clone());
        }
        run_halving(chosen, config, |budget, p| score(budget, p))
    }
}

fn run_halving<F>(
    mut candidates: Vec<BTreeMap<String, f64>>,
    config: HalvingConfig,
    mut score: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, &BTreeMap<String, f64>) -> Result<f64>,
{
    if candidates.is_empty() {
        return Err(Error::Value("HalvingSearch: no candidates".into()));
    }
    let mut budget = config.min_resources.max(1);
    loop {
        let mut scored: Vec<(BTreeMap<String, f64>, f64)> = Vec::with_capacity(candidates.len());
        for p in &candidates {
            let s = score(budget, p)?;
            scored.push((p.clone(), s));
        }
        // Rank descending; keep top 1/factor (rounded up).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if scored.len() == 1 || budget >= config.max_resources {
            let (best_params, best_score) = scored.into_iter().next().unwrap();
            return Ok(SearchResult {
                params: best_params,
                mean_score: best_score,
                std_score: 0.0,
            });
        }
        let keep = (scored.len() + config.factor - 1) / config.factor;
        candidates = scored.into_iter().take(keep).map(|(p, _)| p).collect();
        budget = (budget * config.factor).min(config.max_resources);
    }
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

    #[test]
    fn halving_grid_selects_the_best_parameter_at_max_budget() {
        // Score is `p["c"]` — larger is better.
        let grid = ParamGrid::default().add("c", vec![0.1, 1.0, 10.0, 100.0]);
        let cfg = HalvingConfig::new(4, 32).unwrap();
        let r = HalvingGridSearchCV::run(grid, cfg, |_, p| Ok(p["c"])).unwrap();
        assert!((r.mean_score - 100.0).abs() < 1e-12);
    }
}
