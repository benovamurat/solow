//! SequentialFeatureSelector — greedy forward / backward selection
//! wrapping a caller-supplied cross-validation score.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Direction for the greedy search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SfsDirection {
    /// Start from the empty set and add columns.
    Forward,
    /// Start from the full set and drop columns.
    Backward,
}

/// SequentialFeatureSelector.
pub struct SequentialFeatureSelector {
    /// Kept column indices, sorted ascending.
    pub selected: Vec<usize>,
    /// Direction used.
    pub direction: SfsDirection,
    /// Target size.
    pub n_features_to_select: usize,
}

impl SequentialFeatureSelector {
    /// Fit. `score` is a closure that accepts the selected sub-matrix
    /// and returns a scalar to *maximise* (e.g. `-mse` for regression).
    pub fn fit<F>(
        x: ArrayView2<'_, f64>,
        n_features_to_select: usize,
        direction: SfsDirection,
        mut score: F,
    ) -> Result<Self>
    where
        F: FnMut(ArrayView2<'_, f64>) -> Result<f64>,
    {
        let d = x.ncols();
        if n_features_to_select == 0 || n_features_to_select > d {
            return Err(Error::Value(format!(
                "SequentialFeatureSelector: n_features_to_select must be in [1, {d}] (got {n_features_to_select})"
            )));
        }
        let mut chosen: Vec<usize> = match direction {
            SfsDirection::Forward => Vec::new(),
            SfsDirection::Backward => (0..d).collect(),
        };
        let target = n_features_to_select;
        loop {
            match direction {
                SfsDirection::Forward => {
                    if chosen.len() >= target {
                        break;
                    }
                    let mut best = usize::MAX;
                    let mut best_score = f64::NEG_INFINITY;
                    for cand in 0..d {
                        if chosen.contains(&cand) {
                            continue;
                        }
                        let mut trial = chosen.clone();
                        trial.push(cand);
                        trial.sort();
                        let sub = subset_cols(x, &trial);
                        let s = score(sub.view())?;
                        if s > best_score {
                            best_score = s;
                            best = cand;
                        }
                    }
                    if best == usize::MAX {
                        break;
                    }
                    chosen.push(best);
                }
                SfsDirection::Backward => {
                    if chosen.len() <= target {
                        break;
                    }
                    let mut best = usize::MAX;
                    let mut best_score = f64::NEG_INFINITY;
                    for &cand in &chosen {
                        let trial: Vec<usize> =
                            chosen.iter().filter(|&&c| c != cand).copied().collect();
                        let sub = subset_cols(x, &trial);
                        let s = score(sub.view())?;
                        if s > best_score {
                            best_score = s;
                            best = cand;
                        }
                    }
                    chosen.retain(|&c| c != best);
                }
            }
        }
        chosen.sort();
        Ok(Self {
            selected: chosen,
            direction,
            n_features_to_select,
        })
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Ok(subset_cols(x, &self.selected))
    }
}

fn subset_cols(x: ArrayView2<'_, f64>, cols: &[usize]) -> Array2<f64> {
    let n = x.nrows();
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for i in 0..n {
        for (r, &c) in cols.iter().enumerate() {
            out[[i, r]] = x[[i, c]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sequential_forward_picks_the_top_scoring_column() {
        let x = array![
            [1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
        ];
        // Reward the column whose sum is largest.
        let sfs = SequentialFeatureSelector::fit(
            x.view(),
            1,
            SfsDirection::Forward,
            |sub| {
                let mut s = 0.0_f64;
                for v in sub.iter() {
                    s += v;
                }
                Ok(s)
            },
        ).unwrap();
        assert_eq!(sfs.selected, vec![2]);
    }
}
